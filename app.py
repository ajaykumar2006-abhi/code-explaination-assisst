"""Code Explanation Assistant — Streamlit entry point.

Run with:  python -m streamlit run app.py

Architecture: the UI calls :mod:`pipeline` (segmentation + coverage),
:mod:`polyglot`/`analyzer` (structural analysis), :mod:`explainer` (explanation
modes), :mod:`diagrams` (SVG visuals), :mod:`ai_client` (optional LLM +
contextual Q&A) and :mod:`exports` (Markdown/PDF). Source code is analysed,
never executed.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import streamlit as st

import ai_client
import diagrams
import explainer
import exports
import pipeline
import polyglot
from utils import (
    code_line_count,
    severity_badge,
    validate_source,
)

try:  # real code-editor component (ACE-based); falls back to a text area
    from code_editor import code_editor as _ace_editor

    HAVE_ACE = True
except Exception:  # pragma: no cover - depends on optional component
    HAVE_ACE = False

MAX_SOURCE_CHARS = 400_000
MAX_PROJECT_FILES = 25
MAX_PROJECT_FILE_CHARS = 200_000
SKIP_DIRS = {
    "node_modules", ".git", "venv", ".venv", "env", "__pycache__", "dist",
    "build", ".next", "target", "vendor", ".idea", ".vscode", "site-packages",
}

PAGES = ["Home", "Workspace", "Results", "Ask", "Diagrams", "Settings"]

EXAMPLES = {
    "Simple function": '''def average(values):
    """Return the mean of a list of numbers, or None when it is empty."""
    if not values:
        return None
    total = 0
    for value in values:
        total += value
    return total / len(values)
''',
    "Class with methods": '''class BankAccount:
    """A tiny account that refuses to go negative."""

    def __init__(self, owner, balance=0.0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            return False
        self.balance -= amount
        return True
''',
    "Java example": '''import java.util.List;

public class Inventory {
    private int items;

    public Inventory() {
        this.items = 0;
    }

    public void add(int count) {
        if (count < 0) {
            throw new IllegalArgumentException("count must be >= 0");
        }
        items += count;
    }

    public int total() {
        return items;
    }

    public static void main(String[] args) {
        Inventory inv = new Inventory();
        inv.add(5);
        System.out.println(inv.total());
    }
}
''',
    "Code with problems": '''import os
import json

def collect(items, cache=[]):
    try:
        for item in items:
            if item is None:
                continue
            cache.append(item)
    except:
        pass
    print("collected", len(cache))
    return cache
''',
}

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --ink: #0d1017;
    --panel: #151a24;
    --panel-2: #11151f;
    --line: #262e3d;
    --accent: #57d9a3;
    --accent-2: #7aa2f7;
    --text: #dde5f0;
    --muted: #8494a9;
}

.stApp { background: var(--ink); }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3, h4, h5, h6 { font-family: 'Inter', system-ui, sans-serif; letter-spacing: -0.01em; color: #f2f6fb; }
p, li, .stMarkdown { color: var(--text); }
code, pre, .stCode { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
a { color: var(--accent); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--panel-2); border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .stMarkdown p { font-size: .9rem; }

/* ---------- chrome ---------- */
.masthead {
    display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
    padding-bottom: 14px; border-bottom: 1px solid var(--line); margin-bottom: 18px;
}
.masthead .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent); }
.masthead .word { font-family: 'JetBrains Mono', monospace; font-weight: 600; letter-spacing: .18em; font-size: .86rem; color: #f2f6fb; }
.masthead .path { font-family: 'JetBrains Mono', monospace; font-size: .74rem; color: var(--muted); }
.chip {
    font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .08em;
    text-transform: uppercase; color: var(--muted); border: 1px solid var(--line);
    border-radius: 3px; padding: 3px 8px; background: var(--panel);
}
.chip.live { color: var(--accent); border-color: rgba(87,217,163,.45); }

.pane-label {
    font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .12em;
    text-transform: uppercase; color: var(--muted); margin: 0 0 8px 0;
}
.tabbar {
    display: flex; align-items: center; gap: 10px;
    background: var(--panel); border: 1px solid var(--line); border-bottom: none;
    border-radius: 6px 6px 0 0; padding: 9px 14px;
}
.tabbar .tab { font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: var(--text); }
.tabbar .lang { margin-left: auto; font-family: 'JetBrains Mono', monospace; font-size: .66rem;
    color: var(--accent); border: 1px solid rgba(87,217,163,.4); border-radius: 3px; padding: 2px 7px; }

/* ---------- controls ---------- */
.stButton > button {
    background: transparent; color: #cfd8e5; border: 1px solid var(--line);
    border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: .74rem;
    letter-spacing: .05em; padding: .45rem .9rem;
    transition: color .15s ease, border-color .15s ease, background .15s ease;
}
.stButton > button:hover, .stButton > button:focus-visible {
    color: var(--accent); border-color: var(--accent); background: rgba(87,217,163,.07);
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent); border-color: var(--accent); color: #0b0e13; font-weight: 600;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #7ff0c0; border-color: #7ff0c0; color: #0b0e13;
}
.stButton > button:disabled { color: #46506400; color: #4d5766; border-color: #232936; }

/* ---------- surfaces ---------- */
[data-testid="stVerticalBlockBorderWrapper"],
.stExpander {
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
}
[data-testid="stExpander"] summary { font-family: 'JetBrains Mono', monospace; font-size: .78rem; color: #cbd5e2; }
[data-testid="stExpander"] summary:hover { color: var(--accent); }
[data-testid="stMetricLabel"] p { font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #f2f6fb; }

div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    background: #10141d; color: var(--text); border-color: var(--line);
}

.tile-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 4px 0 18px 0; }
.tile { border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; background: var(--panel); }
.tile .k { font-family: 'JetBrains Mono', monospace; font-size: .64rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
.tile .v { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; color: #f2f6fb; margin-top: 3px; }
.tile .v.mint { color: var(--accent); }
.tile .v.blue { color: var(--accent-2); }

.hero-title { font-size: 2.5rem; line-height: 1.1; font-weight: 700; margin: 0 0 .7rem 0; color: #f6f9fd; }
.hero-sub { font-size: 1rem; color: #aeb9c8; max-width: 62ch; line-height: 1.65; }
.rule { height: 1px; background: var(--line); margin: 22px 0; }
.note { border-left: 2px solid var(--accent); padding: 2px 0 2px 14px; color: #b9c5d4; font-size: .9rem; line-height: 1.6; }
.feature-card { border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; background: var(--panel); height: 100%; }
.feature-card h4 { margin: 0 0 6px 0; font-size: .95rem; }
.feature-card p { margin: 0; font-size: .82rem; color: var(--muted); line-height: 1.55; }

.chat-msg-user { background: #1b2740; border: 1px solid #2c3a5e; border-radius: 8px; padding: 10px 14px; margin: 6px 0; }
.chat-msg-ai { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; margin: 6px 0; }

@media (max-width: 900px) {
    .tile-row { grid-template-columns: repeat(2, 1fr); }
    .hero-title { font-size: 1.9rem; }
}
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def init_state() -> None:
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("source", "")
    st.session_state.setdefault("filename", "analysis.py")
    st.session_state.setdefault("language", "auto")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("mode", "detailed")
    st.session_state.setdefault("project", None)   # {filename: {source, result}}
    st.session_state.setdefault("chat", [])        # [{role, text, provider}]


def resolved_language(source: str, filename: str, choice: str) -> str:
    if choice != "auto":
        return choice
    return polyglot.detect_language(source, filename)


def run_analysis(source: str, filename: str, language_choice: str) -> dict[str, Any]:
    lang = resolved_language(source, filename, language_choice)
    return pipeline.analyze_source(source, lang, filename), lang


# ---------------------------------------------------------------------------
# Chrome
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


def render_masthead() -> None:
    st.markdown(
        """
        <div class="masthead">
            <span class="dot"></span>
            <span class="word">CODE EXPLAINER</span>
            <span class="path">~/workspace</span>
            <span class="chip live">multi-language</span>
            <span class="chip">never executed</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

def page_home() -> None:
    st.markdown('<div class="hero-title">Understand any codebase.<br>Line by line.</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Paste a file or upload a project. Get structural analysis, '
        "explanations in eight modes, complexity metrics, diagrams, and a contextual "
        "assistant — all without executing a single line of your code.</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    cols = st.columns(3, gap="medium")
    cards = [
        ("Structural analysis", "AST parsing for Python; declaration-level heuristics for 15+ other languages."),
        ("Large-file pipeline", "Definition-boundary segmentation with honest coverage reporting — nothing silently skipped."),
        ("Explanation modes", "Simple, detailed, line-by-line, function/class, algorithm, flow and technical views."),
        ("Visual diagrams", "SVG module maps, class diagrams and candidate call graphs generated from real structure."),
        ("Ask assistant", "Contextual Q&A grounded in the analysis — LLM-backed when a key is configured."),
        ("Exports", "Download Markdown, PDF or JSON reports of any analysis."),
    ]
    for col, (title, body) in zip(cols * 2, cards):
        col.markdown(
            f'<div class="feature-card"><h4>{title}</h4><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 3])
    if left.button("Open workspace →", type="primary"):
        st.session_state.page = "Workspace"
        st.rerun()
    right.caption(
        "Everything runs locally. Python files get full AST analysis; other languages "
        "get clearly-labelled heuristic analysis. No API key is required."
    )


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

def _load_uploads(uploaded) -> None:
    """Store uploaded files into session project state."""
    project = st.session_state.setdefault("project", {}) or {}
    for up in uploaded:
        data = up.getvalue()
        if up.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        parts = info.filename.replace("\\", "/").split("/")
                        if any(p in SKIP_DIRS for p in parts):
                            continue
                        if len(project) >= MAX_PROJECT_FILES:
                            st.warning(f"Project limit reached ({MAX_PROJECT_FILES} files); "
                                       f"skipped {info.filename}")
                            break
                        if info.file_size > MAX_PROJECT_FILE_CHARS:
                            st.warning(f"Skipped {info.filename}: over "
                                       f"{MAX_PROJECT_FILE_CHARS // 1000} KB")
                            continue
                        try:
                            text = zf.read(info).decode("utf-8", errors="replace")
                        except Exception:
                            continue
                        project[info.filename] = {"source": text, "result": None}
            except zipfile.BadZipFile:
                st.error(f"{up.name} is not a valid zip archive.")
        else:
            if len(data) > MAX_PROJECT_FILE_CHARS:
                st.warning(f"Skipped {up.name}: over {MAX_PROJECT_FILE_CHARS // 1000} KB")
                continue
            project[up.name] = {
                "source": data.decode("utf-8", errors="replace"),
                "result": None,
            }
    st.session_state.project = project


def _editor_widget(source: str, language: str, key: str) -> str:
    if HAVE_ACE:
        try:
            response = _ace_editor(
                value=source,
                language=language if language != "auto" else "text",
                theme="github_dark",
                height=460,
                key=key,
                auto_update=True,
                wrap=False,
            )
            return response.get("text", source)
        except Exception:
            pass
    return st.text_area(
        "Source code", value=source, height=460,
        help="Paste code here. Up to 400,000 characters. Never executed.",
        label_visibility="collapsed",
    )


def page_workspace() -> None:
    st.markdown('<p class="pane-label">Workspace — editor & inputs</p>',
                unsafe_allow_html=True)

    # --- project upload row -------------------------------------------------
    up_col, lang_col = st.columns([2, 1])
    with up_col:
        uploaded = st.file_uploader(
            "Upload source file(s) or a .zip project",
            type=["py", "java", "js", "ts", "c", "h", "cpp", "cc", "hpp", "cs",
                  "php", "go", "rs", "kt", "swift", "rb", "sh", "sql", "html",
                  "css", "txt", "zip", "json"],
            accept_multiple_files=True,
            help="Archives are filtered: node_modules, venvs, caches and binaries are skipped.",
        )
    if uploaded:
        _load_uploads(uploaded)

    project = st.session_state.get("project") or {}
    if project:
        with st.expander(f"Project files ({len(project)}) — click to load into editor",
                         expanded=False):
            choice = st.selectbox(
                "Select a file",
                ["— choose —"] + sorted(project),
                key="project_pick",
                label_visibility="collapsed",
            )
            if choice != "— choose":
                entry = project[choice]
                st.session_state.source = entry["source"]
                st.session_state.filename = choice
                st.rerun()
        if st.button("Analyse all project files", key="analyse_project"):
            _analyse_project()

    # --- editor row ---------------------------------------------------------
    ex_col, lang_col2, fname_col = st.columns([2.2, 1, 1])
    with ex_col:
        st.markdown(
            '<div class="tabbar"><span class="tab">source</span>'
            '<span class="lang">EDITOR</span></div>',
            unsafe_allow_html=True,
        )
        source = _editor_widget(
            st.session_state.source or EXAMPLES["Simple function"],
            st.session_state.language if st.session_state.language != "auto" else "python",
            key="main_editor",
        )
        st.session_state.source = source

    with lang_col2:
        lang_choice = st.selectbox(
            "Language",
            ["auto"] + polyglot.list_languages(),
            index=0,
            format_func=lambda x: "Auto-detect" if x == "auto" else polyglot.language_name(x),
            key="lang_select",
        )
        st.session_state.language = lang_choice

    with fname_col:
        st.text_input("Filename", value=st.session_state.filename,
                      key="fname_input", help="Used for language detection and reports.")
        st.session_state.filename = st.session_state.get("fname_input", st.session_state.filename)

    lines = code_line_count(source)
    st.caption(f"{lines} lines · {len(source):,} characters · "
               f"detected: {polyglot.language_name(resolved_language(source, st.session_state.filename, lang_choice))}")

    if st.session_state.error:
        st.error(st.session_state.error)
    error = validate_source(source, MAX_SOURCE_CHARS)
    if error:
        st.error(error)

    btn_col, hint_col = st.columns([1, 3])
    if btn_col.button("Analyse code", type="primary", disabled=bool(error)):
        _run_and_store(source)

    with st.expander("Examples"):
        cols = st.columns(4)
        for col, (label, snippet) in zip(cols, EXAMPLES.items()):
            if col.button(label, key=f"ex_{label}"):
                st.session_state.source = snippet
                st.session_state.filename = "example." + {
                    "Java example": "java",
                }.get(label, "py")
                st.session_state.result = None
                st.rerun()

    hint_col.caption(
        "Analysis is structural only — the code is never executed. Python gets full "
        "AST analysis; other languages get labelled heuristic analysis."
    )


def _run_and_store(source: str) -> None:
    filename = st.session_state.get("fname_input") or st.session_state.filename
    st.session_state.filename = filename
    with st.status("Analysing source…", expanded=True) as status:
        try:
            status.update(label="Running structural analysis", state="running")
            result, lang = run_analysis(source, filename, st.session_state.language)
            st.session_state.result = result
            st.session_state.error = None
            status.update(label=f"Analysis complete — {lang}", state="complete")
        except SyntaxError as exc:
            line = getattr(exc, "lineno", None)
            where = f"Line {line}: " if line else ""
            st.session_state.error = f"{where}{getattr(exc, 'msg', exc)}"
            st.session_state.result = None
            st.rerun()
        except Exception as exc:  # defensive: never crash the UI on odd input
            st.session_state.error = f"Analysis failed: {exc}"
            st.session_state.result = None
            st.rerun()
    st.session_state.page = "Results"
    st.rerun()


def _analyse_project() -> None:
    project = st.session_state.get("project") or {}
    analysed = 0
    for name, entry in project.items():
        if entry.get("result"):
            analysed += 1
            continue
        source = entry["source"]
        if len(source) > MAX_SOURCE_CHARS:
            entry["error"] = "file too large"
            continue
        lang = polyglot.detect_language(source, name)
        try:
            entry["result"] = pipeline.analyze_source(source, lang, name)
            entry["language"] = lang
            analysed += 1
        except Exception as exc:
            entry["error"] = str(exc)
    st.session_state.project = project
    st.success(f"Analysed {analysed} project file(s). Select one to view results, "
               "or open Results for the active file.")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def page_results() -> None:
    result = st.session_state.result
    if not result:
        st.warning("Nothing analysed yet — open the Workspace and run an analysis.")
        if st.button("Go to workspace"):
            st.session_state.page = "Workspace"
            st.rerun()
        return

    source = st.session_state.source
    s = result["summary"]
    total_issues = sum(result["issue_counts"].values())

    st.markdown(
        f"""<div class="tile-row">
            <div class="tile"><div class="k">Language</div><div class="v">{s.get('language','?')}</div></div>
            <div class="tile"><div class="k">Lines</div><div class="v">{s['total_lines']}</div></div>
            <div class="tile"><div class="k">Functions</div><div class="v blue">{s['functions']}</div></div>
            <div class="tile"><div class="k">Findings</div><div class="v {'mint' if total_issues else ''}">{total_issues}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    cov = result.get("coverage")
    if cov and cov.get("percent", 100) < 100:
        st.warning(
            f"Coverage: {cov['percent']}% of lines analysed "
            f"({cov['segments_analyzed']}/{cov['segments_total']} segments). "
            "Remaining ranges are listed below — no section was silently skipped."
        )

    mode_col, exp_col = st.columns([1, 2])
    mode = mode_col.selectbox(
        "Explanation mode",
        list(explainer.MODES),
        format_func=lambda m: explainer.MODES[m],
        key="mode_select",
    )
    st.session_state.mode = mode

    with st.spinner("Generating explanation…"):
        explanation_md = explainer.explain(result, source, mode, s.get("language", ""))
    st.session_state.explanation_md = explanation_md

    tab_src, tab_report = st.tabs(["Source & Structure", "Explanation"])

    with tab_src:
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown(
                f'<div class="tabbar"><span class="tab">{s["filename"]}</span>'
                f'<span class="lang">{s["total_lines"]} LINES</span></div>',
                unsafe_allow_html=True,
            )
            st.code(source[:60000], language="python" if s.get("analysis_kind") == "ast" else "text",
                    line_numbers=True)
            if len(source) > 60000:
                st.caption(f"Showing first 60,000 of {len(source):,} characters.")
            with st.container(border=True):
                st.markdown('<p class="pane-label">Module map</p>', unsafe_allow_html=True)
                st.code(result["structure"], language="text")
        with right:
            if result["imports"]:
                with st.container(border=True):
                    st.markdown('<p class="pane-label">Imports</p>', unsafe_allow_html=True)
                    st.dataframe(
                        [{
                            "module": i["module"], "bound as": i["bound"],
                            "kind": "stdlib" if i.get("stdlib") else "external",
                            "used": i.get("used", True), "line": i["line"],
                        } for i in result["imports"]],
                        hide_index=True, width="stretch",
                    )
            if result.get("segment_details"):
                with st.container(border=True):
                    st.markdown('<p class="pane-label">Segment coverage</p>',
                                unsafe_allow_html=True)
                    st.dataframe(
                        [{
                            "segment": f"{d['kind']} {d['name']}",
                            "lines": f"{d['start_line']}–{d['end_line']}",
                            "status": "analyzed" if d["analyzed"] else f"skipped: {d['skip_reason']}",
                        } for d in result["segment_details"][:60]],
                        hide_index=True, width="stretch",
                    )

    with tab_report:
        search = st.text_input("Search within explanation", key="expl_search",
                               placeholder="Filter lines containing this text…")
        body = explanation_md
        if search:
            body = "\n".join(
                line for line in explanation_md.splitlines()
                if search.lower() in line.lower()
            ) or "*No lines matched the search.*"
        st.markdown(body)

    # Findings table
    st.markdown('<p class="pane-label">Findings</p>', unsafe_allow_html=True)
    if result["issues"]:
        counts = result["issue_counts"]
        st.caption(f"{counts['error']} error(s) · {counts['warning']} warning(s) · "
                   f"{counts['info']} note(s)")
        st.dataframe(
            [{
                "severity": severity_badge(i["severity"]),
                "line": i["line"], "rule": i["code"], "finding": i["message"],
            } for i in result["issues"]],
            hide_index=True, width="stretch",
        )
    else:
        st.success("No rule-based findings for this source.")

    # Exports + navigation
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    ex1, ex2, ex3, ex4, spacer = st.columns([1, 1, 1, 1, 2])
    full_md = explainer.explain_all(result, source, s.get("language", ""))
    ex1.download_button("Markdown", data=exports.to_markdown(
        result, source, full_md, s["filename"]),
        file_name=f"{s['filename']}.explanation.md", mime="text/markdown")
    ex2.download_button("PDF", data=exports.to_pdf(result, full_md, s["filename"]),
                        file_name=f"{s['filename']}.explanation.pdf",
                        mime="application/pdf")
    ex3.download_button("JSON", data=json.dumps(result, indent=2, default=str),
                        file_name=f"{s['filename']}.analysis.json",
                        mime="application/json")
    if ex4.button("Ask about this code →", key="goto_ask"):
        st.session_state.page = "Ask"
        st.rerun()
    if st.button("New analysis"):
        st.session_state.source = ""
        st.session_state.result = None
        st.session_state.chat = []
        st.session_state.page = "Workspace"
        st.rerun()


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

def page_ask() -> None:
    st.markdown('<p class="pane-label">Ask — contextual assistant</p>',
                unsafe_allow_html=True)
    result = st.session_state.result
    if not result:
        st.warning("Analyse some code first — the assistant answers questions "
                   "about the current analysis.")
        if st.button("Go to workspace"):
            st.session_state.page = "Workspace"
            st.rerun()
        return

    s = result["summary"]
    if ai_client.provider_configured():
        st.caption("LLM provider configured — answers are model-generated with "
                   "the analysis as context.")
    else:
        st.caption("No `OPENAI_API_KEY` set — using the built-in contextual "
                   "assistant, which answers strictly from analysis facts.")

    for msg in st.session_state.chat:
        cls = "chat-msg-user" if msg["role"] == "user" else "chat-msg-ai"
        who = "You" if msg["role"] == "user" else f"Assistant · {msg.get('provider','')}"
        st.markdown(
            f'<div class="{cls}"><b>{who}</b><br>{msg["text"]}</div>',
            unsafe_allow_html=True,
        )

    prompt = st.chat_input("Ask about this code…")
    if prompt:
        st.session_state.chat.append({"role": "user", "text": prompt})
        try:
            answer, provider = ai_client.ask(
                prompt, result, st.session_state.source, s.get("language", ""))
        except Exception as exc:
            answer, provider = f"Assistant error: {exc}", "error"
        st.session_state.chat.append({"role": "assistant", "text": answer,
                                      "provider": provider})
        st.rerun()

    if st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

def page_diagrams() -> None:
    st.markdown('<p class="pane-label">Diagrams — generated from analysis</p>',
                unsafe_allow_html=True)
    result = st.session_state.result
    if not result:
        st.warning("Analyse some code first — diagrams are drawn from the "
                   "detected structure.")
        if st.button("Go to workspace"):
            st.session_state.page = "Workspace"
            st.rerun()
        return

    choice = st.radio(
        "Diagram type",
        ["Module structure", "Class diagram", "Call graph (candidate)"],
        horizontal=True, key="diagram_pick",
    )
    source = st.session_state.source
    lang = result["summary"].get("language", "")
    lang_key = "python" if result["summary"].get("analysis_kind") == "ast" else lang.lower()

    if choice == "Module structure":
        svg = diagrams.structure_svg(result)
        fname = "module-structure.svg"
    elif choice == "Class diagram":
        svg = diagrams.class_svg(result)
        fname = "class-diagram.svg"
    else:
        svg = diagrams.call_svg(result, source, lang_key)
        fname = "call-graph.svg"

    import streamlit.components.v1 as components
    components.html(svg, height=560, scrolling=True)

    st.download_button("Download SVG", data=svg.encode("utf-8"),
                       file_name=fname, mime="image/svg+xml")
    st.caption(
        "Diagrams show only what the analysis detected. The call graph is a "
        "*candidate* graph: it links definitions whose names are invoked inside "
        "other definitions — dynamic dispatch and imports are not resolved."
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def page_settings() -> None:
    st.markdown('<p class="pane-label">Settings & about</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**AI provider**")
        configured = ai_client.provider_configured()
        st.metric("LLM configured", "Yes" if configured else "No")
        st.caption(
            "Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `AI_MODEL`) "
            "in the environment to enable model-generated answers in the Ask page. "
            "Without a key the built-in contextual assistant is used. Keys are "
            "never stored in the repo or sent to the browser."
        )
        st.code(
            "OPENAI_API_KEY=sk-...\nOPENAI_BASE_URL=https://api.openai.com/v1\nAI_MODEL=gpt-4o-mini",
            language="bash",
        )

    with st.container(border=True):
        st.markdown("**Limits**")
        st.caption(
            f"Single source: {MAX_SOURCE_CHARS:,} characters · "
            f"Project: {MAX_PROJECT_FILES} files, {MAX_PROJECT_FILE_CHARS // 1000} KB each · "
            "Archives skip venvs, node_modules, caches and binaries."
        )

    with st.container(border=True):
        st.markdown("**What the analysis is — and is not**")
        st.markdown(
            "- Python files: full AST parsing (definitions, calls, complexity, findings).\n"
            "- Other languages: declaration-level heuristics, clearly labelled.\n"
            "- Source code is **never executed** on this machine or any server.\n"
            "- Complexity numbers are structural counts, not measured runtime cost.\n"
            "- The assistant answers from analysis facts unless an LLM key is set."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Code Explanation Assistant", page_icon="◆",
                       layout="wide")
    inject_css()
    init_state()

    with st.sidebar:
        st.markdown("**CODE EXPLAINER**")
        st.caption("AI-assisted code understanding")
        st.markdown('<div class="rule" style="margin:10px 0"></div>',
                    unsafe_allow_html=True)
        for p in PAGES:
            if st.button(p, key=f"nav_{p}",
                         type="primary" if st.session_state.page == p else "secondary",
                         width="stretch"):
                st.session_state.page = p
                st.rerun()
        st.markdown('<div class="rule" style="margin:10px 0"></div>',
                    unsafe_allow_html=True)
        result = st.session_state.result
        if result:
            s = result["summary"]
            st.caption(f"Active: `{s['filename']}` — {s['total_lines']} lines")
        st.caption("Static analysis · nothing executed")

    render_masthead()
    page = st.session_state.page
    {"Home": page_home, "Workspace": page_workspace, "Results": page_results,
     "Ask": page_ask, "Diagrams": page_diagrams, "Settings": page_settings}[page]()


if __name__ == "__main__":
    main()
