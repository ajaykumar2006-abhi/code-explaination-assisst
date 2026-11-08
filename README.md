# Advanced Code Explanation Assistant

An AI-assisted code understanding platform: paste a file or upload a project,
get structural analysis, eight explanation modes, complexity metrics, SVG
diagrams, a contextual Q&A assistant and exportable reports — **without ever
executing the submitted code**.

## Problem statement

Reading unfamiliar code — especially large files — is slow. Developers and
students need a tool that explains *what a program contains and how it is put
together*, not just a syntax highlighter. This project provides deterministic
structural analysis plus optional LLM-backed conversation, with honest coverage
reporting for large sources.

## Main features

- **Multi-language input** — Python gets full AST analysis (`ast` module);
  Java, JavaScript, TypeScript, C, C++, C#, PHP, Go, Rust, Kotlin, Swift,
  Ruby, Bash, SQL, HTML and CSS get clearly-labelled heuristic analysis.
- **Large-code pipeline** — sources are segmented at definition boundaries
  (functions, classes, module header), each segment is analysed, and a
  coverage report states exactly which line ranges were examined. Skipped or
  summarised ranges are listed with reasons — nothing is silently dropped.
- **Eight explanation modes** — Simple, Detailed, Line-by-Line, Function-by-
  Function, Class-by-Class, Algorithm & Complexity, Execution Flow, and
  Advanced Technical. All are generated strictly from analysis facts.
- **Static findings** — unused imports, mutable defaults, bare/swallowed
  exceptions, missing docstrings, long functions/lines, TODO markers and
  complexity hotspots (Python rules are AST-based; other languages heuristic).
- **Visual diagrams** — SVG module maps, class diagrams and a *candidate* call
  graph, generated only from detected structure and downloadable as SVG.
- **Ask assistant** — contextual Q&A about the current analysis. With
  `OPENAI_API_KEY` set it uses an OpenAI-compatible chat endpoint; without a
  key a deterministic assistant answers from the same analysis facts.
- **Project upload** — multiple files or a `.zip`; archives are filtered
  (`node_modules`, venvs, caches, binaries skipped) with per-file size limits.
- **Exports** — Markdown, PDF (reportlab) and JSON reports.
- **Premium dark UI** — sidebar navigation, code-editor component (ACE via
  `streamlit-code-editor`, text-area fallback), search inside explanations,
  collapsible findings, responsive layout.

## Technology stack

- Python 3.10+ (developed on 3.13)
- Streamlit 1.66 + `streamlit-code-editor` (optional editor component)
- `ast`, `re`, `zipfile`, `urllib` (stdlib only for analysis)
- `reportlab` for PDF export
- Optional OpenAI-compatible HTTP API for the Ask page
- pytest + `streamlit.testing.v1.AppTest` for tests

## Architecture

```
app.py            Streamlit UI: Home / Workspace / Results / Ask / Diagrams / Settings
analyzer.py       Python AST analysis (definitions, complexity, findings)
polyglot.py       Language detection + heuristic analysis for non-Python
pipeline.py       Segmentation, per-segment analysis, coverage reporting
explainer.py      Explanation-mode renderers (Markdown from analysis facts)
diagrams.py       SVG diagram generation (structure, classes, call graph)
ai_client.py      Optional LLM provider + deterministic contextual Q&A
exports.py        Markdown + PDF export
tests.py          pytest suite (unit + AppTest UI tests)
utils.py          Validation and display helpers
```

## Installation (Windows)

```powershell
cd 07_code_explanation_assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux: `source .venv/bin/activate` instead of the Activate line.

## Configuration

Copy `.env.example` values into your environment **only if** you want
model-generated answers:

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:AI_MODEL="gpt-4o-mini"          # optional
$env:OPENAI_BASE_URL="https://api.openai.com/v1"   # optional
```

No key is required — the app works fully offline with the built-in assistant.

## Run

```powershell
python -m streamlit run app.py
```

Open http://localhost:8501. If the port is busy, pass `--server.port 8511`.

## Usage

1. **Workspace** — paste code, upload files/a zip, pick a language (or Auto),
   press **Analyse code**.
2. **Results** — choose an explanation mode, search the explanation, review
   findings and segment coverage, then export Markdown/PDF/JSON.
3. **Ask** — ask contextual questions about the analysed code.
4. **Diagrams** — view/download SVG module, class and call-graph diagrams.
5. **Settings** — provider status, limits and scope notes.

## Supported languages

Python (AST), Java, JavaScript, TypeScript, C, C++, C#, PHP, Go, Rust, Kotlin,
Swift, Ruby, Bash, SQL, HTML, CSS — plus plain-text fallback.

## Large-code approach

The file is split at AST/regex definition boundaries into segments
(module header, classes, functions, top-level statements, tail). Each segment
is analysed independently for findings, and `pipeline.coverage` records how
many lines were covered. Segments larger than 400 lines are structurally
summarised and marked accordingly. The UI shows the coverage percentage and
any skipped ranges — analysis completeness is never overstated.

## Testing

```powershell
python -m pytest tests.py -q
```

57 tests: AST analysis rules, heuristic language analysis, segmentation and
coverage, every explanation mode, SVG generation, the Q&A assistant, exports,
and Streamlit UI flows via AppTest.

## Known limitations

- Heuristic (non-Python) analysis finds declarations and approximates
  complexity; it is not a grammar parse and can miss edge cases.
- The call graph is *candidate*-level: it links definitions by name and does
  not resolve dynamic dispatch or cross-file calls.
- Line-by-line annotations are structural (statement kind), not semantic
  translation.
- PDF export is a plain-text render of the report, not a styled document.
- Multi-file projects analyse each file independently; there is no cross-file
  type resolution or unified call graph yet.
- Without an API key, Ask answers are deterministic lookups over the analysis,
  not free-form generated prose.

## License

MIT — see repository for details.
