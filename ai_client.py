"""AI layer: optional LLM provider + deterministic contextual Q&A.

Two paths exist:

1. If ``OPENAI_API_KEY`` (or ``AI_API_KEY``) is set, questions are sent to an
   OpenAI-compatible chat endpoint with the analysis result as context. The
   key is read from the environment only — never from the frontend.
2. Otherwise a deterministic assistant answers from the analysis data itself:
   function lookups, complexity questions, import questions, "explain in
   simple terms", etc. These answers are real — they quote the same facts the
   analyzer found — but they are *not* generated prose.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

import polyglot


def provider_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("AI_API_KEY"))


def _endpoint() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _model() -> str:
    return os.environ.get("AI_MODEL", "gpt-4o-mini")


def ask_llm(question: str, context: dict[str, Any]) -> tuple[str, str]:
    """Send the question to the configured provider.

    Returns (answer_text, provider_label). Raises on failure so the caller can
    fall back to the deterministic assistant.
    """
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("AI_API_KEY")
    if not key:
        raise RuntimeError("No API key configured")

    # Compact context: structure + summaries, not the whole source.
    payload_context = {
        "summary": context.get("summary"),
        "structure": context.get("structure"),
        "functions": [
            {k: f.get(k) for k in ("name", "line", "end_line", "arguments",
                                   "complexity", "returns", "walkthrough")}
            for f in context.get("functions", [])[:30]
        ],
        "classes": [
            {"name": c["name"], "line": c["line"], "methods": [m["name"] for m in c["methods"]]}
            for c in context.get("classes", [])[:20]
        ],
        "issues": context.get("issues", [])[:30],
    }
    system = (
        "You are a code explanation assistant. Answer questions about the "
        "provided code analysis only. Be concise and accurate; say when the "
        "analysis data does not contain the answer."
    )
    body = json.dumps({
        "model": _model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content":
                f"Analysis data:\n{json.dumps(payload_context, indent=1)[:12000]}\n\n"
                f"Question: {question}"},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }).encode()

    request = urllib.request.Request(
        f"{_endpoint()}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode())
        text = data["choices"][0]["message"]["content"].strip()
        return text, f"{_model()} via {_endpoint()}"
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Provider request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Deterministic contextual assistant
# ---------------------------------------------------------------------------

def _find_function(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    for f in result.get("functions", []):
        if f["name"].lower() == name.lower():
            return f
    for c in result.get("classes", []):
        for m in c["methods"]:
            if m["name"].lower() == name.lower():
                item = dict(m)
                item["_class"] = c["name"]
                return item
    return None


def answer_deterministic(question: str, result: dict[str, Any],
                         source: str, language: str) -> tuple[str, str]:
    """Answer from analysis facts. Returns (answer, provider_label)."""
    q = question.lower().strip()
    s = result.get("summary", {})
    label = "rule-based assistant (no LLM configured)"

    # Explicit function lookup: "explain function X", "what does method X do"
    stopwords = {"is", "the", "a", "an", "this", "that", "most", "which",
                 "does", "do", "it", "its", "my", "your", "our"}
    m = re.search(r"(?:function|method|def|about)\s+[`'\"]?(\w+)", q)
    fname = m.group(1) if m else None
    if fname in stopwords:
        fname = None
    if fname:
        f = _find_function(result, fname)
        if f:
            where = f" method of class `{f['_class']}`" if "_class" in f else ""
            args = ", ".join(
                a["name"] + (f"={a['default']}" if a.get("default") else "")
                for a in f.get("arguments", [])
            ) or "no parameters"
            lines = [
                f"**`{f['name']}({args})`**{where} — lines {f['line']}–{f['end_line']}, "
                f"complexity {f['complexity']}.",
                "",
                "What the structure shows:",
            ]
            for step in (f.get("walkthrough") or [])[:8]:
                lines.append(f"- {step}")
            if f.get("returns"):
                lines.append(f"- Returns: {', '.join('`' + r + '`' for r in f['returns'][:5])}")
            lines.append("")
            lines.append(
                "*This describes the syntax, not the author's intent.*"
            )
            return "\n".join(lines), label
        return (f"I could not find a function or method named `{fname}` in the "
                "analysis. The detected names are: "
                + ", ".join(f"`{f['name']}`" for f in result.get("functions", [])[:15])
                + ".", label)

    if re.search(r"\b(complex|cyclomatic|difficulty|hard)\b", q):
        funcs = result.get("functions", []) + [
            m for c in result.get("classes", []) for m in c["methods"]
        ]
        if funcs:
            top = sorted(funcs, key=lambda f: -f["complexity"])[:5]
            body = "\n".join(
                f"- `{f['name']}` — complexity {f['complexity']}" for f in top
            )
            return (f"Complexity is a structural count (branches, loops, handlers). "
                    f"The highest values:\n{body}\n\n"
                    "Higher means more paths to test; it is not a runtime measure.",
                    label)
        return "No functions were detected, so no complexity figures exist.", label

    if re.search(r"\b(import|depend|module|library|package)\b", q):
        imports = result.get("imports", [])
        if not imports:
            return "No imports were detected in this source.", label
        rows = "\n".join(
            f"- `{i['module']}` (line {i['line']})"
            + (" — stdlib" if i.get("stdlib") else "")
            for i in imports[:20]
        )
        return f"Detected imports/dependencies:\n{rows}", label

    if re.search(r"\b(class|inherit|object|method)\b", q) and result.get("classes"):
        rows = []
        for c in result["classes"][:10]:
            bases = f" extends {', '.join(c['bases'])}" if c.get("bases") else ""
            rows.append(f"- `{c['name']}`{bases} — {len(c['methods'])} methods")
        return "Detected classes:\n" + "\n".join(rows), label

    if re.search(r"\b(issue|problem|bug|warning|error|find|smell)\b", q):
        issues = result.get("issues", [])
        if not issues:
            return ("The rule-based checks found no issues in this source. "
                    "That means no *known-pattern* problems — not that the code "
                    "is bug-free.", label)
        rows = "\n".join(
            f"- [{i['severity']}] " + (f"line {i['line']}: " if i.get("line") else "")
            + i["message"]
            for i in issues[:15]
        )
        return f"Findings from static rules:\n{rows}", label

    # Bare function-name mention (no explicit keyword): answer about that function.
    words = set(re.findall(r"\b\w{3,}\b", q))
    bare = next(
        (f for f in result.get("functions", []) + [
            mm for c in result.get("classes", []) for mm in c["methods"]
        ] if f["name"].lower() in words),
        None,
    )
    if bare and not re.search(r"\b(complex|import|class|issue|problem|bug|"
                              r"warning|error|find|line|each|every)\b", q):
        args = ", ".join(a["name"] for a in bare.get("arguments", [])) or "no parameters"
        steps = "\n".join(f"- {st}" for st in (bare.get("walkthrough") or [])[:6])
        return (f"**`{bare['name']}({args})`** — lines {bare['line']}–{bare['end_line']}, "
                f"complexity {bare['complexity']}.\n\n{steps}", label)

    if re.search(r"\b(what|purpose|does|summary|explain|overview|simple)\b", q):
        import explainer
        return explainer.explain(result, source, "simple", language), label

    if re.search(r"\b(line|each|every|statement)\b", q):
        import explainer
        return explainer.explain(result, source, "line_by_line", language), label

    # Generic fallback: report what we know and suggest useful questions.
    s = result.get("summary", {})
    return (
        f"I can answer questions about this {s.get('language', language)} file "
        f"({s.get('total_lines', '?')} lines, {s.get('functions', 0)} functions, "
        f"{s.get('classes', 0)} classes). Try asking about a specific function "
        "name, complexity, imports, classes, or issues.\n\n"
        "Detected functions: "
        + (", ".join(f"`{f['name']}`" for f in result.get("functions", [])[:15]) or "none")
        + ".",
        label,
    )


def ask(question: str, result: dict[str, Any], source: str, language: str) -> tuple[str, str]:
    """Route to the LLM when configured, else the deterministic assistant."""
    if provider_configured():
        try:
            return ask_llm(question, result)
        except RuntimeError:
            pass  # fall through to deterministic answers
    return answer_deterministic(question, result, source, language)
