"""Explanation-mode renderers.

Every mode produces Markdown built *only* from the analysis result — counts,
names, line numbers, walkthrough steps and findings that the analyzer actually
found. Nothing is invented: when the data does not support a claim, the mode
says so instead of guessing.
"""

from __future__ import annotations

import re
from typing import Any

import polyglot
from utils import complexity_label, plural, short_signature

MODES = {
    "simple": "Simple Explanation",
    "detailed": "Detailed Explanation",
    "line_by_line": "Line-by-Line Explanation",
    "functions": "Function-by-Function Explanation",
    "classes": "Class-by-Class Explanation",
    "algorithm": "Algorithm & Complexity Explanation",
    "execution_flow": "Execution Flow Explanation",
    "advanced": "Advanced Technical Explanation",
}

MAX_LINE_BY_LINE = 400


def _cx_word(value: int) -> str:
    return complexity_label(value)


def _call_names(source: str, language: str) -> dict[str, set[str]]:
    """Best-effort map of definition name -> names it calls.

    Python uses the AST; other languages fall back to identifier matching
    inside each function's body. Only identifiers that match a known
    definition are kept, so the result is a *candidate* call graph.
    """
    import ast

    calls: dict[str, set[str]] = {}
    if language == "python":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return calls
        known = {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                invoked = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        name = func.id if isinstance(func, ast.Name) else (
                            func.attr if isinstance(func, ast.Attribute) else None
                        )
                        if name in known:
                            invoked.add(name)
                calls[node.name] = invoked
        return calls

    # Heuristic fallback: regex identifiers inside each function body.
    result = polyglot.analyze_generic(source, language) if language in polyglot.SPECS else None
    if not result:
        return calls
    known = {f["name"] for f in result["functions"]} | {
        m["name"] for c in result["classes"] for m in c["methods"]
    }
    lines = source.splitlines()
    for func in result["functions"] + [
        m for c in result["classes"] for m in c["methods"]
    ]:
        body = "\n".join(lines[func["line"]:func["end_line"]])
        invoked = {
            name for name in re.findall(r"\b([A-Za-z_]\w*)\s*\(", body)
            if name in known and name != func["name"]
        }
        calls[func["name"]] = invoked
    return calls


def _intro(result: dict[str, Any], source: str, language: str) -> list[str]:
    s = result["summary"]
    out = [
        f"**Language:** {s.get('language', language)} "
        f"({s.get('analysis_kind', 'unknown')} analysis)",
        f"**Size:** {plural(s['total_lines'], 'line')} "
        f"({s['non_blank_lines']} non-blank, {s['comment_lines']} comment lines)",
        f"**Structure:** {plural(s['functions'], 'function')}, "
        f"{plural(s['classes'], 'class')}, {plural(s['methods'], 'method')}, "
        f"{plural(s['imports'], 'import')}",
    ]
    cov = result.get("coverage")
    if cov and cov.get("percent") is not None and cov["percent"] < 100:
        out.append(
            f"**Coverage:** {cov['percent']}% of lines were inside analysed segments "
            f"({cov['segments_analyzed']}/{cov['segments_total']} segments)."
        )
    return out


def explain(result: dict[str, Any], source: str, mode: str, language: str) -> str:
    renderer = {
        "simple": _simple,
        "detailed": _detailed,
        "line_by_line": _line_by_line,
        "functions": _functions,
        "classes": _classes,
        "algorithm": _algorithm,
        "execution_flow": _execution_flow,
        "advanced": _advanced,
    }.get(mode, _detailed)
    return renderer(result, source, language)


def _simple(result: dict[str, Any], source: str, language: str) -> str:
    s = result["summary"]
    parts = ["## Simple Explanation", ""]
    parts += _intro(result, source, language)
    parts.append("")
    parts.append(
        f"This {s.get('language', language)} file contains "
        f"{plural(s['functions'], 'function')} and {plural(s['classes'], 'class')}. "
        "In plain terms:"
    )
    parts.append("")
    if result["functions"]:
        names = ", ".join(f"`{f['name']}`" for f in result["functions"][:8])
        more = f" and {len(result['functions']) - 8} more" if len(result["functions"]) > 8 else ""
        parts.append(f"- The functions are {names}{more}.")
    if result["classes"]:
        names = ", ".join(f"`{c['name']}`" for c in result["classes"][:8])
        parts.append(f"- The classes are {names}.")
    if result["imports"]:
        mods = ", ".join(dict.fromkeys(i["module"] for i in result["imports"][:8]))
        parts.append(f"- It relies on these modules: {mods}.")
    if result["issues"]:
        parts.append(
            f"- The static checks flagged {plural(sum(result['issue_counts'].values()), 'finding')}; "
            "see the Findings section of the full report for details."
        )
    else:
        parts.append("- The rule-based checks found nothing to flag.")
    parts.append("")
    parts.append(
        "*This is a structural explanation generated from static analysis. "
        "It describes what the code contains, not what the author intended.*"
    )
    return "\n".join(parts)


def _detailed(result: dict[str, Any], source: str, language: str) -> str:
    s = result["summary"]
    parts = ["## Detailed Explanation", ""]
    parts += _intro(result, source, language)
    parts.append("")
    parts.append("### Program structure")
    parts.append("```")
    parts.append(result["structure"])
    parts.append("```")
    parts.append("")

    if result["imports"]:
        parts.append("### Imports and dependencies")
        for item in result["imports"][:20]:
            tag = "stdlib" if item.get("stdlib") else "third-party/unknown"
            used = "used" if item.get("used", True) else "possibly unused"
            parts.append(f"- `{item['module']}` (line {item['line']}) — {tag}, {used}")
        if len(result["imports"]) > 20:
            parts.append(f"- … {len(result['imports']) - 20} more imports")
        parts.append("")

    if result["functions"]:
        parts.append("### Functions")
        for f in result["functions"]:
            args = short_signature(f["arguments"]) or "no arguments"
            parts.append(
                f"- **`{f['name']}({args})`** — lines {f['line']}–{f['end_line']}, "
                f"complexity {f['complexity']} ({_cx_word(f['complexity'])}). "
                + (" ".join(f["walkthrough"][:2]) if f["walkthrough"] else "")
            )
        parts.append("")

    if result["classes"]:
        parts.append("### Classes")
        for c in result["classes"]:
            bases = f" extends {', '.join(c['bases'])}" if c.get("bases") else ""
            parts.append(
                f"- **`{c['name']}`**{bases} — lines {c['line']}–{c['end_line']}, "
                f"{plural(len(c['methods']), 'method')}"
            )
            for m in c["methods"][:6]:
                parts.append(f"  - `{m['name']}()` at line {m['line']}, complexity {m['complexity']}")
        parts.append("")

    if result["issues"]:
        parts.append("### Findings")
        counts = result["issue_counts"]
        parts.append(
            f"{counts['error']} error(s), {counts['warning']} warning(s), "
            f"{counts['info']} info note(s)."
        )
        for issue in result["issues"][:15]:
            where = f"line {issue['line']}: " if issue.get("line") else ""
            parts.append(f"- [{issue['severity']}] {where}{issue['message']}")
        parts.append("")
    return "\n".join(parts)


def _line_by_line(result: dict[str, Any], source: str, language: str) -> str:
    lines = source.splitlines()
    parts = ["## Line-by-Line Explanation", ""]
    parts += _intro(result, source, language)
    parts.append("")
    if len(lines) > MAX_LINE_BY_LINE:
        parts.append(
            f"*This file has {len(lines)} lines; the first {MAX_LINE_BY_LINE} are "
            "annotated below and the rest are covered by the structural analysis.*"
        )
        parts.append("")
        lines = lines[:MAX_LINE_BY_LINE]

    defs = _definition_lines(result)
    for number, text in enumerate(lines, start=1):
        stripped = text.strip()
        kind = _classify_line(stripped, language, number, defs)
        if not stripped:
            continue
        parts.append(f"**{number}.** `{stripped[:90]}` — {kind}")
    parts.append("")
    parts.append(
        "*Line annotations are structural (what kind of statement each line is), "
        "not a semantic translation of intent.*"
    )
    return "\n".join(parts)


def _definition_lines(result: dict[str, Any]) -> dict[int, str]:
    defs: dict[int, str] = {}
    for f in result["functions"]:
        defs[f["line"]] = f"definition of function `{f['name']}`"
    for c in result["classes"]:
        defs[c["line"]] = f"definition of class `{c['name']}`"
        for m in c["methods"]:
            defs[m["line"]] = f"method `{m['name']}` inside class `{c['name']}`"
    for i in result["imports"]:
        defs.setdefault(i["line"], f"import of `{i['module']}`")
    return defs


def _classify_line(stripped: str, language: str, number: int,
                   defs: dict[int, str]) -> str:
    if number in defs:
        return defs[number] + "."
    low = stripped.lower()
    if language == "python":
        if stripped.startswith("#"):
            return "comment."
        if stripped.startswith(('"""', "'''", '"', "'")) and len(stripped) > 3:
            return "string literal or docstring."
        if re.match(r"^(def|class|async def)\b", low):
            return "definition."
        if re.match(r"^(import|from)\b", low):
            return "import statement."
        if re.match(r"^(if|elif|else|for|while|try|except|finally|with|match|case)\b:?", low):
            return "control-flow statement."
        if re.match(r"^(return|yield|raise|assert|pass|break|continue|del|global|nonlocal)\b", low):
            return "jump/exit statement."
        if "=" in stripped and not stripped.startswith(("==", "!=")):
            return "assignment or expression."
        return "expression statement."
    if stripped.startswith(("//", "#", "--", "/*", "*", "<!--")):
        return "comment."
    if re.match(r"^(import|from|#include|use|require|package|using|namespace)\b", low):
        return "import/include statement."
    if re.match(r"^(public|private|protected|static|final|abstract|class|interface|"
                r"struct|enum|trait|impl|fn|func|function|def|fun|sub|type|module)\b", low):
        return "declaration."
    if re.match(r"^(if|elif|else|for|foreach|while|switch|case|default|catch|try|"
                r"finally|do|loop|when|guard|unless)\b", low):
        return "control-flow statement."
    if re.match(r"^(return|break|continue|throw|raise|yield|goto|exit)\b", low):
        return "exit/exception statement."
    if re.search(r"[{};]\s*$", stripped):
        return "block or terminated statement."
    return "statement."


def _functions(result: dict[str, Any], source: str, language: str) -> str:
    calls = _call_names(source, language)
    parts = ["## Function-by-Function Explanation", ""]
    parts += _intro(result, source, language)
    parts.append("")
    all_funcs = list(result["functions"])
    for c in result["classes"]:
        for m in c["methods"]:
            item = dict(m)
            item["_class"] = c["name"]
            all_funcs.append(item)
    if not all_funcs:
        parts.append("No functions or methods were detected in this source.")
        return "\n".join(parts)

    for f in all_funcs:
        where = f" (method of `{f['_class']}`)" if "_class" in f else ""
        args = short_signature(f["arguments"]) or "no parameters"
        parts.append(f"### `{f['name']}({args})`{where}")
        parts.append(f"- **Location:** lines {f['line']}–{f['end_line']}")
        parts.append(f"- **Complexity:** {f['complexity']} ({_cx_word(f['complexity'])})")
        invoked = calls.get(f["name"], set())
        if invoked:
            parts.append(f"- **Calls:** {', '.join(sorted(invoked))}")
        callers = sorted(n for n, targets in calls.items() if f["name"] in targets)
        if callers:
            parts.append(f"- **Called by:** {', '.join(callers)}")
        if f.get("returns"):
            parts.append(f"- **Returns:** {', '.join(f'`{r}`' for r in f['returns'][:5])}")
        if f.get("raises"):
            parts.append(f"- **Raises/throws:** {', '.join(f'`{r}`' for r in f['raises'][:5])}")
        if f.get("defaults"):
            parts.append(f"- **Mutable default arguments:** {', '.join(f['defaults'])}")
        parts.append("- **Logic:**")
        for step in (f.get("walkthrough") or ["No structural steps were extracted."])[:8]:
            parts.append(f"  - {step}")
        parts.append("")
    return "\n".join(parts)


def _classes(result: dict[str, Any], source: str, language: str) -> str:
    parts = ["## Class-by-Class Explanation", ""]
    parts += _intro(result, source, language)
    parts.append("")
    if not result["classes"]:
        parts.append("No classes were detected in this source.")
        return "\n".join(parts)
    for c in result["classes"]:
        bases = f" — inherits from {', '.join(c['bases'])}" if c.get("bases") else ""
        parts.append(f"### `{c['name']}`{bases}")
        parts.append(f"- **Location:** lines {c['line']}–{c['end_line']}")
        parts.append(f"- **Methods:** {plural(len(c['methods']), 'method')}")
        for m in c["methods"]:
            parts.append(
                f"  - `{m['name']}()` at line {m['line']} — complexity {m['complexity']}"
            )
        parts.append("")
    return "\n".join(parts)


def _algorithm(result: dict[str, Any], source: str, language: str) -> str:
    s = result["summary"]
    parts = ["## Algorithm & Complexity Explanation", ""]
    parts += _intro(result, source, language)
    parts.append("")
    counts = polyglot.count_branches_text(source)
    parts.append("### Control-flow profile (keyword counts over the whole file)")
    parts.append(
        f"- Conditionals: ~{counts['if']} · Loops: ~{counts['loops']} · "
        f"Try/catch blocks: ~{counts['try']} · Returns: ~{counts['returns']} · "
        f"Switch/match arms: ~{counts['switch']}"
    )
    parts.append("")
    parts.append("### Per-function complexity")
    funcs = result["functions"] + [
        m for c in result["classes"] for m in c["methods"]
    ]
    if funcs:
        hottest = max(funcs, key=lambda f: f["complexity"])
        for f in sorted(funcs, key=lambda x: -x["complexity"])[:10]:
            parts.append(
                f"- `{f['name']}` — complexity {f['complexity']} ({_cx_word(f['complexity'])})"
            )
        parts.append("")
        parts.append(
            f"The most branch-heavy definition is `{hottest['name']}` "
            f"(complexity {hottest['complexity']}). Higher structural complexity "
            "means more independent paths through the code; it correlates with "
            "harder testing and maintenance but is *not* a measured runtime cost."
        )
    else:
        parts.append("No functions were detected, so no per-function complexity exists.")
    parts.append("")
    parts.append(
        "### Complexity estimate\n\n"
        "Time/space complexity cannot be proven from syntax alone. The structural "
        "signals above (nested loops multiply work; comprehensions and large "
        "data structures hint at extra memory) are the honest basis for an "
        "estimate. For a rigorous bound, identify the dominant loop nest and "
        "the data structure sizes — or run a profiler."
    )
    return "\n".join(parts)


def _execution_flow(result: dict[str, Any], source: str, language: str) -> str:
    s = result["summary"]
    lines = source.splitlines()
    parts = ["## Execution Flow Explanation", ""]
    parts += _intro(result, source, language)
    parts.append("")
    parts.append("### How this file runs, top to bottom")
    parts.append("1. **Imports load first.** Every `import`/`#include`/`use` statement "
                 "executes before any other statement in the file.")
    parts.append(f"2. **Definitions are registered.** The {plural(s['functions'], 'function')} "
                 f"and {plural(s['classes'], 'class')} are defined but not yet executed "
                 "(their bodies run only when called).")
    top_level = [
        (i + 1, l.strip()) for i, l in enumerate(lines)
        if l.strip() and not l.strip().startswith(("#", "//", "import", "from",
                                                 "#include", "use ", "package ",
                                                 "def ", "class ", "function ", "func ",
                                                 "fn ", "fun "))
    ]
    if top_level:
        parts.append("3. **Top-level statements execute in order:**")
        for number, text in top_level[:15]:
            parts.append(f"   - line {number}: `{text[:80]}`")
        if len(top_level) > 15:
            parts.append(f"   - … {len(top_level) - 15} more top-level statements")
    else:
        parts.append("3. **No top-level executable statements** were detected outside "
                     "definitions and imports — the file only defines things.")
    if language == "python":
        parts.append(
            "4. **Entry point:** unless an `if __name__ == \"__main__\"` guard exists, "
            "top-level statements run on import; with the guard they run only when "
            "the file is executed directly."
        )
    else:
        parts.append("4. **Entry point:** execution starts at the language's designated "
                     "entry point (`main`, the script body, etc.); definitions above "
                     "only register names.")
    parts.append("")
    return "\n".join(parts)


def _advanced(result: dict[str, Any], source: str, language: str) -> str:
    s = result["summary"]
    parts = ["## Advanced Technical Explanation", ""]
    parts += _intro(result, source, language)
    parts.append("")
    parts.append("### Architecture signals")
    third_party = [i for i in result["imports"] if not i.get("stdlib", True)]
    stdlib = [i for i in result["imports"] if i.get("stdlib")]
    parts.append(
        f"- {plural(len(stdlib), 'standard-library')} import and "
        f"{plural(len(third_party), 'external')} import detected"
        + (f" ({', '.join(i['module'] for i in third_party[:8])})." if third_party else ".")
    )
    deepest = max(
        (f["complexity"] for f in result["functions"]
         + [m for c in result["classes"] for m in c["methods"]] or [0]),
        default=0,
    )
    parts.append(f"- Peak structural complexity: {deepest} "
                 f"({_cx_word(deepest) if deepest else 'none detected'})")
    parts.append("")
    parts.append("### Design observations (static, conservative)")
    if result["classes"]:
        with_bases = [c for c in result["classes"] if c.get("bases")]
        parts.append(
            f"- {plural(len(result['classes']), 'class')} detected"
            + (f"; {len(with_bases)} use inheritance ({', '.join(c['name'] for c in with_bases[:6])})."
               if with_bases else "; none declare explicit bases.")
        )
    else:
        parts.append("- No classes detected; the code is procedural/functional in shape.")
    unused = [i for i in result["imports"] if not i.get("used", True)]
    if unused:
        parts.append(f"- {plural(len(unused), 'import')} appear unused: "
                     + ", ".join(f"`{i['module']}`" for i in unused[:8]) + ".")
    hot = [i for i in result["issues"] if i["severity"] == "warning"]
    if hot:
        parts.append(f"- {plural(len(hot), 'warning')} from the rule checks — review the "
                     "Findings table; they are confirmed static findings, not guesses.")
    parts.append("")
    parts.append(
        "### What static analysis cannot tell you\n\n"
        "True design patterns, runtime performance, type safety across modules and "
        "business-logic correctness require either execution or an LLM pass over the "
        "source. This report deliberately stops at what the syntax proves."
    )
    return "\n".join(parts)


def explain_all(result: dict[str, Any], source: str, language: str) -> str:
    """Concatenate every mode — used for the downloadable full explanation."""
    chunks = []
    for mode in MODES:
        chunks.append(explain(result, source, mode, language))
        chunks.append("\n\n---\n\n")
    return "".join(chunks)
