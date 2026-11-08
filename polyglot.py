"""Heuristic multi-language analysis for the Code Explanation Assistant.

Python source is analysed with :mod:`analyzer` (real AST parsing). For every
other supported language we fall back to pattern-based heuristics: declaration
regexes, brace matching for block extents, and keyword counting for complexity.

This is honest static analysis — it finds *where* functions, classes and
imports live and how complex they look structurally. It does not parse the
grammar of each language, so findings are labelled ``heuristic`` in the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

LONG_LINE = 100
MAX_ISSUES = 40


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    comment: tuple[str, ...]
    imports: re.Pattern[str]
    types: re.Pattern[str]          # class / interface / struct / enum / trait
    functions: re.Pattern[str]
    calls_extra: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

_COMMON_IMPORTS = re.compile(
    r"^\s*(?:import|from|#include|use|require|require_once|package|using|"
    r"open\s+|source\s+|load\s+|import\s+static)\b[^\n]*",
    re.IGNORECASE,
)

SPECS: dict[str, LanguageSpec] = {
    "java": LanguageSpec(
        "Java", (".java",), ("//", "/*", "*"),
        re.compile(r"^\s*(?:import\s+[\w.]+|package\s+[\w.]+)\s*;"),
        re.compile(r"^\s*(?:public|private|protected|abstract|final|static|\s)*"
                   r"(?:class|interface|enum|record|@interface)\s+\w+"),
        re.compile(r"^\s*(?:public|private|protected|static|final|abstract|synchronized|"
                   r"native|default|\s)*[\w<>\[\],.?]+\s+\w+\s*\([^;]*\)\s*(?:throws[\w,\s]+)?\{"),
    ),
    "javascript": LanguageSpec(
        "JavaScript", (".js", ".jsx", ".mjs", ".cjs"), ("//", "/*", "*"),
        re.compile(r"^\s*(?:import\b[^;\n]*|const\s+\w+\s*=\s*require\(|export\b)"),
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+\w+"),
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s*\w+\s*\(|"
                   r"^\s*(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|"
                   r"^\s*\w+\s*\([^)]*\)\s*\{"),
    ),
    "typescript": LanguageSpec(
        "TypeScript", (".ts", ".tsx"), ("//", "/*", "*"),
        re.compile(r"^\s*(?:import\b[^;\n]*|export\b[^;\n]*|require\()"),
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+\w+|"
                   r"^\s*(?:export\s+)?interface\s+\w+"),
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s*\w+\s*[(<]|"
                   r"^\s*(?:public|private|protected|static|readonly|\s)*[\w<>\[\],.?]+\s+"
                   r"\w+\s*\([^)]*\)\s*[:{]|"
                   r"^\s*(?:const|let)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"),
    ),
    "c": LanguageSpec(
        "C", (".c", ".h"), ("//", "/*", "*"),
        re.compile(r"^\s*#\s*include\b[^\n]*"),
        re.compile(r"^\s*(?:typedef\s+)?(?:struct|enum|union)\s+\w+"),
        re.compile(r"^[\w\s\*]+\w+\s*\([^;]*\)\s*\{"),
    ),
    "cpp": LanguageSpec(
        "C++", (".cpp", ".cc", ".cxx", ".hpp", ".hh"), ("//", "/*", "*"),
        re.compile(r"^\s*#\s*include\b[^\n]*|^\s*using\s+namespace\b[^\n]*"),
        re.compile(r"^\s*class\s+\w+|^\s*struct\s+\w+|^\s*template\s*<[^>]*>"),
        re.compile(r"^\s*(?:template\s*<[^>]*>\s*)?[\w:<>,\s\*&]+\s+\w+::?\w*\s*\([^;]*\)\s*(?:const)?\s*\{|"
                   r"^\s*[\w:<>,\s\*&]+\s+\w+\s*\([^;]*\)\s*\{"),
    ),
    "csharp": LanguageSpec(
        "C#", (".cs",), ("//", "/*", "*"),
        re.compile(r"^\s*(?:using\s+[\w.]+;|namespace\s+[\w.]+)"),
        re.compile(r"^\s*(?:public|private|protected|internal|abstract|sealed|static|partial|\s)*"
                   r"(?:class|interface|struct|enum|record)\s+\w+"),
        re.compile(r"^\s*(?:public|private|protected|internal|static|virtual|override|abstract|"
                   r"async|sealed|new|partial|\s)*[\w<>\[\],.?]+\s+\w+\s*\([^)]*\)\s*"),
    ),
    "php": LanguageSpec(
        "PHP", (".php", ".phtml"), ("//", "#", "/*", "*"),
        re.compile(r"^\s*(?:require|include|use|namespace)\b[^\n;]*;?"),
        re.compile(r"^\s*(?:abstract\s+|final\s+)?class\s+\w+|^\s*interface\s+\w+|^\s*trait\s+\w+"),
        re.compile(r"^\s*(?:public|private|protected|static|abstract|final|\s)*function\s*&?\w+\s*\(|"
                   r"^\s*(?:public|private|protected|static|\s)*\w+\s*\([^)]*\)\s*[:{]"),
    ),
    "go": LanguageSpec(
        "Go", (".go",), ("//", "/*", "*"),
        re.compile(r"^\s*(?:import\s*\(|import\s+\"|package\s+\w+)"),
        re.compile(r"^\s*type\s+\w+\s+(?:struct|interface)\b"),
        re.compile(r"^func\s+(?:\([^)]*\)\s*)?\w+\s*\([^)]*\)\s*(?:[\w\.\[\]\*]+\s*)?\{"),
    ),
    "rust": LanguageSpec(
        "Rust", (".rs",), ("//", "/*", "*"),
        re.compile(r"^\s*(?:use\s+[\w:]+|mod\s+\w+|extern\s+crate\s+\w+)"),
        re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait|impl|mod)\s+\w+"),
        re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+\w+\s*[<(]"),
    ),
    "kotlin": LanguageSpec(
        "Kotlin", (".kt", ".kts"), ("//", "/*", "*"),
        re.compile(r"^\s*(?:import\s+[\w.]+|package\s+[\w.]+)"),
        re.compile(r"^\s*(?:data\s+|sealed\s+|enum\s+|annotation\s+|open\s+|abstract\s+|"
                   r"inner\s+)*class\s+\w+|^\s*interface\s+\w+|^\s*object\s+\w+"),
        re.compile(r"^\s*(?:override\s+|suspend\s+|inline\s+|private\s+|public\s+|"
                   r"protected\s+|internal\s+|tailrec\s+|operator\s+|infix\s+)*fun\s+\w+"),
    ),
    "swift": LanguageSpec(
        "Swift", (".swift",), ("//", "/*", "*"),
        re.compile(r"^\s*(?:import\s+[\w.]+)"),
        re.compile(r"^\s*(?:public|private|internal|fileprivate|open|final|indirect|\s)*"
                   r"(?:class|struct|enum|protocol|extension)\s+\w+"),
        re.compile(r"^\s*(?:public|private|internal|fileprivate|open|final|static|class|mutating|"
                   r"override|convenience|required|lazy|weak|unowned|\s)*func\s+\w+"),
    ),
    "ruby": LanguageSpec(
        "Ruby", (".rb", ".rake"), ("#",),
        re.compile(r"^\s*(?:require|require_relative|load|include|extend)\b[^\n]*"),
        re.compile(r"^\s*class\s+\w+|^\s*module\s+\w+"),
        re.compile(r"^\s*def\s+\w+[!?=]?"),
    ),
    "bash": LanguageSpec(
        "Bash", (".sh", ".bash", ".zsh"), ("#",),
        re.compile(r"^\s*(?:source\s+\S+|\.\s+\S+|export\s+\w+=)"),
        re.compile(r""),  # never matches
        re.compile(r"^\s*(?:function\s+)?\w+\s*\(\s*\)\s*\{|^\s*function\s+\w+"),
    ),
    "sql": LanguageSpec(
        "SQL", (".sql",), ("--", "/*", "*"),
        re.compile(r""),  # never matches
        re.compile(r""),
        re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:PROCEDURE|FUNCTION|TRIGGER|VIEW)\s+\w+",
                   re.IGNORECASE),
    ),
    "html": LanguageSpec(
        "HTML", (".html", ".htm"), ("<!--",),
        re.compile(r""),
        re.compile(r""),
        re.compile(r""),
    ),
    "css": LanguageSpec(
        "CSS", (".css",), ("/*", "*"),
        re.compile(r"^\s*@import\b[^;]*;"),
        re.compile(r""),
        re.compile(r""),
    ),
}

EXTENSION_MAP = {
    ext: lang for lang, spec in SPECS.items() for ext in spec.extensions
}


def language_for_filename(filename: str) -> str | None:
    lowered = filename.lower()
    for ext, lang in EXTENSION_MAP.items():
        if lowered.endswith(ext):
            return lang
    return None


def language_name(lang: str) -> str:
    return SPECS[lang].name if lang in SPECS else lang.capitalize()


def list_languages() -> list[str]:
    return sorted(SPECS)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_SHEBANG = re.compile(r"^#!.*\b(?:python|node|ruby|perl|php|bash|sh|zsh)\b")

_CONTENT_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(?:import\s+[\w.]+|from\s+[\w.]+\s+import)\s*$", re.M), "python"),
    (re.compile(r"^\s*(?:public\s+)?(?:static\s+)?(?:void|int|String|boolean|double|float|"
                r"long|char)\s+main\s*\(", re.M), "java"),
    (re.compile(r"^\s*public\s+class\s+\w+", re.M), "java"),
    (re.compile(r"^\s*(?:const|let|var)\s+\w+\s*=\s*require\(", re.M), "javascript"),
    (re.compile(r"^\s*(?:import|export)\s+.*\bfrom\b", re.M), "typescript"),
    (re.compile(r"^\s*package\s+main\s*$", re.M), "go"),
    (re.compile(r"^\s*fn\s+main\s*\(", re.M), "rust"),
    (re.compile(r"^\s*#include\s*<", re.M), "c"),
    (re.compile(r"^\s*using\s+namespace\s+std", re.M), "cpp"),
    (re.compile(r"^\s*using\s+System", re.M), "csharp"),
    (re.compile(r"<\?php", re.M), "php"),
    (re.compile(r"^\s*CREATE\s+(?:TABLE|DATABASE|PROCEDURE)\b", re.I | re.M), "sql"),
    (re.compile(r"^\s*def\s+\w+\s*\(", re.M), "python"),
    (re.compile(r"^\s*func\s+\w+\s*\(", re.M), "go"),
    (re.compile(r"^\s*(?:SELECT|INSERT|UPDATE|DELETE)\b", re.I | re.M), "sql"),
]


def detect_language(source: str, filename: str = "") -> str:
    """Best-effort language detection: filename extension first, then content."""
    by_name = language_for_filename(filename)
    if by_name:
        return by_name
    if filename:
        lower = filename.lower()
        if lower.endswith((".txt", ".code", ".snippet")) or "." not in lower.rsplit("/", 1)[-1]:
            pass  # fall through to content detection

    first = source.lstrip()[:200]
    shebang = _SHEBANG.search(first)
    if shebang:
        token = shebang.group(0).lower()
        if "python" in token:
            return "python"
        if "node" in token:
            return "javascript"
        if "ruby" in token:
            return "ruby"
        if "php" in token:
            return "php"
        return "bash"

    scores: dict[str, int] = {}
    for pattern, lang in _CONTENT_HINTS:
        hits = len(pattern.findall(source[:50_000]))
        if hits:
            scores[lang] = scores.get(lang, 0) + hits
    if scores:
        return max(scores.items(), key=lambda item: item[1])[0]
    return "text"


# ---------------------------------------------------------------------------
# Brace matching and block extraction
# ---------------------------------------------------------------------------

def _line_comment_prefixes(spec: LanguageSpec) -> tuple[str, ...]:
    return spec.comment


def _strip_comments(lines: list[str], spec: LanguageSpec) -> list[str]:
    """Blank out comment text so braces/keywords inside comments don't count."""
    out = []
    in_block = False
    block_start, block_end = "/*", "*/"
    for line in lines:
        stripped = line.strip()
        if spec.name in {"HTML"} and "<!--" in stripped:
            line = re.sub(r"<!--.*?-->", "", line)
            out.append(line)
            continue
        if "/*" in line or in_block:
            # crude block-comment removal
            res = ""
            i = 0
            while i < len(line):
                if in_block:
                    end = line.find("*/", i)
                    if end == -1:
                        i = len(line)
                    else:
                        in_block = False
                        i = end + 2
                elif line.startswith("/*", i):
                    in_block = True
                    i += 2
                else:
                    res += line[i]
                    i += 1
            line = res
        for prefix in _line_comment_prefixes(spec):
            if prefix in ("//", "#", "--"):
                idx = line.find(prefix)
                if idx != -1:
                    line = line[:idx] + " " * (len(line) - idx)
                    break
        out.append(line)
    return out


def _brace_end(lines: list[str], start: int) -> int:
    """Return the 1-based end line of the brace block opened at ``start``."""
    depth = 0
    opened = False
    for index in range(start - 1, len(lines)):
        for char in lines[index]:
            if char == "{":
                depth += 1
                opened = True
            elif char == "}":
                depth -= 1
                if opened and depth <= 0:
                    return index + 1
        if not opened and ";" in lines[index]:
            return index + 1
    return len(lines)


def _indent_end(lines: list[str], start: int) -> int:
    """For indentation languages (Python is handled by AST; used as fallback)."""
    base = len(lines[start - 1]) - len(lines[start - 1].lstrip())
    for index in range(start, len(lines)):
        line = lines[index]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base:
            return index
    return len(lines)


def _paren_args(text: str) -> list[str]:
    """Crude argument-name extraction from a parameter list."""
    inner = text.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    if not inner.strip() or inner.strip() in {"void", "..."}:
        return []
    parts = []
    depth = 0
    current = ""
    for char in inner:
        if char in "(<[":
            depth += 1
        elif char in ")>]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current)

    names = []
    for part in parts:
        cleaned = re.sub(r"[\[\]<>:?&*].*$", "", part).strip()
        cleaned = re.sub(r"^(?:final|const|var|out|ref|in|public|private|protected|"
                         r"static|readonly|mut|val|var)\s+", "", cleaned)
        tokens = cleaned.split()
        if not tokens:
            continue
        name = tokens[-1].lstrip("*&")
        if name and re.match(r"^[\w$]+$", name):
            names.append(name)
    return names


def _extract_signature(line: str) -> tuple[str, str]:
    """Pull (name, args_text) out of a declaration line, best effort."""
    # Arrow functions / assignments: `const add = (a, b) => ...`
    arrow = re.search(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(", line)
    if arrow:
        name = arrow.group(1)
        open_paren = line.find("(", arrow.end() - 1)
        depth = 0
        for index in range(open_paren, len(line)):
            if line[index] == "(":
                depth += 1
            elif line[index] == ")":
                depth -= 1
                if depth == 0:
                    return name, line[open_paren + 1:index]
        return name, ""
    # `function name(...)`, `def name(...)`, `fn name(...)`, `name(...) {`
    match = re.search(r"(?:function|def|fn|func|fun|sub)\s+(\w+)\s*\(|(\w+)\s*\(", line)
    if not match:
        return "", ""
    name = match.group(1) or match.group(2)
    open_paren = line.find("(", match.start())
    depth = 0
    for index in range(open_paren, len(line)):
        if line[index] == "(":
            depth += 1
        elif line[index] == ")":
            depth -= 1
            if depth == 0:
                return name, line[open_paren + 1:index]
    return name, line[open_paren + 1:]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_generic(source: str, language: str, filename: str = "<pasted code>") -> dict[str, Any]:
    """Heuristic analysis for non-Python languages. Same schema as analyzer.analyze."""
    if not source or not source.strip():
        raise SyntaxError("There is no code to analyse.")

    spec = SPECS.get(language)
    if spec is None:
        return _analyze_text(source, filename)

    lines = source.splitlines()
    clean = _strip_comments(lines, spec)

    imports: list[dict[str, Any]] = []
    types: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []

    type_stack: list[dict[str, Any]] = []

    for index, (raw, line) in enumerate(zip(lines, clean), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        if spec.imports.pattern and spec.imports.match(stripped):
            module = stripped.split()[1] if len(stripped.split()) > 1 else stripped
            module = re.sub(r"[;\"']", "", module)
            imports.append({
                "module": module or stripped,
                "bound": module or stripped,
                "alias": None,
                "line": index,
                "stdlib": False,
                "used": True,  # heuristic: cannot prove unused
            })
            continue

        if spec.types.pattern and spec.types.match(stripped):
            name_match = re.search(r"\b(?:class|interface|struct|enum|trait|union|"
                                   r"record|protocol|object|module|type|extension)\s+(\w+)",
                                   stripped)
            name = name_match.group(1) if name_match else f"block_{index}"
            bases = re.findall(r"(?::\s*|extends\s+|implements\s+|<\s*)([\w.,\s]+)", stripped)
            base_names = []
            for group in bases:
                base_names += [b.strip() for b in re.split(r"[,\s]+", group) if b.strip() and b.strip() != "{"]
            brace_line = stripped if "{" in stripped else next(
                (lines[j] for j in range(index, min(index + 3, len(lines)))
                 if "{" in lines[j]), "" )
            item = {
                "name": name,
                "kind": "type",
                "line": index,
                "end_line": _brace_end(clean, index) if "{" in brace_line else index,
                "bases": base_names[:5],
                "docstring": False,
                "methods": [],
            }
            while type_stack and type_stack[-1]["end_line"] < index:
                type_stack.pop()
            if type_stack:
                # Nested type: record it inside the enclosing type's methods list
                # is wrong — keep it as a top-level type but note nesting via name.
                pass
            types.append(item)
            type_stack.append(item)
            continue

        if spec.functions.pattern and spec.functions.match(stripped):
            name, args_text = _extract_signature(stripped)
            if not name or name in {"if", "for", "while", "switch", "catch", "return",
                                    "function", "fn", "def", "fun", "func", "sub"}:
                continue
            func = _build_function(name, args_text, index, clean, raw)
            if type_stack and type_stack[-1]["end_line"] >= index:
                type_stack[-1]["methods"].append(func)
            else:
                functions.append(func)
            continue

    # For languages where type blocks were found, keep methods out of top-level
    # function list where appropriate (they stay reachable via classes).
    top_functions = [f for f in functions
                     if not any(f in t["methods"] for t in types)]

    summary = {
        "filename": filename,
        "language": spec.name,
        "analysis_kind": "heuristic",
        "total_lines": len(lines),
        "non_blank_lines": sum(1 for line in lines if line.strip()),
        "comment_lines": sum(1 for line in lines if line.strip().startswith(spec.comment[0])),
        "functions": len(top_functions),
        "classes": len(types),
        "methods": sum(len(t["methods"]) for t in types),
        "imports": len(imports),
        "has_docstring": False,
        "max_complexity": max(
            [f["complexity"] for f in top_functions]
            + [m["complexity"] for t in types for m in t["methods"]]
            + [0]
        ),
    }

    issues = _generic_issues(lines, clean, top_functions, types, imports, spec)

    return {
        "summary": summary,
        "imports": imports,
        "functions": top_functions,
        "classes": types,
        "issues": issues,
        "issue_counts": {
            severity: sum(1 for i in issues if i["severity"] == severity)
            for severity in ("error", "warning", "info")
        },
        "structure": _generic_structure(summary, imports, types, top_functions),
    }


def _build_function(name: str, args_text: str, line: int, clean: list[str],
                    raw: str) -> dict[str, Any]:
    args = _paren_args("(" + args_text + ")") if args_text else []
    end = _brace_end(clean, line) if "{" in raw else _indent_end(clean, line)
    body_lines = clean[line:end]
    body_text = "\n".join(body_lines)

    complexity = 1 + len(re.findall(
        r"\b(?:if|elif|else if|for|foreach|while|until|switch|case|catch|except|"
        r"when|guard|elsif)\b", body_text))
    complexity += len(re.findall(r"&&|\|\||\band\b|\bor\b", body_text)) // 2

    returns = re.findall(r"\breturn\s+([^;\n}]+)", body_text)[:8]
    raises = []
    throws = re.search(r"\bthrows\s+([\w.,\s]+)", raw)
    if throws:
        raises = [t.strip() for t in throws.group(1).split(",")]
    for exc in re.findall(r"\bthrow\s+([\w.]+)", body_text):
        raises.append(exc)

    walkthrough = [
        f"Declared at line {line} with {len(args)} parameter(s): "
        f"{', '.join(args) if args else 'none'}."
    ]
    if "{" in raw:
        walkthrough.append(f"Body spans lines {line + 1}–{end} (brace-delimited block).")
    if returns:
        walkthrough.append(f"Returns values including `{returns[0].strip()}`.")
    if raises:
        walkthrough.append(f"Can raise/throw: {', '.join(sorted(set(raises)))}.")
    walkthrough.append(
        f"Structural complexity ≈ {complexity} (keyword-based heuristic, not a grammar parse)."
    )

    return {
        "name": name,
        "line": line,
        "end_line": end,
        "arguments": [{"name": a, "default": None} for a in args],
        "defaults": [],
        "docstring": False,
        "statements": sum(1 for l in body_lines if l.strip() and not l.strip().startswith("}")),
        "complexity": max(complexity, 1),
        "returns": [r.strip() for r in returns],
        "raises": sorted(set(raises)),
        "walkthrough": walkthrough,
        "is_async": "async" in raw.split("(")[0],
        "decorators": [],
        "signature": raw.strip()[:200],
    }


def _generic_issues(lines: list[str], clean: list[str], functions: list[dict[str, Any]],
                    types: list[dict[str, Any]], imports: list[dict[str, Any]],
                    spec: LanguageSpec) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    def add(severity: str, code: str, message: str, line: int | None = None) -> None:
        issues.append({"severity": severity, "code": code, "message": message, "line": line})

    for number, line in enumerate(lines, start=1):
        if len(line) > LONG_LINE:
            add("info", "long-line",
                f"Line {number} is {len(line)} characters; the limit here is {LONG_LINE}.", number)

    all_funcs = list(functions) + [m for t in types for m in t["methods"]]
    for func in all_funcs:
        if func["complexity"] > 10:
            add("warning", "complex-function",
                f"`{func['name']}` has structural complexity ≈{func['complexity']}; "
                "consider splitting it.", func["line"])
        if func["statements"] > 80:
            add("warning", "long-function",
                f"`{func['name']}` spans ~{func['statements']} statements; consider splitting it.",
                func["line"])

    for number, line in enumerate(clean, start=1):
        if re.search(r"\b(TODO|FIXME|XXX|HACK)\b", line):
            add("info", "todo-marker", "A TODO/FIXME marker is present; the intent is unfinished.",
                number)

    if len(issues) > MAX_ISSUES:
        issues = issues[:MAX_ISSUES]
        add("info", "truncated", f"Only the first {MAX_ISSUES} findings are shown.")

    order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: (order.get(i["severity"], 3), i["line"] or 0))
    return issues


def _generic_structure(summary: dict[str, Any], imports: list[dict[str, Any]],
                       types: list[dict[str, Any]], functions: list[dict[str, Any]]) -> str:
    lines = [f"{summary['filename']} — {summary['total_lines']} lines "
             f"({summary['analysis_kind']} analysis)"]
    blocks = []
    for item in imports[:30]:
        blocks.append(f"import {item['module']}")
    if len(imports) > 30:
        blocks.append(f"… {len(imports) - 30} more imports")
    for item in types:
        blocks.append(f"{item['kind']} {item['name']}  ({len(item['methods'])} method(s))")
        for method in item["methods"]:
            blocks.append(f"    {method['name']}()  ~cx {method['complexity']}")
    for item in functions:
        blocks.append(f"fn {item['name']}()  ~cx {item['complexity']}")
    for index, block in enumerate(blocks):
        last = index == len(blocks) - 1
        lines.append(("└─ " if last else "├─ ") + block)
    if not blocks:
        lines.append("└─ no imports, types or functions detected")
    return "\n".join(lines)


def _analyze_text(source: str, filename: str) -> dict[str, Any]:
    """Fallback for plain text / unknown languages: line stats only."""
    lines = source.splitlines()
    summary = {
        "filename": filename,
        "language": "Plain text",
        "analysis_kind": "basic",
        "total_lines": len(lines),
        "non_blank_lines": sum(1 for line in lines if line.strip()),
        "comment_lines": 0,
        "functions": 0, "classes": 0, "methods": 0, "imports": 0,
        "has_docstring": False, "max_complexity": 0,
    }
    return {
        "summary": summary,
        "imports": [], "functions": [], "classes": [],
        "issues": [],
        "issue_counts": {"error": 0, "warning": 0, "info": 0},
        "structure": f"{filename} — {len(lines)} lines of plain text; "
                     "no code structure was detected.",
    }


def analyze_any(source: str, language: str, filename: str = "<pasted code>") -> dict[str, Any]:
    """Dispatch: Python goes through the AST analyzer; everything else is heuristic."""
    if language == "python":
        import analyzer
        result = analyzer.analyze(source, filename)
        result["summary"]["language"] = "Python"
        result["summary"]["analysis_kind"] = "ast"
        return result
    return analyze_generic(source, language, filename)


def count_branches_text(text: str) -> dict[str, int]:
    """Keyword counts used for algorithm/flow explanations on heuristic languages."""
    return {
        "if": len(re.findall(r"\b(?:if|elif)\b", text)),
        "loops": len(re.findall(r"\b(?:for|while|foreach|until)\b", text)),
        "try": len(re.findall(r"\b(?:try|catch|except|finally)\b", text)),
        "returns": len(re.findall(r"\breturn\b", text)),
        "switch": len(re.findall(r"\b(?:switch|case|match|when)\b", text)),
    }
