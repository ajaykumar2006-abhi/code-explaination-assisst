"""Static analysis behind the Code Explanation Assistant.

Everything here reads the abstract syntax tree produced by :mod:`ast`. There is no
model and no training step, so the output is deterministic: the same source always
produces the same description. The explanations are structural, not semantic — see
the README's limitations section.
"""

from __future__ import annotations

import ast
import sys
from typing import Any

LONG_LINE = 100
LONG_FUNCTION_STATEMENTS = 50
MAX_ISSUES = 40
MUTABLE_DEFAULT_TEXT = {"[]", "{}", "set()", "list()", "dict()"}

BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.IfExp,
)
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def analyze(source: str, filename: str = "<pasted code>") -> dict[str, Any]:
    """Return a description of ``source`` or raise ``SyntaxError`` if it does not parse."""
    if not source or not source.strip():
        raise SyntaxError("There is no code to analyse.")

    tree = ast.parse(source, filename=filename)

    used_names = _used_names(tree)
    imports = _collect_imports(tree, used_names)
    functions = [
        _describe_function(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = [_describe_class(node) for node in tree.body if isinstance(node, ast.ClassDef)]

    lines = source.splitlines()
    summary = {
        "filename": filename,
        "total_lines": len(lines),
        "non_blank_lines": sum(1 for line in lines if line.strip()),
        "comment_lines": sum(1 for line in lines if line.strip().startswith("#")),
        "functions": len(functions),
        "classes": len(classes),
        "methods": sum(len(item["methods"]) for item in classes),
        "imports": len(imports),
        "has_docstring": bool(ast.get_docstring(tree)),
        "max_complexity": max(
            [item["complexity"] for item in functions]
            + [method["complexity"] for item in classes for method in item["methods"]]
            + [0]
        ),
    }

    issues = _collect_issues(tree, lines, functions, classes, imports)

    return {
        "summary": summary,
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "issues": issues,
        "issue_counts": {
            severity: sum(1 for issue in issues if issue["severity"] == severity)
            for severity in ("error", "warning", "info")
        },
        "structure": _structure(summary, imports, classes, functions),
    }


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # unparse can fail on hand-built nodes
        return "<expression>"


def _used_names(tree: ast.AST) -> set[str]:
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            root: ast.AST = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used.add(root.id)
    return used


def _collect_imports(tree: ast.AST, used: set[str]) -> list[dict[str, Any]]:
    imports: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                root = alias.name.split(".")[0]
                imports.append(
                    {
                        "module": alias.name,
                        "bound": bound,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "stdlib": root in sys.stdlib_module_names,
                        "used": bound in used,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            for alias in node.names:
                if alias.name == "*":
                    bound = "*"
                else:
                    bound = alias.asname or alias.name
                imports.append(
                    {
                        "module": module or ".",
                        "bound": bound,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "stdlib": bool(root) and root in sys.stdlib_module_names,
                        "used": alias.name == "*" or bound in used or module == "__future__",
                    }
                )
    return imports


def _describe_arguments(node: ast.AST) -> list[dict[str, Any]]:
    args = node.args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    names = [argument.arg for argument in positional]
    defaults = [_unparse(default) for default in args.defaults]
    offset = len(names) - len(defaults)

    described: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        described.append(
            {"name": name, "default": defaults[index - offset] if index >= offset else None}
        )
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        described.append(
            {"name": argument.arg, "default": _unparse(default) if default is not None else None}
        )
    if args.vararg:
        described.append({"name": f"*{args.vararg.arg}", "default": None})
    if args.kwarg:
        described.append({"name": f"**{args.kwarg.arg}", "default": None})
    return described


def _mutable_defaults(node: ast.AST) -> list[str]:
    args = node.args  # type: ignore[attr-defined]
    candidates = [*args.defaults, *(item for item in args.kw_defaults if item is not None)]
    found = []
    for default in candidates:
        if isinstance(default, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
            found.append(_unparse(default))
        elif _unparse(default) in MUTABLE_DEFAULT_TEXT:
            found.append(_unparse(default))
    return found


def _complexity(node: ast.AST) -> int:
    """McCabe-style count: one path plus every branch, loop, handler and boolean split."""
    score = 1
    for child in ast.walk(node):
        if isinstance(child, BRANCH_NODES):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += max(len(child.values) - 1, 0)
        elif isinstance(child, COMPREHENSIONS):
            score += len(child.generators)
    match_case = getattr(ast, "match_case", None)
    if match_case is not None:
        score += sum(1 for child in ast.walk(node) if isinstance(child, match_case))
    return score


def _statements(node: ast.AST) -> int:
    return sum(1 for child in ast.walk(node) if isinstance(child, ast.stmt))


def _returns_and_raises(node: ast.AST) -> tuple[list[str], list[str]]:
    returns: list[str] = []
    raises: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Return):
            returns.append(_unparse(child.value) if child.value is not None else "None")
        elif isinstance(child, ast.Raise):
            raises.append(_unparse(child.exc) if child.exc is not None else "the current exception")
    return returns, raises


def _walkthrough(node: ast.AST) -> list[str]:
    steps: list[str] = []
    arguments = _describe_arguments(node)
    if not arguments:
        steps.append("Takes no arguments.")
    else:
        rendered = ", ".join(
            f"{item['name']} (default {item['default']})" if item["default"] else item["name"]
            for item in arguments
        )
        steps.append(f"Takes {len(arguments)} argument(s): {rendered}.")

    if ast.get_docstring(node):
        steps.append("Documents itself with a docstring.")

    constructs = sorted(
        (
            child
            for child in ast.walk(node)
            if isinstance(
                child,
                (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Return, ast.Raise),
            )
        ),
        key=lambda child: (child.lineno, child.col_offset),
    )

    for child in constructs[:12]:
        line = child.lineno
        if isinstance(child, ast.If):
            steps.append(f"Line {line}: branches on `{_unparse(child.test)}`.")
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            steps.append(f"Line {line}: loops over `{_unparse(child.iter)}` into `{_unparse(child.target)}`.")
        elif isinstance(child, ast.While):
            steps.append(f"Line {line}: repeats while `{_unparse(child.test)}`.")
        elif isinstance(child, ast.Try):
            steps.append(f"Line {line}: guards work with try/except and optionally finally.")
        elif isinstance(child, (ast.With, ast.AsyncWith)):
            managers = ", ".join(_unparse(item.context_expr) for item in child.items)
            steps.append(f"Line {line}: opens a context manager ({managers}).")
        elif isinstance(child, ast.Return):
            if child.value is None:
                steps.append(f"Line {line}: returns early without a value.")
            else:
                steps.append(f"Line {line}: returns `{_unparse(child.value)}`.")
        elif isinstance(child, ast.Raise):
            steps.append(f"Line {line}: raises `{_unparse(child.exc) if child.exc else 'the current exception'}`.")

    if len(constructs) > 12:
        steps.append(f"…and {len(constructs) - 12} more return/branch statements.")

    if not any(isinstance(child, ast.Return) for child in ast.walk(node)):
        steps.append("Falls off the end; it returns nothing explicitly.")

    return steps


def _describe_function(node: ast.AST) -> dict[str, Any]:
    returns, raises = _returns_and_raises(node)
    return {
        "name": node.name,
        "line": node.lineno,
        "end_line": getattr(node, "end_lineno", node.lineno),
        "arguments": _describe_arguments(node),
        "defaults": _mutable_defaults(node),
        "docstring": bool(ast.get_docstring(node)),
        "statements": _statements(node),
        "complexity": _complexity(node),
        "returns": returns,
        "raises": raises,
        "walkthrough": _walkthrough(node),
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "decorators": [_unparse(item) for item in node.decorator_list],
    }


def _describe_class(node: ast.ClassDef) -> dict[str, Any]:
    methods = [
        _describe_function(child)
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return {
        "name": node.name,
        "line": node.lineno,
        "end_line": getattr(node, "end_lineno", node.lineno),
        "bases": [_unparse(base) for base in node.bases],
        "docstring": bool(ast.get_docstring(node)),
        "methods": methods,
    }


def _main_guard(tree: ast.Module) -> tuple[int, int] | None:
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in _unparse(node.test):
            return node.lineno, getattr(node, "end_lineno", node.lineno)
    return None


def _collect_issues(
    tree: ast.Module,
    lines: list[str],
    functions: list[dict[str, Any]],
    classes: list[dict[str, Any]],
    imports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    def add(severity: str, code: str, message: str, line: int | None = None) -> None:
        issues.append({"severity": severity, "code": code, "message": message, "line": line})

    if not ast.get_docstring(tree):
        add("info", "module-docstring", "The module has no docstring, so its purpose is not stated.", 1)

    all_functions = list(functions) + [method for item in classes for method in item["methods"]]
    for function in all_functions:
        if not function["docstring"]:
            add("info", "missing-docstring", f"`{function['name']}` has no docstring.", function["line"])
        if function["statements"] > LONG_FUNCTION_STATEMENTS:
            add(
                "warning",
                "long-function",
                f"`{function['name']}` contains {function['statements']} statements; consider splitting it.",
                function["line"],
            )
        for default in function["defaults"]:
            add(
                "warning",
                "mutable-default",
                f"`{function['name']}` uses `{default}` as a default argument, which is shared across calls.",
                function["line"],
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                add("warning", "bare-except", "Bare `except:` catches everything, including KeyboardInterrupt.", node.lineno)
            elif _unparse(node.type) in {"Exception", "BaseException"} and all(
                isinstance(statement, ast.Pass) for statement in node.body
            ):
                add(
                    "warning",
                    "swallowed-exception",
                    f"`except {_unparse(node.type)}: pass` hides the failure; at least log it.",
                    node.lineno,
                )

    guard = _main_guard(tree)
    prints_seen = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "print":
            continue
        inside_guard = guard is not None and guard[0] <= node.lineno <= guard[1]
        if not inside_guard and prints_seen < 3:
            prints_seen += 1
            add(
                "info",
                "print-outside-main",
                "`print()` writes to stdout outside an `if __name__ == \"__main__\"` guard.",
                node.lineno,
            )

    for item in imports:
        if not item["used"]:
            add("warning", "unused-import", f"`{item['module']}` is imported but never used.", item["line"])

    for number, line in enumerate(lines, start=1):
        if len(line) > LONG_LINE:
            add("info", "long-line", f"Line {number} is {len(line)} characters; the limit here is {LONG_LINE}.", number)

    if len(issues) > MAX_ISSUES:
        issues = issues[:MAX_ISSUES]
        add("info", "truncated", f"Only the first {MAX_ISSUES} findings are shown.")

    order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda item: (order.get(item["severity"], 3), item["line"] or 0))
    return issues


def _structure(
    summary: dict[str, Any],
    imports: list[dict[str, Any]],
    classes: list[dict[str, Any]],
    functions: list[dict[str, Any]],
) -> str:
    lines = [f"{summary['filename']} — {summary['total_lines']} lines"]

    blocks = []
    for item in imports:
        alias = f" as {item['alias']}" if item["alias"] else ""
        suffix = "" if item["used"] else "   (unused)"
        blocks.append(f"import {item['module']}{alias}{suffix}")
    for item in classes:
        blocks.append(f"class {item['name']}  ({len(item['methods'])} method(s))")
        for method in item["methods"]:
            blocks.append(f"    def {method['name']}(…)  complexity {method['complexity']}")
    for item in functions:
        blocks.append(f"def {item['name']}(…)  complexity {item['complexity']}")

    for index, block in enumerate(blocks):
        last = index == len(blocks) - 1
        lines.append(("└─ " if last else "├─ ") + block)

    if not blocks:
        lines.append("└─ no imports, functions or classes")

    return "\n".join(lines)
