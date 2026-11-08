"""Validation and display helpers for the Code Explanation Assistant."""

from __future__ import annotations

MAX_SOURCE_CHARS = 200_000

SEVERITY_BADGE = {"error": "🔴 error", "warning": "🟠 warning", "info": "🔵 info"}


def validate_source(source: str, max_chars: int = MAX_SOURCE_CHARS) -> str | None:
    """Return an error message, or None when the snippet is worth parsing."""
    if source is None or not source.strip():
        return "Paste some code first."
    if len(source) > max_chars:
        return f"That is {len(source):,} characters; the limit is {max_chars:,}."
    return None


def code_line_count(source: str) -> int:
    return len(source.splitlines()) if source else 0


def complexity_label(value: int) -> str:
    if value <= 4:
        return "simple"
    if value <= 10:
        return "moderate"
    return "complex"


def severity_badge(severity: str) -> str:
    return SEVERITY_BADGE.get(severity, severity)


def short_signature(arguments: list[dict]) -> str:
    parts = []
    for item in arguments:
        parts.append(f"{item['name']}={item['default']}" if item.get("default") else item["name"])
    return ", ".join(parts)


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    word = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count} {word}"
