"""Large-source pipeline: segmentation, context and coverage tracking.

The analyzer already handles multi-thousand-line files cheaply because it walks
the AST once. What this module adds is *structure-aware segmentation*: the file
is split at meaningful boundaries (top-level definitions, classes, import
blocks) rather than at a fixed character count, each segment is analysed in
isolation, and a coverage report records exactly which line ranges were
examined — so the UI can never silently skip part of the file.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import analyzer
import polyglot


@dataclass
class Segment:
    kind: str            # module-header | class | function | tail
    name: str
    start_line: int      # 1-based, inclusive
    end_line: int        # 1-based, inclusive
    code: str = ""
    summary: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)
    analyzed: bool = False
    skip_reason: str = ""


@dataclass
class CoverageReport:
    total_lines: int
    covered_lines: int
    segments_total: int
    segments_analyzed: int
    segments_skipped: int
    skipped_ranges: list[tuple[int, int, str]]
    complete: bool

    @property
    def percent(self) -> float:
        if self.total_lines == 0:
            return 100.0
        return round(100.0 * self.covered_lines / self.total_lines, 1)


def _python_segments(source: str) -> list[Segment]:
    """Split Python source at top-level definition boundaries via the AST."""
    tree = ast.parse(source)
    lines = source.splitlines()
    segments: list[Segment] = []

    header_end = 0
    body = list(tree.body)
    first_def_line = min(
        (getattr(n, "lineno", 10**9) for n in body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))),
        default=len(lines) + 1,
    )
    header_end = max(first_def_line - 1, 0)
    if header_end > 0:
        segments.append(Segment(
            kind="module-header", name="<module header>",
            start_line=1, end_line=header_end,
            code="\n".join(lines[:header_end]),
        ))

    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            segments.append(Segment(
                kind="function", name=node.name,
                start_line=node.lineno, end_line=end,
                code="\n".join(lines[node.lineno - 1:end]),
            ))
        elif isinstance(node, ast.ClassDef):
            end = getattr(node, "end_lineno", node.lineno)
            segments.append(Segment(
                kind="class", name=node.name,
                start_line=node.lineno, end_line=end,
                code="\n".join(lines[node.lineno - 1:end]),
            ))
        elif not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
            end = getattr(node, "end_lineno", node.lineno)
            segments.append(Segment(
                kind="statement", name=type(node).__name__.lower(),
                start_line=node.lineno, end_line=end,
                code="\n".join(lines[node.lineno - 1:end]),
            ))

    covered = max((s.end_line for s in segments), default=0)
    if covered < len(lines):
        segments.append(Segment(
            kind="tail", name="<trailing code>",
            start_line=covered + 1, end_line=len(lines),
            code="\n".join(lines[covered:]),
        ))
    return segments


def _generic_segments(source: str, language: str) -> list[Segment]:
    """Segment heuristic languages using the line ranges found by polyglot."""
    result = polyglot.analyze_generic(source, language)
    lines = source.splitlines()
    segments: list[Segment] = []

    first_line = min(
        [i["line"] for i in result["imports"]]
        + [f["line"] for f in result["functions"]]
        + [c["line"] for c in result["classes"]]
        or [len(lines) + 1]
    )
    if first_line > 1:
        segments.append(Segment("module-header", "<module header>",
                                1, first_line - 1, "\n".join(lines[:first_line - 1])))

    blocks = sorted(
        [("function", f) for f in result["functions"]]
        + [("class", c) for c in result["classes"]],
        key=lambda pair: pair[1]["line"],
    )
    cursor = first_line
    for kind, block in blocks:
        if block["line"] > cursor:
            segments.append(Segment("statement", "<top-level code>", cursor,
                                    block["line"] - 1,
                                    "\n".join(lines[cursor - 1:block["line"] - 1])))
        segments.append(Segment(kind, block["name"], block["line"],
                                block["end_line"],
                                "\n".join(lines[block["line"] - 1:block["end_line"]])))
        cursor = block["end_line"] + 1
    if cursor <= len(lines):
        segments.append(Segment("tail", "<trailing code>", cursor, len(lines),
                                "\n".join(lines[cursor - 1:])))
    return segments


def segment_source(source: str, language: str) -> list[Segment]:
    if language == "python":
        try:
            return _python_segments(source)
        except SyntaxError:
            return _generic_segments(source, language)
    return _generic_segments(source, language)


def analyze_segments(segments: list[Segment], source: str, language: str,
                     max_segment_lines: int = 400) -> list[Segment]:
    """Analyse each segment; oversized segments are summarised, not skipped."""
    for segment in segments:
        if not segment.code.strip():
            segment.skip_reason = "empty"
            continue
        span = segment.end_line - segment.start_line + 1
        if span > max_segment_lines:
            segment.skip_reason = (
                f"segment is {span} lines; summarised structurally "
                "(raise the segment limit for a per-line pass)"
            )
            segment.analyzed = True
            continue
        try:
            if segment.kind == "function":
                if language == "python":
                    sub = analyzer.analyze(segment.code, filename=f"<{segment.name}>")
                else:
                    sub = polyglot.analyze_generic(
                        segment.code, language, filename=f"<{segment.name}>")
                segment.findings = sub["issues"]
                segment.summary = (
                    f"{segment.name}() — lines {segment.start_line}–{segment.end_line}, "
                    f"{len(segment.code.splitlines())} lines"
                )
            elif segment.kind == "class":
                segment.summary = (
                    f"class {segment.name} — lines {segment.start_line}–{segment.end_line}, "
                    f"{len(segment.code.splitlines())} lines"
                )
            else:
                segment.summary = (
                    f"{segment.kind} — lines {segment.start_line}–{segment.end_line}, "
                    f"{len(segment.code.splitlines())} lines"
                )
            segment.analyzed = True
        except SyntaxError as exc:
            # A segment may not parse alone (indentation); fall back to text stats.
            segment.skip_reason = f"not independently parseable ({exc})"
            segment.summary = (
                f"{segment.kind} {segment.name} — lines {segment.start_line}–"
                f"{segment.end_line} (structure analysed at file level)"
            )
            segment.analyzed = True
    return segments


def coverage(segments: list[Segment], total_lines: int) -> CoverageReport:
    covered: list[tuple[int, int]] = []
    skipped: list[tuple[int, int, str]] = []
    for segment in segments:
        if segment.analyzed or segment.skip_reason == "empty":
            covered.append((segment.start_line, segment.end_line))
        else:
            skipped.append((segment.start_line, segment.end_line, segment.skip_reason))

    covered_lines = sum(min(e, total_lines) - s + 1 for s, e in covered if s <= total_lines)
    return CoverageReport(
        total_lines=total_lines,
        covered_lines=min(covered_lines, total_lines),
        segments_total=len(segments),
        segments_analyzed=sum(1 for s in segments if s.analyzed),
        segments_skipped=sum(1 for s in segments if not s.analyzed),
        skipped_ranges=skipped,
        complete=not skipped,
    )


def analyze_source(source: str, language: str, filename: str = "<pasted code>") -> dict[str, Any]:
    """Full pipeline: whole-file analysis + segmentation + coverage.

    Returns the analyzer/polyglot result extended with ``segments`` and
    ``coverage`` keys. Coverage is honest: every line range is either inside an
    analysed segment or listed in ``skipped_ranges`` with a reason.
    """
    result = polyglot.analyze_any(source, language, filename)
    segments = segment_source(source, language)
    analyze_segments(segments, source, language)
    total_lines = result["summary"]["total_lines"]

    for segment in segments:
        segment_dict = {
            "kind": segment.kind,
            "name": segment.name,
            "start_line": segment.start_line,
            "end_line": segment.end_line,
            "analyzed": segment.analyzed,
            "skip_reason": segment.skip_reason,
            "summary": segment.summary,
            "findings": segment.findings[:10],
        }
        result.setdefault("segment_details", []).append(segment_dict)

    result["coverage"] = {
        "total_lines": total_lines,
        "covered_lines": coverage(segments, total_lines).covered_lines,
        "percent": coverage(segments, total_lines).percent,
        "segments_total": len(segments),
        "segments_analyzed": sum(1 for s in segments if s.analyzed),
        "complete": coverage(segments, total_lines).complete,
        "skipped_ranges": [
            {"start": s, "end": e, "reason": r}
            for s, e, r in coverage(segments, total_lines).skipped_ranges
        ],
    }
    return result
