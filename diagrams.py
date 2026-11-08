"""SVG diagram generation from analysis results.

Diagrams are drawn only from detected structure: imports, classes, methods and
the candidate call graph. Every node corresponds to something the analyzer
actually found — nothing is decorative or invented. SVGs are returned as
strings so the UI can render them inline and offer them as downloads.
"""

from __future__ import annotations

import html
from typing import Any

import explainer

MAX_NODES = 40

_COLORS = {
    "module": "#57d9a3",
    "class": "#7aa2f7",
    "function": "#bb9af7",
    "import": "#565f89",
}


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _wrap(text: str, width: int = 34) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def _box(x: int, y: int, w: int, h: int, title: str, subtitle: str,
         color: str, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    lines = "".join(
        f'<text x="{x + 10}" y="{y + 18 + i * 13}" font-size="11" '
        f'font-family="Consolas, monospace" fill="#dfe6f0">{_esc(part)}</text>'
        for i, part in enumerate(_wrap(title, max(10, int((w - 20) / 6.5))))
    )
    sub = (
        f'<text x="{x + 10}" y="{y + h - 6}" font-size="9" '
        f'font-family="Consolas, monospace" fill="#8b98a9">{_esc(subtitle)}</text>'
        if subtitle else ""
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" '
        f'fill="#181c25" stroke="{color}" stroke-width="1.4"{dash}/>'
        f'{lines}{sub}'
    )


def _arrow(x1: int, y1: int, x2: int, y2: int, label: str = "") -> str:
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    label_svg = (
        f'<text x="{mx}" y="{my - 4}" font-size="9" text-anchor="middle" '
        f'font-family="Consolas, monospace" fill="#8b98a9">{_esc(label)}</text>'
        if label else ""
    )
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#3a4356" '
        f'stroke-width="1.2" marker-end="url(#arrow)"/>{label_svg}'
    )


def _svg_wrap(width: int, height: int, body: str, defs: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Inter, sans-serif">'
        f'<defs>{defs or _DEFAULT_DEFS}<rect id="bg" width="100%" height="100%" fill="#0f1117"/></defs>'
        f'<use href="#bg"/>{body}</svg>'
    )


_DEFAULT_DEFS = (
    '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
    'markerHeight="7" orient="auto-start-reverse">'
    '<path d="M0,0 L10,5 L0,10 z" fill="#3a4356"/></marker>'
)


def structure_svg(result: dict[str, Any]) -> str:
    """Module map: imports → classes → functions in a layered layout."""
    imports = result["imports"][:12]
    classes = result["classes"][:12]
    functions = result["functions"][:12]

    rows = []
    if imports:
        rows.append(("imports", imports))
    if classes:
        rows.append(("classes", classes))
    if functions:
        rows.append(("functions", functions))
    if not rows:
        return _svg_wrap(420, 120, _box(20, 30, 380, 60, "No structure detected",
                                       "nothing to draw", _COLORS["module"], True))

    col_w, row_h, pad = 210, 92, 26
    max_rows = max(len(items) for _, items in rows)
    width = 60 + len(rows) * (col_w + pad)
    height = 60 + max_rows * row_h
    body = []

    centers: dict[str, list[tuple[int, int]]] = {}
    for col, (kind, items) in enumerate(rows):
        x = 40 + col * (col_w + pad)
        color = _COLORS[kind.rstrip("s")] if kind.rstrip("s") in _COLORS else _COLORS["module"]
        pts = []
        for i, item in enumerate(items):
            y = 30 + i * row_h
            if kind == "imports":
                title, sub = item["module"], f"line {item['line']}"
            elif kind == "classes":
                title, sub = f"class {item['name']}", f"{len(item['methods'])} methods"
            else:
                title, sub = f"def {item['name']}", f"cx {item['complexity']}"
            body.append(_box(x, y, col_w, row_h - 18, title, sub, color))
            pts.append((x + col_w // 2, y + (row_h - 18) // 2))
        centers[kind] = pts

    # Draw arrows from imports column to the first class/function column.
    ordered_kinds = [k for k, _ in rows]
    for a, b in zip(ordered_kinds, ordered_kinds[1:]):
        for (x1, y1), (x2, y2) in zip(centers.get(a, [])[:6], centers.get(b, [])[:6]):
            body.append(_arrow(x1 + col_w // 2, y1, x2 - col_w // 2, y2))

    if len(result["imports"]) > 12 or len(result["functions"]) > 12 or len(result["classes"]) > 12:
        body.append(
            f'<text x="40" y="{height - 12}" font-size="10" fill="#8b98a9" '
            'font-family="Consolas, monospace">diagram truncated — see full report</text>'
        )
    return _svg_wrap(width, height, "".join(body))


def class_svg(result: dict[str, Any]) -> str:
    """Class boxes listing their methods, with inheritance arrows."""
    classes = result["classes"]
    if not classes:
        return _svg_wrap(420, 120, _box(20, 30, 380, 60, "No classes detected",
                                       "nothing to draw", _COLORS["class"], True))

    box_w, header_h, method_h, gap = 230, 34, 20, 40
    cols = 2 if len(classes) > 3 else 1
    rows_count = (len(classes) + cols - 1) // cols
    positions = {}
    body = []

    for i, cls in enumerate(classes[:16]):
        col, row = i % cols, i // cols
        x = 40 + col * (box_w + gap)
        y = 30 + row * (header_h + len(cls["methods"]) * method_h + 30)
        positions[cls["name"]] = (x, y)
        height = header_h + max(len(cls["methods"]), 1) * method_h + 14
        body.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{height}" rx="6" '
            f'fill="#181c25" stroke="{_COLORS["class"]}" stroke-width="1.5"/>'
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{header_h}" rx="6" '
            f'fill="#1c2438"/>'
            f'<text x="{x + 10}" y="{y + 22}" font-size="12" font-weight="600" '
            f'font-family="Consolas, monospace" fill="#dfe6f0">class {_esc(cls["name"])}</text>'
        )
        for j, method in enumerate(cls["methods"][:8]):
            body.append(
                f'<text x="{x + 14}" y="{y + header_h + 16 + j * method_h}" font-size="10" '
                f'font-family="Consolas, monospace" fill="#aeb9c8">'
                f'{_esc(method["name"])}() · L{method["line"]}</text>'
            )
        if len(cls["methods"]) > 8:
            body.append(
                f'<text x="{x + 14}" y="{y + header_h + 16 + 8 * method_h}" font-size="9" '
                f'fill="#8b98a9" font-family="Consolas, monospace">'
                f'… {len(cls["methods"]) - 8} more methods</text>'
            )

    # Inheritance arrows: base -> derived (only when both are known classes).
    known = {c["name"] for c in classes}
    for cls in classes:
        x1, y1 = positions[cls["name"]]
        for base in cls.get("bases", []):
            base_name = base.split(".")[-1].split("<")[0].strip()
            if base_name in known and base_name != cls["name"]:
                x2, y2 = positions[base_name]
                body.append(_arrow(x2 + box_w // 2, y2 + header_h,
                                   x1 + box_w // 2, y1 - 4, "extends"))

    width = 80 + cols * (box_w + gap)
    height = 70 + rows_count * (header_h + 8 * method_h + 40)
    return _svg_wrap(width, height, "".join(body))


def call_svg(result: dict[str, Any], source: str, language: str) -> str:
    """Candidate call graph between detected functions/methods."""
    calls = explainer._call_names(source, language)
    if not calls:
        return _svg_wrap(420, 120, _box(20, 30, 380, 60,
                                       "No internal calls detected",
                                       "functions may still call builtins or imports",
                                       _COLORS["function"], True))

    names = sorted(calls)[:MAX_NODES]
    index = {name: i for i, name in enumerate(names)}
    cols = 4
    box_w, box_h, gap_x, gap_y = 170, 44, 30, 34
    rows = (len(names) + cols - 1) // cols
    width = 60 + cols * (box_w + gap_x)
    height = 60 + rows * (box_h + gap_y)
    body = []
    pos = {}
    for name, i in index.items():
        col, row = i % cols, i // cols
        x = 40 + col * (box_w + gap_x)
        y = 30 + row * (box_h + gap_y)
        pos[name] = (x + box_w // 2, y + box_h // 2)
        body.append(_box(x, y, box_w, box_h, name, "function", _COLORS["function"]))
    for caller, targets in calls.items():
        if caller not in pos:
            continue
        x1, y1 = pos[caller]
        for target in sorted(targets):
            if target in pos:
                x2, y2 = pos[target]
                body.append(_arrow(x1, y1 + box_h // 2 + 4, x2, y2 - box_h // 2 - 4))
    return _svg_wrap(width, height, "".join(body))


def to_downloadable(svg: str, name: str = "diagram.svg") -> tuple[bytes, str]:
    return svg.encode("utf-8"), name
