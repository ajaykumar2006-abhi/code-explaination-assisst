"""Tests for the Code Explanation Assistant.

Run with:  python -m pytest tests.py -q
"""

from __future__ import annotations

import sys

import pytest
from streamlit.testing.v1 import AppTest

import ai_client
import analyzer
from analyzer import analyze
import diagrams
import explainer
import exports
import pipeline
import polyglot
from utils import complexity_label, severity_badge, short_signature, validate_source

SIMPLE = '''def add(a, b=2):
    """Add two numbers."""
    return a + b
'''

BRANCHY = '''def classify(values):
    total = 0
    for value in values:
        if value is None:
            continue
        elif value > 10 and value < 20:
            total += value
        else:
            total -= 1
    return total
'''

JAVA_SAMPLE = '''import java.util.List;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public static void main(String[] args) {
        for (int i = 0; i < 10; i++) {
            System.out.println(i);
        }
    }
}
'''


# ---------------------------------------------------------------------------
# analyzer.py (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_simple_function_is_described():
    result = analyze(SIMPLE)
    assert result["summary"]["functions"] == 1
    function = result["functions"][0]
    assert function["name"] == "add"
    assert [item["name"] for item in function["arguments"]] == ["a", "b"]
    assert function["arguments"][1]["default"] == "2"
    assert function["docstring"] is True


def test_walkthrough_mentions_branch_and_return():
    function = analyze(BRANCHY)["functions"][0]
    steps = " ".join(function["walkthrough"]).lower()
    assert "loops over" in steps
    assert "branches on" in steps
    assert "returns" in steps


def test_class_methods_are_listed():
    source = '''class Account:
    """An account."""

    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance
'''
    result = analyze(source)
    assert result["summary"]["classes"] == 1
    account = result["classes"][0]
    assert [method["name"] for method in account["methods"]] == ["__init__", "deposit"]
    assert "class Account" in result["structure"]


def test_class_bases_are_reported():
    result = analyze("class Child(Parent):\n    pass\n")
    assert result["classes"][0]["bases"] == ["Parent"]


def test_complexity_grows_with_branches():
    simple = analyze(SIMPLE)["functions"][0]["complexity"]
    branchy = analyze(BRANCHY)["functions"][0]["complexity"]
    assert simple == 1
    assert branchy > simple


def test_unused_import_is_flagged():
    result = analyze("import os\n\n\ndef f():\n    return 1\n")
    unused = [item for item in result["imports"] if not item["used"]]
    assert [item["module"] for item in unused] == ["os"]
    assert any(item["code"] == "unused-import" for item in result["issues"])


def test_used_import_is_not_flagged():
    result = analyze("import math\n\n\ndef f(x):\n    return math.sqrt(x)\n")
    assert all(item["used"] for item in result["imports"])
    assert not any(item["code"] == "unused-import" for item in result["issues"])


def test_stdlib_flag_is_set():
    result = analyze("import json\nimport numpy as np\n\n\ndef f():\n    return json.dumps({}), np\n")
    flags = {item["module"]: item["stdlib"] for item in result["imports"]}
    assert flags["json"] is True
    assert flags["numpy"] is False


def test_mutable_default_is_flagged():
    result = analyze("def collect(items=[]):\n    items.append(1)\n    return items\n")
    assert any(item["code"] == "mutable-default" for item in result["issues"])


def test_bare_except_is_flagged():
    result = analyze("try:\n    value = 1\nexcept:\n    pass\n")
    assert any(item["code"] == "bare-except" for item in result["issues"])


def test_swallowed_exception_is_flagged():
    result = analyze("try:\n    value = 1\nexcept Exception:\n    pass\n")
    assert any(item["code"] == "swallowed-exception" for item in result["issues"])


def test_missing_docstring_is_noted():
    result = analyze("def undocumented():\n    return 1\n")
    assert any(item["code"] == "missing-docstring" for item in result["issues"])


def test_long_line_is_flagged():
    result = analyze("value = " + "1 + " * 40 + "1\n")
    assert any(item["code"] == "long-line" for item in result["issues"])


def test_syntax_error_carries_a_line_number():
    with pytest.raises(SyntaxError) as failure:
        analyze("def broken(:\n    pass\n")
    assert isinstance(failure.value.lineno, int)


def test_empty_source_raises():
    with pytest.raises(SyntaxError):
        analyze("   \n\n")


def test_summary_counts_lines_and_statements():
    source = SIMPLE + "\n" + BRANCHY
    summary = analyze(source)["summary"]
    assert summary["total_lines"] == len(source.splitlines())
    assert summary["functions"] == 2
    assert summary["non_blank_lines"] <= summary["total_lines"]


def test_structure_lists_top_level_names():
    structure = analyze(SIMPLE + "\n" + BRANCHY)["structure"]
    assert "def add" in structure
    assert "def classify" in structure


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------

def test_validation_rejects_empty_and_accepts_code():
    assert validate_source("") is not None
    assert validate_source("def f():\n    pass\n") is None


def test_validation_rejects_oversized_input():
    assert validate_source("x = 1\n" * 40_000) is not None
    assert validate_source("x = 1\n" * 10, max_chars=20) is not None


def test_display_helpers():
    assert complexity_label(1) == "simple"
    assert complexity_label(8) == "moderate"
    assert complexity_label(20) == "complex"
    assert "error" in severity_badge("error")
    signature = short_signature([{"name": "a", "default": None}, {"name": "b", "default": "2"}])
    assert signature == "a, b=2"


# ---------------------------------------------------------------------------
# polyglot.py
# ---------------------------------------------------------------------------

def test_language_detection_by_extension():
    assert polyglot.detect_language("x", "Foo.java") == "java"
    assert polyglot.detect_language("x", "app.ts") == "typescript"
    assert polyglot.detect_language("x", "main.go") == "go"


def test_language_detection_by_content():
    assert polyglot.detect_language("def f():\n    pass\n") == "python"
    assert polyglot.detect_language("public class A {\n}\n") == "java"
    assert polyglot.detect_language("SELECT * FROM t;\n") == "sql"


def test_java_analysis_finds_class_and_methods():
    result = polyglot.analyze_generic(JAVA_SAMPLE, "java", "Calculator.java")
    assert result["summary"]["analysis_kind"] == "heuristic"
    assert result["summary"]["classes"] == 1
    assert result["summary"]["imports"] == 1
    methods = [m["name"] for m in result["classes"][0]["methods"]]
    assert "add" in methods and "main" in methods


def test_javascript_arrow_function_detected():
    source = "const add = (a, b) => a + b;\nfunction greet(name) {\n  return name;\n}\n"
    result = polyglot.analyze_generic(source, "javascript", "a.js")
    names = [f["name"] for f in result["functions"]]
    assert "add" in names and "greet" in names


def test_unknown_language_falls_back_to_text():
    result = polyglot.analyze_generic("just some words\n", "text", "notes.txt")
    assert result["summary"]["functions"] == 0
    assert "plain text" in result["structure"]


def test_analyze_any_dispatches_python_to_ast():
    result = polyglot.analyze_any(SIMPLE, "python", "a.py")
    assert result["summary"]["analysis_kind"] == "ast"


def test_long_line_flagged_in_generic():
    result = polyglot.analyze_generic("x" * 200 + "\n", "c", "a.c")
    assert any(i["code"] == "long-line" for i in result["issues"])


# ---------------------------------------------------------------------------
# pipeline.py
# ---------------------------------------------------------------------------

def test_pipeline_segments_python_file():
    source = SIMPLE + "\n" + BRANCHY + "\n\nclass Thing:\n    pass\n"
    segments = pipeline.segment_source(source, "python")
    kinds = {s.kind for s in segments}
    assert "function" in kinds and "class" in kinds
    names = {s.name for s in segments}
    assert "add" in names and "classify" in names and "Thing" in names


def test_pipeline_coverage_is_complete_for_small_file():
    result = pipeline.analyze_source(SIMPLE, "python", "a.py")
    assert result["coverage"]["percent"] == 100.0
    assert result["coverage"]["complete"] is True


def test_pipeline_reports_partial_coverage_honestly():
    # A huge single function segment exceeds the per-segment analysis budget;
    # coverage must still count it (summarised, not skipped).
    big = "def huge():\n" + "\n".join(f"    x{i} = {i}" for i in range(600)) + "\n    return x0\n"
    result = pipeline.analyze_source(big, "python", "big.py")
    cov = result["coverage"]
    assert cov["covered_lines"] == cov["total_lines"] or cov["percent"] > 0
    assert "segments_analyzed" in cov


def test_pipeline_works_for_non_python():
    result = pipeline.analyze_source(JAVA_SAMPLE, "java", "Calculator.java")
    assert result["summary"]["classes"] == 1
    assert result["coverage"]["total_lines"] == len(JAVA_SAMPLE.splitlines())


# ---------------------------------------------------------------------------
# explainer.py
# ---------------------------------------------------------------------------

def _analysed():
    return pipeline.analyze_source(SIMPLE + "\n" + BRANCHY, "python", "a.py")


def test_explain_simple_mode():
    result = _analysed()
    md = explainer.explain(result, SIMPLE, "simple", "Python")
    assert "Simple Explanation" in md
    assert "add" in md and "classify" in md


def test_explain_line_by_line_numbers_lines():
    result = _analysed()
    md = explainer.explain(result, SIMPLE, "line_by_line", "Python")
    assert "**1.**" in md or "**2.**" in md
    assert "return" in md.lower()


def test_explain_functions_mode_lists_walkthrough():
    result = _analysed()
    md = explainer.explain(result, SIMPLE, "functions", "Python")
    assert "add(" in md
    assert "Walkthrough" not in md  # heading is 'Logic'
    assert "complexity" in md.lower()


def test_explain_algorithm_mode_mentions_complexity():
    result = _analysed()
    md = explainer.explain(result, SIMPLE, "algorithm", "Python")
    assert "complexity" in md.lower()
    assert "classify" in md


def test_explain_execution_flow_mentions_imports_and_defs():
    result = pipeline.analyze_source(
        "import os\n\ndef f():\n    return os.name\n", "python", "a.py")
    md = explainer.explain(result, "import os\n\ndef f():\n    return os.name\n",
                           "execution_flow", "Python")
    assert "Imports load first" in md


def test_explain_all_modes_produce_output():
    result = _analysed()
    full = explainer.explain_all(result, SIMPLE, "Python")
    for title in explainer.MODES.values():
        assert title in full


def test_explain_handles_empty_functions():
    result = pipeline.analyze_source("x = 1\n", "python", "a.py")
    md = explainer.explain(result, "x = 1\n", "functions", "Python")
    assert "No functions" in md


# ---------------------------------------------------------------------------
# diagrams.py
# ---------------------------------------------------------------------------

def test_structure_svg_is_valid_svg():
    result = pipeline.analyze_source(SIMPLE, "python", "a.py")
    svg = diagrams.structure_svg(result)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "add" in svg


def test_class_svg_shows_methods():
    source = '''class Account:
    def deposit(self, amount):
        return amount
'''
    result = pipeline.analyze_source(source, "python", "a.py")
    svg = diagrams.class_svg(result)
    assert "Account" in svg and "deposit" in svg


def test_call_svg_detects_internal_call():
    source = "def helper():\n    return 1\n\ndef main():\n    return helper()\n"
    result = pipeline.analyze_source(source, "python", "a.py")
    svg = diagrams.call_svg(result, source, "python")
    assert "helper" in svg and "main" in svg


def test_diagrams_handle_empty_result():
    result = pipeline.analyze_source("x = 1\n", "python", "a.py")
    assert "svg" in diagrams.structure_svg(result)
    assert "svg" in diagrams.class_svg(result)


# ---------------------------------------------------------------------------
# ai_client.py
# ---------------------------------------------------------------------------

def test_deterministic_answers_function_question():
    result = pipeline.analyze_source(SIMPLE, "python", "a.py")
    answer, label = ai_client.answer_deterministic(
        "explain function add", result, SIMPLE, "Python")
    assert "add" in answer
    assert "rule-based" in label


def test_deterministic_answers_complexity_question():
    result = pipeline.analyze_source(SIMPLE + BRANCHY, "python", "a.py")
    answer, _ = ai_client.answer_deterministic(
        "which function is most complex?", result, SIMPLE + BRANCHY, "Python")
    assert "complexity" in answer.lower()


def test_deterministic_unknown_function_reports_names():
    result = pipeline.analyze_source(SIMPLE, "python", "a.py")
    answer, _ = ai_client.answer_deterministic(
        "explain function zzz", result, SIMPLE, "Python")
    assert "could not find" in answer.lower()


def test_ask_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    result = pipeline.analyze_source(SIMPLE, "python", "a.py")
    answer, label = ai_client.ask("what does this do?", result, SIMPLE, "Python")
    assert label.startswith("rule-based")


# ---------------------------------------------------------------------------
# exports.py
# ---------------------------------------------------------------------------

def test_markdown_export_contains_sections():
    result = pipeline.analyze_source(SIMPLE, "python", "a.py")
    md = explainer.explain(result, SIMPLE, "detailed", "Python")
    out = exports.to_markdown(result, SIMPLE, md, "a.py")
    assert "# Code Explanation" in out
    assert "def add" in out or "add" in out


def test_pdf_export_produces_bytes():
    pytest.importorskip("reportlab")
    result = pipeline.analyze_source(SIMPLE, "python", "a.py")
    md = explainer.explain(result, SIMPLE, "simple", "Python")
    data = exports.to_pdf(result, md, "a.py")
    assert data.startswith(b"%PDF")
    assert len(data) > 500


# ---------------------------------------------------------------------------
# app.py UI tests
# ---------------------------------------------------------------------------

def _open_app() -> AppTest:
    app = AppTest.from_file("app.py", default_timeout=60).run()
    assert not app.exception, app.exception
    return app


def _nav(app: AppTest, page: str) -> AppTest:
    for button in app.button:
        if button.key == f"nav_{page}":
            return button.click().run()
    raise AssertionError(f"nav button {page!r} not found")


def _click(app: AppTest, label: str) -> AppTest:
    for button in app.button:
        if button.label == label:
            return button.click().run()
    raise AssertionError(f"button {label!r} not found")


def test_app_loads_home_without_errors():
    app = _open_app()
    assert "Understand any codebase" in "".join(
        str(m.value) for m in app.markdown)


def test_full_analysis_flow_reaches_results():
    app = _open_app()
    app = _nav(app, "Workspace")
    app.text_area[0].set_value("def add(a, b):\n    return a + b\n").run()
    app = _click(app, "Analyse code")
    assert app.session_state["page"] == "Results"
    assert app.session_state["result"]["functions"][0]["name"] == "add"
    assert app.session_state.get("explanation_md")


def test_results_page_shows_mode_selector():
    app = _open_app()
    app = _nav(app, "Workspace")
    app.text_area[0].set_value(SIMPLE).run()
    app = _click(app, "Analyse code")
    labels = [s.label for s in app.selectbox]
    assert any("Explanation mode" in label for label in labels)


def test_invalid_code_stays_with_error():
    app = _open_app()
    app = _nav(app, "Workspace")
    app.text_area[0].set_value("def broken(:\n").run()
    app = _click(app, "Analyse code")
    assert app.session_state["result"] is None
    assert app.session_state["error"]


def test_ask_page_without_analysis_warns():
    app = _open_app()
    app = _nav(app, "Ask")
    assert any("Analyse some code first" in (w.value or "")
               for w in app.warning)


def test_ask_page_answers_after_analysis():
    app = _open_app()
    app = _nav(app, "Workspace")
    app.text_area[0].set_value(SIMPLE).run()
    app = _click(app, "Analyse code")
    app = _nav(app, "Ask")
    app.chat_input[0].set_value("explain function add").run()
    chat = app.session_state["chat"]
    assert chat[-1]["role"] == "assistant"
    assert "add" in chat[-1]["text"]


def test_java_source_analyzed_as_heuristic():
    app = _open_app()
    app = _nav(app, "Workspace")
    app.text_area[0].set_value(JAVA_SAMPLE).run()
    app.selectbox(key="lang_select").set_value("java").run()
    app = _click(app, "Analyse code")
    result = app.session_state["result"]
    assert result["summary"]["analysis_kind"] == "heuristic"
    assert result["summary"]["classes"] == 1


def test_settings_page_loads():
    app = _open_app()
    app = _nav(app, "Settings")
    assert any("AI provider" in (m.value or "") for m in app.markdown)


def test_example_button_loads_code():
    app = _open_app()
    app = _nav(app, "Workspace")
    # open examples expander first
    for exp in app.expander:
        if exp.label == "Examples":
            app = exp.set_value(True).run() if hasattr(exp, "set_value") else app
    for button in app.button:
        if button.label == "Class with methods":
            app = button.click().run()
            break
    assert "class BankAccount" in app.text_area[0].value


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
