from monday.pipeline import _section, edit_warnings
from tests.test_resume_tailor import BANK, REPORT


def content(text: str, summary: str = "Engineer with 10,000+ users.") -> dict:
    return {"summary": summary, "sections": [
        {"key": "role:0", "label": "Engineer @ Acme",
         "bullets": [{"text": text, "fact_ids": ["r2"], "fallback": False}]},
    ]}


def test_section_keys_resolve_to_bank_entries():
    assert _section("role:0", BANK).source.company == "Acme"
    assert _section("project:1", BANK).source.name == "Game"


def test_faithful_edit_has_no_warnings():
    assert edit_warnings(content("Cut **p95 latency** from 800ms to 120ms"), BANK, REPORT) == []


def test_edit_that_invents_a_number_is_flagged():
    warnings = edit_warnings(content("Cut p95 latency by 85%"), BANK, REPORT)
    assert any("85" in w for w in warnings)


def test_summary_edit_checked_too():
    warnings = edit_warnings(content("Cut p95 latency from 800ms to 120ms", "Served 1M users"), BANK, REPORT)
    assert any(w.startswith("summary") for w in warnings)
