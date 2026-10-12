from types import SimpleNamespace

import pytest

from monday.agents.cover_letter import PersonalLine, _context, check_line, fill, run

SOURCES = [
    {"title": "Acme - About", "url": "https://acme.test/about", "content": "Acme serves 2,000 hospitals."},
    {"title": "Acme blog", "url": "https://acme.test/blog", "content": "We moved triage to on-device LLMs."},
]

TEMPLATE = ("Hey,\n\nI saw that {company} is hiring for the {role} role.\n\n[personal line]\n\n"
            "I am Shrish and I build AI agents.\n\nYour team at {company}.\n")

GOOD = PersonalLine(line="Moving triage to on-device LLMs for 2,000 hospitals is the kind of problem I want "
                         "to spend my days on.", source_ids=[1, 2])


def test_grounded_line_passes():
    c = check_line(GOOD, SOURCES)
    assert c.problems == [] and c.warnings == []


def test_uncited_invented_numbers_and_long_lines_fail():
    assert any("doesn't cite" in p for p in check_line(PersonalLine(line="Great company.", source_ids=[9]),
                                                       SOURCES).problems)
    assert any("5000" in p for p in check_line(GOOD.model_copy(update={"line": "Serving 5,000 hospitals."}),
                                                SOURCES).problems)
    assert any("too long" in p for p in check_line(PersonalLine(line="x" * 400, source_ids=[1]),
                                                   SOURCES).problems)
    assert any("placeholders" in p for p in check_line(PersonalLine(line="I like [Company].", source_ids=[1]),
                                                       SOURCES).problems)


def test_numbers_from_the_jd_are_allowed():
    line = PersonalLine(line="A 6-person team shipping LLM triage is exactly where I want to be.", source_ids=[2])
    assert check_line(line, SOURCES, jd_text="Join our 6-person AI team.").problems == []


def test_no_research_is_a_warning_not_a_failure():
    c = check_line(PersonalLine(line="Building LLM triage tools is what I want to work on.", source_ids=[]), [])
    assert c.problems == [] and any("no web research" in w for w in c.warnings)


def test_cliches_are_flagged():
    c = check_line(PersonalLine(line="I am passionate about your cutting-edge LLM triage.", source_ids=[2]), SOURCES)
    assert len(c.warnings) == 2


def test_fill_changes_only_the_markers():
    letter = fill(TEMPLATE, company="Acme", role="AI Engineer", line="  The line.  ")
    assert letter == ("Hey,\n\nI saw that Acme is hiring for the AI Engineer role.\n\nThe line.\n\n"
                      "I am Shrish and I build AI agents.\n\nYour team at Acme.\n")


def test_context_is_the_paragraphs_around_the_slot():
    assert _context(TEMPLATE) == ("I saw that {company} is hiring for the {role} role.",
                                  "I am Shrish and I build AI agents.")


class StubLLM:
    def __init__(self, *outputs):
        self.outputs, self.prompts = list(outputs), []

    def structured(self, agent, system, prompt, schema, **kw):
        self.prompts.append(prompt)
        return SimpleNamespace(output=self.outputs.pop(0), model="stub")


def test_run_retries_once_on_a_failed_check():
    llm = StubLLM(PersonalLine(line="Serving 5,000 hospitals.", source_ids=[1]), GOOD)
    out, check, model = run(llm, TEMPLATE, company="Acme", role="AI Engineer", jd_text="", sources=SOURCES)
    assert out.line == GOOD.line and check.problems == [] and model == "stub"
    assert "5000" in llm.prompts[1] and "Acme is hiring" in llm.prompts[0]


def test_template_without_the_slot_is_refused():
    with pytest.raises(ValueError, match="personal line"):
        run(StubLLM(), "Hi {company}", company="Acme", role="AI Engineer", jd_text="", sources=SOURCES)
