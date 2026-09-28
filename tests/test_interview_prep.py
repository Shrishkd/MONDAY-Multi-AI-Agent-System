from types import SimpleNamespace

import pytest

from monday.agents.interview_prep import (BriefPoint, CompanyBrief, PrepOutput, Question, StarOutline,
                                          _drop_failing, check_brief, check_prep)
from monday.config import Config, read_env_file
from monday.experience import load_bank
from monday.pipeline import prep_markdown, save_story
from monday.research import Researcher, ResearchError, mentions_company
from monday.resume_import import bank_yaml
from tests.test_resume_tailor import BANK, REPORT

SOURCES = [
    {"title": "Acme - About", "url": "https://acme.test/about", "content": "Acme serves 2,000 hospitals."},
    {"title": "Acme blog", "url": "https://acme.test/blog", "content": "We run Python and PostgreSQL."},
]


# --- company brief ---------------------------------------------------------------

def test_brief_keeps_sourced_points_and_drops_the_rest():
    brief = CompanyBrief(
        overview=[BriefPoint(text="Serves 2,000 hospitals", source_ids=[1]),
                  BriefPoint(text="Serves 5,000 hospitals", source_ids=[1]),      # number not in source
                  BriefPoint(text="Founded by ex-Googlers", source_ids=[])],     # no source
        tech_and_engineering=[BriefPoint(text="Uses PostgreSQL", source_ids=[2, 9])],  # 9 doesn't exist
    )
    kept, dropped = check_brief(brief, SOURCES)
    assert [p.text for p in kept.overview] == ["Serves 2,000 hospitals"]
    assert kept.tech_and_engineering[0].source_ids == [2]
    assert len(dropped) == 2


# --- questions & STAR ------------------------------------------------------------

def q(points, fact_ids=("r2",), category="technical"):
    return Question(category=category, question="Q?", why_they_ask="why", talking_points=list(points),
                    fact_ids=list(fact_ids))


def star(**kw):
    base = dict(title="Latency", fact_ids=["r2"], situation="p95 was 800ms", task="get it down",
                action="profiled", result="120ms", fill_in=["What broke first?"])
    return StarOutline(**{**base, **kw})


def prep(questions=(), stars=()):
    return PrepOutput(questions=list(questions), star_outlines=list(stars), questions_to_ask=["What's next?"])


def test_grounded_prep_passes():
    out = prep([q(["Cut p95 latency from 800ms to 120ms"]),
                q(["Haven't run Kubernetes; used containers via Docker-style deploys"], (), "gap")],
               [star()])
    assert check_prep(out, BANK, REPORT, allowed_text="JD text").problems == []


def test_invented_numbers_unknown_ids_and_gap_claims_flagged():
    out = prep([q(["Cut latency by 85%"]),
                q(["Explain it"], ["nope"]),
                q(["I have deployed services on Kubernetes at scale"], (), "gap")],
               [star(result="cut it by 90%"), star(fact_ids=[])])
    problems = check_prep(out, BANK, REPORT, allowed_text="").problems
    joined = " | ".join(problems)
    assert "question 1" in joined and "85" in joined
    assert "question 2: cites unknown ids" in joined
    assert "question 3: claims experience with 'Kubernetes'" in joined
    assert "STAR outline 1" in joined and "STAR outline 2" in joined


def test_numbers_from_jd_or_brief_are_allowed():
    out = prep([q(["Their team serves 2,000 hospitals"], (), "company")])
    assert check_prep(out, BANK, REPORT, allowed_text="Acme serves 2,000 hospitals.").problems == []


def test_failing_items_dropped_after_retry():
    out = prep([q(["ok"]), q(["bad 85%"])], [star(), star(result="90%")])
    cleaned = _drop_failing(out, check_prep(out, BANK, REPORT, ""))
    assert [x.talking_points for x in cleaned.questions] == [["ok"]]
    assert len(cleaned.star_outlines) == 1


# --- research --------------------------------------------------------------------

def test_company_name_matching_is_loose():
    assert mentions_company("Square Yards", "SquareYards raises funding")
    assert mentions_company("SquareYards", "square yards property portal")
    assert not mentions_company("Acme", "Globex quarterly results")


class FakeWeb:
    def __init__(self, fetch_fails=False, search_fails=False):
        self.fetch_fails, self.search_fails = fetch_fails, search_fails

    def web_fetch(self, url):
        if self.fetch_fails:
            raise RuntimeError("404")
        return SimpleNamespace(title="Acme home", content="Acme builds robots")

    def web_search(self, query, max_results=3):
        if self.search_fails:
            raise RuntimeError("down")
        return SimpleNamespace(results=[
            SimpleNamespace(title="Acme raises", url=f"https://news.test/{hash(query)}", content="Acme news"),
            SimpleNamespace(title="Other Co", url="https://other.test", content="unrelated company"),
        ])


def test_research_filters_irrelevant_and_survives_fetch_failure():
    sources, notes = Researcher(None, client=FakeWeb(fetch_fails=True)).gather("Acme", "Eng", "https://acme.test")
    assert all("Acme" in s.title for s in sources) and len(sources) == 4
    assert any("could not fetch" in n for n in notes)
    assert any("doesn't mention Acme" in n for n in notes)


def test_research_errors_when_everything_fails():
    with pytest.raises(ResearchError):
        Researcher(None, client=FakeWeb(search_fails=True)).gather("Acme", "Eng")


def test_research_needs_a_key():
    with pytest.raises(ResearchError, match=".env"):
        Researcher(None)


# --- config, stories, export ---------------------------------------------------

def test_env_file_parsing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nOLLAMA_API_KEY = 'abc.123'\nEMPTY=\nnot a pair\n", encoding="utf-8")
    assert read_env_file(env) == {"OLLAMA_API_KEY": "abc.123", "EMPTY": ""}
    assert read_env_file(tmp_path / "missing") == {}


def test_config_repr_hides_the_key():
    assert "secret" not in repr(Config(ollama_api_key="secret"))


def test_save_story_appends_with_backup(tmp_path):
    cfg = Config(experience_bank=tmp_path / "bank.yaml")
    cfg.experience_bank.write_text(bank_yaml(BANK), encoding="utf-8")
    story = {"title": "Latency fix", "situation": "s", "task": "t", "action": "a", "result": "r",
             "skills": [], "fact_ids": ["r2"]}
    backup = save_story(cfg, story)
    save_story(cfg, story)
    ids = [s.id for s in load_bank(cfg.experience_bank).stories]
    assert ids == ["story-latency-fix", "story-latency-fix-2"]
    assert backup.exists()


def test_markdown_export_links_sources():
    content = {
        "brief": {"overview": [{"text": "Serves hospitals", "source_ids": [1]}]},
        "sources": [{"id": 1, "title": "About", "url": "https://acme.test/about"}],
        "questions": [q(["Cut p95 latency"]).model_dump()],
        "star_outlines": [star().model_dump()], "questions_to_ask": ["What's next?"],
    }
    md = prep_markdown({"title": "Eng", "company": "Acme"}, content)
    assert "# Interview prep: Eng @ Acme" in md
    assert "[1](https://acme.test/about)" in md and "- [ ] What broke first?" in md
