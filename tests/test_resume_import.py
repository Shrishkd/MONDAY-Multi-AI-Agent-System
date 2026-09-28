from pathlib import Path

from monday.config import Config
from monday.experience import ExperienceBank, load_bank
from monday.resume_import import (_unique_ids, apply, bank_yaml, check_template, diff,
                                  grounding_warnings, keep_ids, keep_stories)

MACROS = "\n".join(rf"\newcommand{{\{m}}}{{x}}" for m in (
    "resumeSubheading", "resumeItem", "resumeProjectHeading", "resumeSubHeadingListStart",
    "resumeSubHeadingListEnd", "resumeItemListStart", "resumeItemListEnd",
    "resumeProjectListStart", "resumeProjectListEnd"))
BODY = r"""
\begin{document}
\section{Summary} s
\section{Experience} e
\section{Projects} \faFileAlt\ Report \faExternalLinkAlt\ Demo \faGithub
\section{Skills} k
\end{document}
"""


def bank(facts, stories=()):
    return ExperienceBank.model_validate({
        "profile": {"name": "A", "email": "a@x.com", "summary": "Served 10,000+ users."},
        "roles": [{"company": "X", "title": "Eng", "start": "2025", "facts": facts}],
        "stories": list(stories),
    })


def test_template_icons_fixed_and_macros_found():
    check = check_template(MACROS + BODY)
    assert r"\faFile*" in check.tex and r"\faExternalLink*" in check.tex and r"\faGithub" in check.tex
    assert len(check.fixes) == 2
    assert check.problems == []


def test_template_missing_macros_and_sections_reported():
    # \resumeItemListStart alone must not count as \resumeItem being defined
    tex = "\\newcommand{\\resumeItemListStart}{x}\n\\begin{document}\n\\section{Education} e\n\\end{document}\n"
    macros, sections = check_template(tex).problems
    assert "resumeItem," in macros and "resumeItemListStart" not in macros
    assert "Summary, Experience, Projects, Skills" in sections


def test_duplicate_ids_made_unique():
    raw = {"roles": [{"facts": [{"id": "a"}, {"id": "a"}, {"id": ""}]}]}
    assert [f["id"] for f in _unique_ids(raw)["roles"][0]["facts"]] == ["a", "a-2", "fact-2"]


def test_grounding_flags_numbers_not_in_source():
    b = bank([{"id": "f", "text": "Cut latency by 40%", "metrics": ["800ms -> 120ms"]}])
    source = "a@x.com Served 10,000+ users. Cut latency from 800ms to 120ms"
    assert grounding_warnings(b, source) == ["f: numbers not in your resume: ['40']"]


def test_stories_kept_only_if_their_facts_survive():
    story = {"id": "s", "title": "t", "situation": "s", "task": "t", "action": "a", "result": "r",
             "fact_ids": ["f"]}
    old = bank([{"id": "f", "text": "x"}], [story])
    assert keep_stories(bank([{"id": "f", "text": "x2"}]), old).stories[0].id == "s"
    assert keep_stories(bank([{"id": "g", "text": "y"}]), old).stories == []


OLD = bank([
    {"id": "lat", "text": "Cut **p95 latency** from 800ms to 120ms with a Redis cache"},
    {"id": "etl", "text": "Rebuilt the nightly ETL as idempotent Celery tasks"},
    {"id": "gone", "text": "Maintained the legacy PHP admin panel"},
])
NEW = bank([
    {"id": "x1", "text": "Cut p95 latency from 800ms to 120ms with a Redis cache."},       # same, new id
    {"id": "x2", "text": "Rebuilt the nightly ETL pipeline as idempotent Celery tasks"},  # reworded
    {"id": "x3", "text": "Launched a GraphQL gateway for mobile clients"},                 # new
])


def test_diff_ignores_id_and_markup_changes():
    assert diff(OLD, NEW) == {
        "added": ["x3: Launched a GraphQL gateway for mobile clients"],
        "removed": ["gone: Maintained the legacy PHP admin panel"],
        "reworded": ["etl: Rebuilt the nightly ETL as idempotent Celery tasks  ->  "
                     "Rebuilt the nightly ETL pipeline as idempotent Celery tasks"],
    }


def test_reimport_keeps_old_ids():
    assert list(keep_ids(NEW, OLD).fact_index()) == ["lat", "etl", "x3"]


def test_apply_backs_up_then_writes(tmp_path: Path):
    cfg = Config(experience_bank=tmp_path / "bank.yaml", resume_template=tmp_path / "resume" / "base.tex")
    cfg.resume_template.parent.mkdir()
    cfg.resume_template.write_text("OLD TEX", encoding="utf-8")
    cfg.experience_bank.write_text(bank_yaml(bank([{"id": "old", "text": "old"}])), encoding="utf-8")

    new = bank([{"id": "new", "text": "new fact"}])
    backup = apply(cfg, new, "NEW TEX", ("upload.tex", b"raw upload"))

    assert (backup / "base.tex").read_text(encoding="utf-8") == "OLD TEX"
    assert "old" in (backup / "bank.yaml").read_text(encoding="utf-8")
    assert cfg.resume_template.read_text(encoding="utf-8") == "NEW TEX"
    assert list(load_bank(cfg.experience_bank).fact_index()) == ["new"]
    assert (cfg.resume_template.parent / "upload.tex").read_bytes() == b"raw upload"
