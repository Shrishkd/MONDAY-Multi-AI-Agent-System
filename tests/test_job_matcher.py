import pytest

from monday.agents.job_matcher import JDRequirements, score_requirements
from monday.experience import ExperienceBank
from monday.skills import appears_in, normalize, skill_matches

BANK = ExperienceBank.model_validate({
    "profile": {"name": "A", "email": "a@example.com"},
    "skills": {"core": ["Python", "PostgreSQL", "RAG (Hybrid Retrieval)", "Node.js",
                        "Fine-Tuning (LoRA/QLoRA)"],
               "Data & Vector Databases": ["Qdrant", "ChromaDB"]},
    "roles": [{
        "company": "X", "title": "Eng", "start": "2025",
        "facts": [{"id": "geo", "text": "Built geospatial search for bookings", "skills": ["MongoDB"]}],
    }],
})

JD = """We need Python, Postgres and RAG experience. You'll build geospatial search.
Nice to have: Kubernetes, MongoDB."""


def req(must, nice=(), years=None, seniority="entry"):
    return JDRequirements(role_title="Eng", seniority=seniority, min_years_experience=years,
                          must_have_skills=list(must), nice_to_have_skills=list(nice),
                          responsibilities=[], work_mode="unknown")


@pytest.mark.parametrize("required, candidate, expected", [
    ("Postgres", "PostgreSQL", True),
    ("NodeJS", "Node.js", True),
    ("RAG", "RAG (Hybrid Retrieval)", True),
    ("Python programming", "Python", True),
    ("Java", "JavaScript", False),
    ("Go", "Google Cloud Platform", False),
    ("SQL", "PostgreSQL", False),
    ("fine-tuning models", "Fine-Tuning (LoRA/QLoRA)", True),
    ("Hands-on experience with Docker", "Docker", True),
    ("Experience", "Docker", False),
])
def test_skill_matching(required, candidate, expected):
    assert skill_matches(required, candidate) is expected


def test_normalize_handles_symbols():
    assert normalize("C++ & C#") == "c++ and c#"
    assert normalize("LLMs") == "large language models"


def test_appears_in_catches_invented_skills():
    assert appears_in("PostgreSQL", JD)       # JD says "Postgres"
    assert appears_in("Geospatial Search", JD)
    assert not appears_in("Terraform", JD)


def test_scores_must_and_nice_with_evidence():
    report = score_requirements(req(["Python", "Postgres", "RAG", "geospatial search"],
                                    ["Kubernetes", "MongoDB"]), JD, BANK)
    assert all(m.matched for m in report.must_have)
    assert {m.skill: m.matched for m in report.nice_to_have} == {"Kubernetes": False, "MongoDB": True}
    assert "geo" in report.must_have[3].evidence
    assert report.score == pytest.approx(100 * (0.75 * 1 + 0.25 * 0.5))
    assert report.recommendation == "pursue"


def test_invented_skills_are_dropped_not_scored():
    report = score_requirements(req(["Python", "Terraform"]), JD, BANK)
    assert [m.skill for m in report.must_have] == ["Python"]
    assert report.dropped_skills == ["Terraform"]
    assert report.score == 100
    assert any("doesn't mention" in f for f in report.flags)


def test_gaps_lower_score_and_flags_seniority():
    report = score_requirements(req(["Python", "Kubernetes"], years=5, seniority="senior"), JD, BANK)
    assert report.gaps == ["Kubernetes"]
    assert report.score == 50 and report.recommendation == "stretch"
    assert any("5+ years" in f for f in report.flags)


def test_duplicates_and_overlap_between_lists_removed():
    report = score_requirements(req(["Python", "python"], ["Python", "MongoDB"]), JD, BANK)
    assert [m.skill for m in report.must_have] == ["Python"]
    assert [m.skill for m in report.nice_to_have] == ["MongoDB"]


def test_skill_group_name_counts_as_evidence():
    jd = "Experience with vector databases is required."
    report = score_requirements(req(["vector databases"]), jd, BANK)
    assert report.must_have[0].matched
    assert report.must_have[0].evidence == ["group: Data & Vector Databases"]


def test_alternatives_are_satisfied_by_any_option():
    jd = "Must know Go or Python. Also Rust or Haskell."
    report = score_requirements(req(["Go or Python", "Rust or Haskell"]), jd, BANK)
    assert [m.matched for m in report.must_have] == [True, False]
    assert "Python" in report.must_have[0].evidence


def test_empty_extraction_is_flagged():
    report = score_requirements(req([]), JD, BANK)
    assert report.score == 0
    assert "meaningless" in report.flags[0]
