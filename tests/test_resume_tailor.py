from monday.agents.job_matcher import JDRequirements, MatchReport, SkillMatch
from monday.agents.resume_tailor import (Bullet, SectionDraft, TailorOutput, _assemble,
                                         check_bullet, check_summary, claimable_terms,
                                         numbers, order_skills, plan_sections)
from monday.experience import ExperienceBank

BANK = ExperienceBank.model_validate({
    "profile": {"name": "A", "email": "a@example.com", "summary": "Engineer with 10,000+ users."},
    "skills": {"Backend": ["Flask", "FastAPI", "PostgreSQL"]},
    "roles": [{
        "company": "Acme", "title": "Engineer", "start": "2025",
        "facts": [
            {"id": "r1", "text": "Built **FastAPI** services for bookings", "skills": ["FastAPI", "System Design"]},
            {"id": "r2", "text": "Cut p95 latency from 800ms to 120ms", "metrics": ["800ms -> 120ms"]},
        ],
    }],
    "projects": [
        {"name": "Chatbot", "tech_stack": ["LangGraph", "FAISS"],
         "facts": [{"id": "p1", "text": "Built a RAG chatbot over 3,500 pages"}]},
        {"name": "Game", "facts": [{"id": "g1", "text": "Made a puzzle game in Unity"}]},
    ],
})

REPORT = MatchReport(
    score=80, recommendation="pursue",
    requirements=JDRequirements(role_title="AI Eng", seniority="entry", min_years_experience=None,
                                must_have_skills=[], nice_to_have_skills=[], responsibilities=[],
                                work_mode="unknown"),
    must_have=[SkillMatch(skill="LangGraph", matched=True), SkillMatch(skill="PostgreSQL", matched=True)],
    nice_to_have=[SkillMatch(skill="Kubernetes", matched=False)],
)

SECTIONS = plan_sections(BANK, REPORT)
ROLE, CHATBOT = SECTIONS[0], next(s for s in SECTIONS if s.key == "project:0")
TERMS = claimable_terms(BANK, REPORT)


def test_numbers_normalise_thousands_separators():
    assert numbers("3,500 pages, 0.083 to 0.917, 7/7") == {"3500", "0.083", "0.917", "7"}


def test_faithful_rewrite_passes():
    b = Bullet(fact_ids=["r2"], text="Reduced **p95 latency** from 800ms to 120ms")
    assert check_bullet(b, ROLE, TERMS) == []


def test_invented_number_rejected():
    b = Bullet(fact_ids=["r2"], text="Reduced p95 latency by 85% (800ms to 120ms)")
    assert any("85" in p for p in check_bullet(b, ROLE, TERMS))


def test_unsupported_jd_skill_rejected():
    b = Bullet(fact_ids=["r1"], text="Built FastAPI services on Kubernetes")
    assert any("Kubernetes" in p for p in check_bullet(b, ROLE, TERMS))


def test_skill_tags_are_not_evidence():
    # "System Design" is only a matching tag on r1 - the user never wrote it.
    b = Bullet(fact_ids=["r1"], text="Built FastAPI services for bookings, applying System Design")
    assert any("System Design" in p for p in check_bullet(b, ROLE, TERMS))


def test_project_tech_stack_counts_as_evidence():
    b = Bullet(fact_ids=["p1"], text="Built a **LangGraph** RAG chatbot over 3,500 pages")
    assert check_bullet(b, CHATBOT, TERMS) == []


def test_fact_from_another_section_rejected():
    b = Bullet(fact_ids=["p1"], text="Built a RAG chatbot over 3,500 pages")
    assert any("not in this section" in p for p in check_bullet(b, ROLE, TERMS))


def test_overlong_bullet_rejected():
    b = Bullet(fact_ids=["r1"], text="Built FastAPI services for bookings " + "really " * 40)
    assert any("too long" in p for p in check_bullet(b, ROLE, TERMS))


def test_relevant_projects_first_and_skills_reordered():
    assert [s.key for s in SECTIONS] == ["role:0", "project:0", "project:1"]
    assert order_skills(BANK.skills, REPORT)["Backend"][0] == "PostgreSQL"


def test_failed_bullets_fall_back_to_original_wording():
    out = TailorOutput(
        summary="Engineer with 99 years of experience.", summary_fact_ids=[],
        sections=[SectionDraft(key="role:0", bullets=[
            Bullet(fact_ids=["r1"], text="Built **FastAPI** booking services"),
            Bullet(fact_ids=["r2"], text="Cut latency by 90%"),            # invented number
        ])],
    )
    result = _assemble(out, SECTIONS, TERMS, BANK, REPORT)
    role_bullets = dict((s.key, b) for s, b in result.sections)["role:0"]
    assert [(b.text, b.fallback) for b in role_bullets] == [
        ("Built **FastAPI** booking services", False),
        ("Cut p95 latency from 800ms to 120ms", True),
    ]
    project_bullets = dict((s.key, b) for s, b in result.sections)["project:0"]
    assert project_bullets[0].fallback                            # section missing -> original
    assert result.summary_fallback and result.summary == BANK.profile.summary
    assert any("90" in v for v in result.violations)


def test_summary_numbers_checked_against_facts_and_original_summary():
    assert check_summary("Served 10,000+ users.", [], BANK) == []
    assert check_summary("Served 20,000 users.", [], BANK)
