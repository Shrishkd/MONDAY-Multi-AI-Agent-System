"""Job Matcher: how well does a JD fit what you've actually done?

Step 1 (LLM): extract requirements from the JD into a fixed schema.
Step 2 (check): drop any extracted skill that doesn't appear in the JD text.
Step 3 (code): match requirements against the Experience Bank and score.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from monday.experience import ExperienceBank
from monday.llm import LLM
from monday.skills import appears_in, normalize, skill_matches

AGENT = "job_matcher"

SYSTEM = """You extract hiring requirements from job descriptions.
Rules:
- Copy skill and technology names exactly as they are written in the job description.
- Only list skills that the job description actually mentions. Never add skills of your own.
- must_have_skills: required / "must have" / "you have" items.
- nice_to_have_skills: "preferred" / "bonus" / "plus" items.
- Each skill is a short name (1-4 words), e.g. "Python", "PostgreSQL", "REST APIs".
- Alternatives are ONE requirement, not several: "vector databases (Qdrant, Pinecone, or pgvector)"
  becomes "vector databases"; "Go or Rust" becomes "Go or Rust".
- min_years_experience: the smallest number of years asked for, or null if none is stated."""


class JDRequirements(BaseModel):
    role_title: str
    seniority: Literal["intern", "entry", "mid", "senior", "lead", "unknown"]
    min_years_experience: int | None
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    responsibilities: list[str] = Field(description="Up to 6 short phrases.")
    work_mode: Literal["remote", "hybrid", "onsite", "unknown"]


class SkillMatch(BaseModel):
    skill: str
    matched: bool
    evidence: list[str] = Field(default=[], description="Bank skills or fact ids that back the match.")


class MatchReport(BaseModel):
    score: float
    recommendation: Literal["pursue", "stretch", "skip"]
    requirements: JDRequirements
    must_have: list[SkillMatch]
    nice_to_have: list[SkillMatch]
    dropped_skills: list[str] = Field(default=[], description="Extracted by the model but not in the JD.")
    flags: list[str] = []

    @property
    def gaps(self) -> list[str]:
        return [m.skill for m in self.must_have if not m.matched]


def match_skill(skill: str, bank: ExperienceBank) -> SkillMatch:
    # "Go or Rust" is satisfied by either one.
    options = [skill] + [p for p in re.split(r"\s+or\s+", skill, flags=re.I) if p != skill]
    evidence = sorted({s for s in bank.all_skills() if any(skill_matches(o, s) for o in options)})
    # "vector databases" should hit the "Data & Vector Databases" skill group.
    evidence += [f"group: {c}" for c in bank.skills if any(skill_matches(o, c) for o in options)]
    evidence += [f.id for f in bank.facts()
                 if any(skill_matches(o, s) for o in options for s in f.skills)
                 or any(appears_in(o, f.text) for o in options)]
    return SkillMatch(skill=skill, matched=bool(evidence), evidence=evidence)


def _dedupe(skills: list[str]) -> list[str]:
    seen, out = set(), []
    for s in skills:
        key = normalize(s)
        if key and key not in seen:
            seen.add(key)
            out.append(s.strip())
    return out


def score_requirements(req: JDRequirements, jd_text: str, bank: ExperienceBank) -> MatchReport:
    """Deterministic part of the matcher; also what the tests exercise."""
    dropped = []

    def grounded(skills: list[str]) -> list[str]:
        keep = []
        for s in _dedupe(skills):
            (keep if appears_in(s, jd_text) else dropped).append(s)
        return keep

    must = [match_skill(s, bank) for s in grounded(req.must_have_skills)]
    must_keys = {normalize(m.skill) for m in must}
    nice = [match_skill(s, bank) for s in grounded(req.nice_to_have_skills)
            if normalize(s) not in must_keys]

    coverage = lambda ms: sum(m.matched for m in ms) / len(ms) if ms else None  # noqa: E731
    must_cov, nice_cov = coverage(must), coverage(nice)
    if must_cov is None and nice_cov is None:
        score = 0.0
    elif nice_cov is None:
        score = 100 * must_cov
    elif must_cov is None:
        score = 100 * nice_cov
    else:
        score = 100 * (0.75 * must_cov + 0.25 * nice_cov)

    flags = []
    if not must and not nice:
        flags.append("No skills could be extracted from this JD - score is meaningless, read it yourself.")
    if req.min_years_experience:
        flags.append(f"Asks for {req.min_years_experience}+ years of experience.")
    if req.seniority in ("senior", "lead"):
        flags.append(f"Seniority looks {req.seniority}.")
    if dropped:
        flags.append(f"Ignored {len(dropped)} skill(s) the model listed but the JD doesn't mention.")

    recommendation = "pursue" if score >= 65 else "stretch" if score >= 40 else "skip"
    return MatchReport(score=round(score, 1), recommendation=recommendation, requirements=req,
                       must_have=must, nice_to_have=nice, dropped_skills=dropped, flags=flags)


def run(llm: LLM, jd_text: str, bank: ExperienceBank) -> tuple[MatchReport, str]:
    """Returns the report and the model that extracted the requirements."""
    result = llm.structured(AGENT, SYSTEM, f"Job description:\n\n{jd_text}", JDRequirements)
    return score_requirements(result.output, jd_text, bank), result.model
