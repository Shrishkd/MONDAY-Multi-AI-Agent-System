"""The Experience Bank: the single source of truth about you.

Agents may select, reorder and rephrase these facts - never invent new ones. Every fact
has a stable id so tailored bullets can cite where they came from, and a validator can
reject any bullet that cites nothing.
"""

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, BeforeValidator, Field, model_validator

# YAML turns `2019` into an int and `2023-07-01` into a date; keep them as text.
DateText = Annotated[str, BeforeValidator(str)]


class Fact(BaseModel):
    id: str
    text: str = Field(description="What you did, true and specific.")
    skills: list[str] = []
    metrics: list[str] = Field(default=[], description='Real numbers, e.g. "p95 latency 800ms -> 120ms".')


class Role(BaseModel):
    company: str
    title: str
    location: str | None = None
    start: DateText
    end: DateText = "present"
    facts: list[Fact] = []


class Project(BaseModel):
    name: str
    dates: DateText | None = None
    links: dict[str, str] = Field(default={}, description='Label -> URL, e.g. {"GitHub": "https://..."}.')
    tech_stack: list[str] = []
    facts: list[Fact] = []


class Certification(BaseModel):
    name: str
    issuer: str
    year: DateText | None = None
    url: str | None = None


class Education(BaseModel):
    school: str
    degree: str
    start: DateText | None = None
    end: DateText | None = None
    details: list[str] = []


class Story(BaseModel):
    """A STAR story for interview prep, grounded in facts from the bank."""

    id: str
    title: str
    situation: str
    task: str
    action: str
    result: str
    skills: list[str] = []
    fact_ids: list[str] = []


class Profile(BaseModel):
    name: str
    email: str
    phone: str | None = None
    location: str | None = None
    headline: str | None = None
    summary: str | None = None
    links: list[str] = []


class ExperienceBank(BaseModel):
    profile: Profile
    skills: dict[str, list[str]] = Field(default={}, description="Category -> skills.")
    roles: list[Role] = []
    projects: list[Project] = []
    education: list[Education] = []
    certifications: list[Certification] = []
    achievements: list[str] = []
    stories: list[Story] = []

    def facts(self) -> list[Fact]:
        return [f for r in self.roles for f in r.facts] + [f for p in self.projects for f in p.facts]

    def fact_index(self) -> dict[str, Fact]:
        return {f.id: f for f in self.facts()}

    def all_skills(self) -> set[str]:
        skills = {s for group in self.skills.values() for s in group}
        skills |= {s for f in self.facts() for s in f.skills}
        skills |= {s for p in self.projects for s in p.tech_stack}
        return skills

    @model_validator(mode="after")
    def _check_ids(self) -> "ExperienceBank":
        seen: set[str] = set()
        for fact in self.facts():
            if fact.id in seen:
                raise ValueError(f"Duplicate fact id '{fact.id}' - fact ids must be unique")
            seen.add(fact.id)
        for story in self.stories:
            missing = [fid for fid in story.fact_ids if fid not in seen]
            if missing:
                raise ValueError(f"Story '{story.id}' cites unknown fact ids: {missing}")
        return self


def load_bank(path: Path) -> ExperienceBank:
    if not path.exists():
        raise FileNotFoundError(
            f"No Experience Bank at {path}. Run `monday init`, then fill it in with your real experience."
        )
    return ExperienceBank.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
