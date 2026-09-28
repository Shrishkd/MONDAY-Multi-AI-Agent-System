"""Resume Tailor: rewrite bullets for one JD without inventing anything.

The model rewrites; code verifies. Every bullet must cite facts from the Experience
Bank, may not introduce numbers or skills its cited facts don't support, and must fit on
the page. A bullet that still fails after one correction round falls back to the
original fact wording - the output is never an unverified claim.
"""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from monday.agents.job_matcher import MatchReport
from monday.experience import ExperienceBank, Fact, Project, Role
from monday.llm import LLM
from monday.skills import appears_in, normalize, skill_matches

AGENT = "resume_tailor"
MAX_BULLET_CHARS = 230
MAX_SUMMARY_CHARS = 420
MAX_JD_CHARS = 8000

SYSTEM = f"""You tailor resume bullets to a job description.

Hard rules - a bullet that breaks one is thrown away:
- Use ONLY what the cited facts say. Rephrase, reorder, merge and emphasise; never add.
- Never add a number, tool, skill, technology, employer or result that is not in the cited facts.
- Every bullet cites the fact ids it is based on, and only ids listed under that same section.
- Keep every number exactly as written in the facts.
- Write exactly the number of bullets requested for each section.
- Each bullet: one line of past-tense or present-tense impact, at most {MAX_BULLET_CHARS} characters.
- Wrap 1-3 key terms per bullet in **double asterisks** (technologies and metrics), like **LangGraph**.
- Do not tack lists of skill names onto a bullet; the Skills section already lists them.

Goal: a recruiter or ATS scanning for this job's requirements finds them fast. Lead with
what the job cares about, and use the job's wording when the cited fact genuinely supports it.
The summary is 2-3 sentences, at most {MAX_SUMMARY_CHARS} characters, and follows the same rules
(cite the facts it uses)."""


class Bullet(BaseModel):
    fact_ids: list[str]
    text: str


class SectionDraft(BaseModel):
    key: str = Field(description='Section key exactly as given, e.g. "role:0" or "project:2".')
    bullets: list[Bullet]


class TailorOutput(BaseModel):
    summary: str
    summary_fact_ids: list[str]
    sections: list[SectionDraft]


@dataclass
class Section:
    key: str
    source: Role | Project
    n_bullets: int

    @property
    def facts(self) -> list[Fact]:
        return self.source.facts

    @property
    def label(self) -> str:
        s = self.source
        return f"{s.title} @ {s.company}" if isinstance(s, Role) else s.name


@dataclass
class TailoredBullet:
    text: str
    fact_ids: list[str]
    fallback: bool = False       # True -> original wording used because the rewrite failed checks


@dataclass
class TailorResult:
    summary: str
    summary_fallback: bool
    sections: list[tuple[Section, list[TailoredBullet]]]
    skills: dict[str, list[str]]
    violations: list[str] = field(default_factory=list)   # what was rejected, for the reviewer
    model: str = ""


# --- deterministic checks ------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


# Models write "3 800" (narrow no-break space) where a source wrote "3,800".
_THOUSANDS_SPACE = re.compile(r"(?<=\d)[   ](?=\d{3}(?!\d))")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(_THOUSANDS_SPACE.sub("", text))}


def _fact_text(facts: list[Fact], with_tags: bool = True) -> str:
    """Fact wording. Skill tags are matching aids, not the user's words - they don't count as
    evidence for what a bullet may say (with_tags=False)."""
    return " ".join(f"{f.text} {' '.join(f.metrics)}" + (f" {' '.join(f.skills)}" if with_tags else "")
                    for f in facts)


def claimable_terms(bank: ExperienceBank, report: MatchReport) -> list[str]:
    """Skills a bullet could name: every JD requirement plus everything in the bank."""
    terms = [m.skill for m in report.must_have + report.nice_to_have] + sorted(bank.all_skills())
    seen, out = set(), []
    for t in terms:
        if normalize(t) not in seen:
            seen.add(normalize(t))
            out.append(t)
    return out


def check_bullet(bullet: Bullet, section: Section, terms: list[str]) -> list[str]:
    problems = []
    by_id = {f.id: f for f in section.facts}
    unknown = [i for i in bullet.fact_ids if i not in by_id]
    if unknown:
        problems.append(f"cites facts not in this section: {unknown}")
    cited = [by_id[i] for i in bullet.fact_ids if i in by_id]
    if not cited:
        return problems + ["cites no valid fact"]

    plain = bullet.text.replace("**", "")
    support = _fact_text(cited, with_tags=False)
    extra_numbers = numbers(plain) - numbers(support)
    if extra_numbers:
        problems.append(f"numbers not in cited facts: {sorted(extra_numbers)}")

    stack = section.source.tech_stack if isinstance(section.source, Project) else []
    for term in terms:
        if appears_in(term, plain) and not appears_in(term, support) \
                and not any(skill_matches(term, s) for s in stack):
            problems.append(f"mentions '{term}' which the cited facts don't support")
    if len(plain) > MAX_BULLET_CHARS:
        problems.append(f"too long ({len(plain)} > {MAX_BULLET_CHARS} chars)")
    return problems


def check_summary(summary: str, fact_ids: list[str], bank: ExperienceBank) -> list[str]:
    facts = [bank.fact_index()[i] for i in fact_ids if i in bank.fact_index()]
    support = _fact_text(facts) + " " + (bank.profile.summary or "")
    problems = []
    extra = numbers(summary) - numbers(support)
    if extra:
        problems.append(f"summary numbers not in cited facts: {sorted(extra)}")
    if len(summary.replace("**", "")) > MAX_SUMMARY_CHARS:
        problems.append("summary too long")
    return problems


# --- deterministic ordering ----------------------------------------------------

def _relevance(source: Role | Project, report: MatchReport) -> int:
    text = _fact_text(source.facts) + " " + " ".join(getattr(source, "tech_stack", []))
    must = sum(2 for m in report.must_have if appears_in(m.skill, text))
    nice = sum(1 for m in report.nice_to_have if appears_in(m.skill, text))
    return must + nice


def order_skills(skills: dict[str, list[str]], report: MatchReport) -> dict[str, list[str]]:
    wanted = [m.skill for m in report.must_have + report.nice_to_have]
    hit = lambda s: any(skill_matches(w, s) for w in wanted)  # noqa: E731
    return {cat: sorted(names, key=lambda s: not hit(s)) for cat, names in skills.items()}


def plan_sections(bank: ExperienceBank, report: MatchReport, bullets_per_item: int = 3) -> list[Section]:
    roles = [Section(f"role:{i}", r, min(bullets_per_item, len(r.facts)))
             for i, r in enumerate(bank.roles) if r.facts]
    projects = sorted(
        (Section(f"project:{i}", p, min(bullets_per_item, len(p.facts))) for i, p in enumerate(bank.projects) if p.facts),
        key=lambda s: -_relevance(s.source, report),   # stable: ties keep your order
    )
    return roles + projects


# --- the agent -----------------------------------------------------------------

def _prompt(jd_text: str, report: MatchReport, sections: list[Section], bank: ExperienceBank) -> str:
    def req(ms):
        return "\n".join(f"- {m.skill}{'' if m.matched else '  (you do NOT have this - do not claim it)'}" for m in ms)

    blocks = []
    for s in sections:
        # Skill tags are deliberately not shown: the model should work from the user's wording.
        facts = "\n".join(
            f"  [{f.id}] {f.text}" + (f" | metrics: {'; '.join(f.metrics)}" if f.metrics else "")
            for f in s.facts
        )
        stack = getattr(s.source, "tech_stack", [])
        blocks.append(f"Section {s.key}: {s.label} - write exactly {s.n_bullets} bullets"
                      + (f"\n  tech stack: {', '.join(stack)}" if stack else "") + f"\n{facts}")

    return (f"JOB DESCRIPTION:\n{jd_text[:MAX_JD_CHARS]}\n\n"
            f"MUST-HAVE REQUIREMENTS:\n{req(report.must_have) or '- none'}\n\n"
            f"NICE-TO-HAVE:\n{req(report.nice_to_have) or '- none'}\n\n"
            f"CURRENT SUMMARY:\n{bank.profile.summary or '(none)'}\n\n"
            "SECTIONS AND THEIR FACTS:\n\n" + "\n\n".join(blocks))


def _validate(out: TailorOutput, sections: list[Section], terms: list[str], bank: ExperienceBank) -> list[str]:
    problems = [f"summary: {p}" for p in check_summary(out.summary, out.summary_fact_ids, bank)]
    drafts = {d.key: d for d in out.sections}
    for s in sections:
        d = drafts.get(s.key)
        if d is None:
            problems.append(f"{s.key}: section missing")
            continue
        if len(d.bullets) != s.n_bullets:
            problems.append(f"{s.key}: wrote {len(d.bullets)} bullets, expected {s.n_bullets}")
        for i, b in enumerate(d.bullets, 1):
            problems += [f"{s.key} bullet {i}: {p}" for p in check_bullet(b, s, terms)]
    return problems


def _assemble(out: TailorOutput, sections: list[Section], terms: list[str], bank: ExperienceBank,
              report: MatchReport) -> TailorResult:
    """Keep bullets that pass; replace the rest with original fact wording."""
    violations = []
    drafts = {d.key: d for d in out.sections}
    assembled = []
    for s in sections:
        kept: list[TailoredBullet] = []
        for b in (drafts[s.key].bullets if s.key in drafts else []):
            problems = check_bullet(b, s, terms)
            if problems:
                violations += [f"{s.label}: rejected \"{b.text[:60]}...\" - {p}" for p in problems]
            elif len(kept) < s.n_bullets:
                kept.append(TailoredBullet(b.text, b.fact_ids))
        used = {i for b in kept for i in b.fact_ids}
        for f in s.facts:                         # fill any gap with untouched original facts
            if len(kept) >= s.n_bullets:
                break
            if f.id not in used:
                kept.append(TailoredBullet(f.text, [f.id], fallback=True))
                used.add(f.id)
        assembled.append((s, kept))

    summary_problems = check_summary(out.summary, out.summary_fact_ids, bank)
    violations += [f"summary: {p}" for p in summary_problems]
    summary_fallback = bool(summary_problems) or not out.summary.strip()
    summary = (bank.profile.summary or "") if summary_fallback else out.summary
    return TailorResult(summary=summary, summary_fallback=summary_fallback, sections=assembled,
                        skills=order_skills(bank.skills, report), violations=violations)


def run(llm: LLM, jd_text: str, report: MatchReport, bank: ExperienceBank) -> TailorResult:
    sections = plan_sections(bank, report)
    terms = claimable_terms(bank, report)
    prompt = _prompt(jd_text, report, sections, bank)

    first = llm.structured(AGENT, SYSTEM, prompt, TailorOutput, temperature=0.4)
    out, model = first.output, first.model
    problems = _validate(out, sections, terms, bank)
    if problems:
        feedback = (f"{prompt}\n\nYOUR PREVIOUS DRAFT:\n{out.model_dump_json()}\n\n"
                    "These problems were found. Fix them and return the complete corrected JSON:\n- "
                    + "\n- ".join(problems))
        second = llm.structured(AGENT, SYSTEM, feedback, TailorOutput, temperature=0.2)
        out, model = second.output, second.model

    result = _assemble(out, sections, terms, bank, report)
    result.model = model
    return result
