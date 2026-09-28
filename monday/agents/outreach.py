"""Outreach Agent: draft a cold email or LinkedIn message to a real person.

Draft-only by design - a bad automated message burns a real relationship, so MONDAY has
no way to send anything. The model writes; code checks both sides of the message:
claims about you must cite Experience Bank facts, and personal details about the
contact must come from the notes you wrote about them.
"""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from monday.agents.job_matcher import MatchReport
from monday.agents.resume_tailor import numbers
from monday.experience import ExperienceBank
from monday.llm import LLM
from monday.skills import appears_in, normalize

AGENT = "outreach"


@dataclass(frozen=True)
class Kind:
    label: str
    max_chars: int
    subject: bool
    guidance: str


def kinds(linkedin_note_chars: int = 200) -> dict[str, Kind]:
    return {
        "cold_email": Kind("Cold email", 1100, True,
                           "90-150 words. Subject line under 60 characters, specific, no clickbait."),
        "linkedin_note": Kind("LinkedIn connection note", linkedin_note_chars, False,
                              f"At most {linkedin_note_chars} characters INCLUDING the greeting. "
                              "One reason for connecting, one proof point at most, no ask for a job."),
        "linkedin_message": Kind("LinkedIn message", 700, False,
                                 "60-110 words, conversational, one clear ask."),
        "follow_up": Kind("Follow-up email", 600, True,
                          "40-80 words. Reference the earlier message briefly, add one new useful "
                          "detail, repeat the ask lightly. Never guilt-trip."),
    }


CLICHES = ["i hope this email finds you well", "i hope this message finds you well", "dear sir",
           "dear madam", "to whom it may concern", "i am writing to", "kindly", "do the needful",
           "passionate about", "i would be a great fit", "esteemed", "revert back", "rockstar", "ninja"]

_PLACEHOLDER = re.compile(r"\[[^\]]{1,40}\]|\{[^}]{1,40}\}|<[^>]{1,40}>")
# "a 15-minute chat" is an ask, not a claim - don't number-check it.
_TIME_ASK = re.compile(r"\b\d+\s*[-‑– ]?\s*(?:min|mins|minute|minutes|hour|hours|hr|hrs)\b", re.I)

SYSTEM = """You write short outreach messages from a job candidate to one specific person.

Hard rules - a draft that breaks one is sent back:
- Write in first person as the candidate. Plain text only: no markdown, no bullet lists.
- Claims about the candidate: ONLY from the listed facts. Cite the fact ids you used. Keep numbers exactly.
- Never claim any skill listed under "candidate does NOT have".
- Details about the contact: ONLY from CONTACT NOTES. In the message, say them naturally in your own
  words, addressed to the contact ("you lead...", not "you Leads..."). Separately, copy each detail you
  used word for word from the notes into contact_details_used. If the notes are empty, don't pretend to
  know them personally - refer to their role and the company instead.
- Don't inflate scope: projects are things the candidate built or designed, not teams they led, unless
  a fact says so.
- Never put fact ids or brackets in the message text; ids go only in fact_ids.
- No placeholders like [Name] or {company}. Sign off with the candidate's real name (LinkedIn
  connection notes may skip the sign-off).
- Emails and LinkedIn messages: short paragraphs separated by blank lines; greeting and sign-off on
  their own lines.
- Respect the length limit exactly.

Make it worth their time: why them (specific), who you are (one line), one or two proof points that
match what this role needs, and one small, easy ask (a 15-minute chat, a pointer to the right person,
or whether they'd be open to referring you). No flattery, no clichés, no "I hope this finds you well"."""


class OutreachOutput(BaseModel):
    subject: str = Field(default="", description="Email subject; empty for LinkedIn.")
    body: str
    fact_ids: list[str] = Field(description="Experience Bank fact ids the message relies on.")
    contact_details_used: list[str] = Field(default=[], description="Verbatim phrases from CONTACT NOTES.")


@dataclass
class OutreachCheck:
    problems: list[str] = field(default_factory=list)   # hard failures (sent back to the model once)
    warnings: list[str] = field(default_factory=list)   # things a human should look at


def _in_notes(detail: str, notes: str) -> bool:
    d, n = normalize(detail), normalize(notes)
    if not d:
        return True
    if d in n:
        return True
    words = [w for w in d.split() if len(w) > 3]
    return bool(words) and sum(w in n.split() for w in words) / len(words) >= 0.7


def check_message(out: OutreachOutput, kind: Kind, *, bank: ExperienceBank, report: MatchReport | None,
                  contact_name: str, contact_notes: str, context: str) -> OutreachCheck:
    """`context` = everything the message may legitimately quote numbers from besides facts
    (JD, company, role title, previous message)."""
    check = OutreachCheck()
    facts = bank.fact_index()
    body = out.body.replace("**", "")
    text = f"{out.subject}\n{body}"

    unknown = [i for i in out.fact_ids if i not in facts]
    if unknown:
        check.problems.append(f"cites unknown fact ids: {unknown}")
    if len(body) > kind.max_chars:
        check.problems.append(f"too long: {len(body)} characters (limit {kind.max_chars})")
    if kind.subject and not out.subject.strip():
        check.problems.append("missing subject line")
    placeholders = _PLACEHOLDER.findall(text)
    if placeholders:
        check.problems.append(f"contains placeholders: {placeholders}")
    leaked = [i for i in facts if re.search(rf"(?<![\w-]){re.escape(i)}(?![\w-])", text)]
    if leaked:
        check.problems.append(f"fact ids leaked into the message: {leaked}")

    cited = [facts[i] for i in out.fact_ids if i in facts]
    support = " ".join(f"{f.text} {' '.join(f.metrics)}" for f in cited)
    profile = bank.profile
    background = " ".join(
        [f"{e.degree} {e.start} {e.end} {' '.join(e.details)}" for e in bank.education]
        + [f"{c.name} {c.year}" for c in bank.certifications] + bank.achievements
        + [f"{r.start} {r.end}" for r in bank.roles]
    )
    allowed = (f"{support} {background} {contact_notes} {context} {profile.phone or ''} "
               f"{' '.join(profile.links)}")
    extra = numbers(_TIME_ASK.sub("", text)) - numbers(allowed)
    if extra:
        check.problems.append(f"numbers that aren't in your cited facts or the notes: {sorted(extra)}")

    if report:
        for gap in report.gaps + [m.skill for m in report.nice_to_have if not m.matched]:
            if appears_in(gap, body):
                check.warnings.append(f"mentions '{gap}', which you don't have - make sure it doesn't claim it")
    for detail in out.contact_details_used:
        if not _in_notes(detail, contact_notes):
            check.warnings.append(f"detail about {contact_name} not found in your notes: \"{detail}\"")
    def says(name: str) -> bool:
        return re.search(rf"\b{re.escape(name)}\b", body, re.I) is not None

    first_name = contact_name.split()[0] if contact_name.split() else ""
    if first_name and not says(first_name):
        check.warnings.append(f"doesn't address {first_name} by name")
    if kind.subject or kind.max_chars > 300:   # connection notes are too short for a sign-off
        if profile.name.split() and not says(profile.name.split()[0]):
            check.warnings.append("isn't signed with your name")
    for phrase in CLICHES:
        if phrase in body.lower():
            check.warnings.append(f"cliché: \"{phrase}\"")
    return check


def _prompt(kind: Kind, *, bank: ExperienceBank, report: MatchReport | None, company: str, role: str,
            jd_text: str, contact: dict, previous: str | None) -> str:
    facts = "\n".join(f"  [{f.id}] {f.text.replace('**', '')}" for f in bank.facts())
    needs = ", ".join(m.skill for m in report.must_have) if report else "(not analysed)"
    lacks = ", ".join(report.gaps + [m.skill for m in report.nice_to_have if not m.matched]) if report else ""
    p = bank.profile
    return (
        f"MESSAGE TYPE: {kind.label}. {kind.guidance} Hard limit: {kind.max_chars} characters of body.\n\n"
        f"CANDIDATE: {p.name} - {p.headline or ''}\nLinks: {', '.join(p.links)}\n"
        f"Candidate does NOT have: {lacks or 'nothing notable'}\n\n"
        f"TARGET: {role} at {company}\nThe role mainly needs: {needs}\n"
        f"Job description (excerpt):\n{jd_text[:2500]}\n\n"
        f"CONTACT: {contact['name']}, {contact.get('role') or 'role unknown'} at {company}\n"
        f"CONTACT NOTES (the only source of personal details):\n{contact.get('notes') or '(none)'}\n\n"
        + (f"EARLIER MESSAGE TO THEM:\n{previous}\n\n" if previous else "")
        + f"CANDIDATE FACTS (cite by id):\n{facts}"
    )


def run(llm: LLM, kind_key: str, *, bank: ExperienceBank, report: MatchReport | None, company: str,
        role: str, jd_text: str, contact: dict, previous: str | None = None,
        linkedin_note_chars: int = 200) -> tuple[OutreachOutput, OutreachCheck, str]:
    kind = kinds(linkedin_note_chars)[kind_key]
    prompt = _prompt(kind, bank=bank, report=report, company=company, role=role, jd_text=jd_text,
                     contact=contact, previous=previous)
    context = f"{jd_text} {company} {role} {previous or ''}"

    def checked(out: OutreachOutput) -> OutreachCheck:
        return check_message(out, kind, bank=bank, report=report, contact_name=contact["name"],
                             contact_notes=contact.get("notes") or "", context=context)

    result = llm.structured(AGENT, SYSTEM, prompt, OutreachOutput, temperature=0.6)
    out, model = result.output, result.model
    check = checked(out)
    if check.problems:
        retry = llm.structured(
            AGENT, SYSTEM,
            f"{prompt}\n\nYOUR PREVIOUS DRAFT:\n{out.model_dump_json()}\n\nFix these problems and return "
            "the complete corrected JSON:\n- " + "\n- ".join(check.problems),
            OutreachOutput, temperature=0.3,
        )
        out, model = retry.output, retry.model
        check = checked(out)
    out.body = out.body.replace("**", "")
    return out, check, model
