"""Mock interview: you answer a question, you get feedback and a follow-up question.

Two layers of feedback:
- Code checks your answer against your Experience Bank: numbers you quote that aren't
  in any fact (misremembered metrics are a classic interview slip), claims of skills you
  don't have, and speaking length.
- The model scores the answer and suggests a stronger version - built only from what
  you said plus your facts, and dropped if it adds numbers or claims anything new.
"""

import random
import re
from typing import Literal

from pydantic import BaseModel, Field

from monday.agents.job_matcher import MatchReport
from monday.agents.resume_tailor import numbers
from monday.experience import ExperienceBank
from monday.llm import LLM
from monday.skills import appears_in

AGENT = "mock_interview"
WORDS_PER_MINUTE = 130
MIN_WORDS, MAX_WORDS = 40, 350
CRITERIA = ("relevance", "structure", "specificity", "clarity")

SYSTEM = """You are an experienced interviewer giving a candidate honest, specific feedback on one answer.

Score each criterion from 1 (poor) to 5 (excellent), with one sentence of reasoning:
- relevance: does it answer the question that was asked?
- structure: easy to follow? For behavioral questions: Situation, Task, Action, Result.
- specificity: concrete details, their own role, real numbers - not generic claims.
- clarity: concise, confident, no rambling or filler.

Then:
- strengths and improvements: short, concrete, about THIS answer. Be direct, not flattering.
- stronger_answer: a better version of the candidate's answer, written in first person, 120-220 words.
  Use ONLY what the candidate said plus the CANDIDATE FACTS; cite the fact ids you used. Never add a
  number, tool, result or experience that isn't in the answer or the facts. Never claim anything
  listed under "candidate does NOT have".
- follow_up_question: the probing question a real interviewer would ask next, based on what the
  candidate actually said.
- If AUTOMATED CHECKS are listed, address them in improvements (e.g. a number that doesn't match the
  candidate's records: name the correct figure from the facts).
- Never put fact ids in the stronger_answer text; ids go only in fact_ids."""


class Criterion(BaseModel):
    name: Literal["relevance", "structure", "specificity", "clarity"]
    score: int = Field(ge=1, le=5)
    comment: str


class Feedback(BaseModel):
    scores: list[Criterion]
    strengths: list[str]
    improvements: list[str]
    stronger_answer: str
    fact_ids: list[str] = []
    follow_up_question: str


_CLAIM = r"(?:have|'ve|has|had)\s+(?:used|worked (?:with|on)|built|deployed|experience (?:with|in)|shipped)"


def _gaps(report: MatchReport | None) -> list[str]:
    return (report.gaps + [m.skill for m in report.nice_to_have if not m.matched]) if report else []


def check_answer(answer: str, bank: ExperienceBank, report: MatchReport | None, context: str = "") -> list[str]:
    """Code-only checks on what YOU said. `context` = the question + JD (numbers you may echo)."""
    notes = []
    words = len(answer.split())
    minutes = words / WORDS_PER_MINUTE
    if words < MIN_WORDS:
        notes.append(f"Short: {words} words (~{minutes * 60:.0f}s spoken). Most answers need 1-2 minutes.")
    elif words > MAX_WORDS:
        notes.append(f"Long: {words} words (~{minutes:.1f} min spoken). Trim to the strongest points.")

    known = " ".join(f"{f.text} {' '.join(f.metrics)}" for f in bank.facts())
    known += " " + " ".join(f"{s.situation} {s.task} {s.action} {s.result}" for s in bank.stories)
    known += " " + (bank.profile.summary or "") + " " + context
    unknown = sorted(numbers(answer) - numbers(known))
    if unknown:
        notes.append(f"Numbers not in your Experience Bank: {', '.join(unknown)} - double-check them; "
                     "a misremembered metric is easy for an interviewer to probe.")
    for gap in _gaps(report):
        if appears_in(gap, answer) and re.search(rf"{_CLAIM}[^.]*{re.escape(gap)}", answer, re.I):
            notes.append(f"You claimed experience with {gap}, which isn't in your Experience Bank.")
    return notes


def check_stronger(fb: Feedback, answer: str, bank: ExperienceBank, report: MatchReport | None) -> list[str]:
    """The suggested rewrite may only use your answer and cited facts."""
    facts = bank.fact_index()
    problems = []
    unknown_ids = [i for i in fb.fact_ids if i not in facts]
    if unknown_ids:
        problems.append(f"cites unknown facts {unknown_ids}")
    support = answer + " " + " ".join(f"{facts[i].text} {' '.join(facts[i].metrics)}" for i in fb.fact_ids if i in facts)
    extra = numbers(fb.stronger_answer) - numbers(support)
    if extra:
        problems.append(f"adds numbers not in your answer or cited facts: {sorted(extra)}")
    for gap in _gaps(report):
        if appears_in(gap, fb.stronger_answer) and not appears_in(gap, answer):
            problems.append(f"brings in {gap}, which you don't have")
    plain = re.sub(r"[‐‑‒–]", "-", fb.stronger_answer)
    leaked = [i for i in facts if re.search(rf"(?<![\w-]){re.escape(i)}(?![\w-])", plain)]
    if leaked:
        problems.append(f"mentions fact ids in the text: {leaked}")
    return problems


def pick_questions(prep: dict, n: int, categories: list[str], seed: int | None = None) -> list[dict]:
    """A mixed set from the prep sheet: round-robin over categories so one type doesn't dominate."""
    rng = random.Random(seed)
    pools = {c: [q for q in prep["questions"] if q["category"] == c] for c in categories}
    for pool in pools.values():
        rng.shuffle(pool)
    picked = []
    while len(picked) < n and any(pools.values()):
        for c in categories:
            if pools[c] and len(picked) < n:
                picked.append(pools[c].pop())
    return picked


def evaluate(llm: LLM, *, question: dict, answer: str, bank: ExperienceBank, report: MatchReport | None,
             company: str, role: str, jd_text: str) -> tuple[dict, float | None, str]:
    """Returns (feedback content, mean score, model)."""
    code_notes = check_answer(answer, bank, report, context=f"{question['question']} {jd_text}")
    facts = "\n".join(f"  [{f.id}] {f.text.replace('**', '')}" for f in bank.facts())
    lacks = ", ".join(_gaps(report)) or "nothing notable"
    prompt = (f"ROLE: {role} at {company}\nCandidate does NOT have: {lacks}\n\n"
              f"QUESTION ({question.get('category', 'general')}): {question['question']}\n"
              + (f"Why interviewers ask it: {question['why_they_ask']}\n" if question.get("why_they_ask") else "")
              + f"\nCANDIDATE'S ANSWER:\n{answer}\n\n"
              + ("AUTOMATED CHECKS:\n- " + "\n- ".join(code_notes) + "\n\n" if code_notes else "")
              + f"CANDIDATE FACTS (cite by id):\n{facts}")
    result = llm.structured(AGENT, SYSTEM, prompt, Feedback, temperature=0.3)
    fb, model = result.output, result.model

    stronger_problems = check_stronger(fb, answer, bank, report)
    if stronger_problems:
        retry = llm.structured(
            AGENT, SYSTEM,
            f"{prompt}\n\nYOUR PREVIOUS FEEDBACK:\n{fb.model_dump_json()}\n\nThe stronger_answer has these "
            "problems. Fix them and return the complete corrected JSON:\n- " + "\n- ".join(stronger_problems),
            Feedback, temperature=0.2,
        )
        fb, model = retry.output, retry.model
        stronger_problems = check_stronger(fb, answer, bank, report)
    scores = [c for c in fb.scores if c.name in CRITERIA]
    mean = sum(c.score for c in scores) / len(scores) if scores else None
    content = {
        "scores": [c.model_dump() for c in scores],
        "strengths": fb.strengths,
        "improvements": fb.improvements,
        # A rewrite that fails the fabrication check is withheld, not shown.
        "stronger_answer": None if stronger_problems else fb.stronger_answer,
        "stronger_answer_withheld": stronger_problems,
        "fact_ids": fb.fact_ids,
        "follow_up_question": fb.follow_up_question,
        "checks": code_notes,
        "words": len(answer.split()),
    }
    return content, mean, model
