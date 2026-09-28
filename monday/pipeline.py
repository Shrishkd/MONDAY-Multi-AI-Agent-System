"""Agent runs as tracked operations - shared by the CLI and the Streamlit app.

Each function runs an agent, stores its output as a pending draft, and returns it.
Nothing here sends anything anywhere; approval is a separate, human step.
"""

import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from monday import resume_import
from monday.agents import interview_prep as prep_agent
from monday.agents import job_matcher, resume_tailor
from monday.agents import mock_interview as mock_agent
from monday.agents import outreach as outreach_agent
from monday.research import Researcher, ResearchError
from monday.agents.job_matcher import MatchReport
from monday.agents.resume_tailor import Bullet, Section, TailoredBullet, TailorResult, check_bullet
from monday.config import Config
from monday.experience import ExperienceBank, load_bank
from monday.llm import LLM
from monday.resume_build import baseline_text, build, keyword_coverage, tailored_text
from monday.tracker import Tracker


def match(tracker: Tracker, config: Config, app_id: int,
          bank: ExperienceBank | None = None) -> tuple[MatchReport, str, int]:
    app = tracker.get_application(app_id)
    bank = bank or load_bank(config.experience_bank)
    report, model = job_matcher.run(LLM(config), app["description"], bank)
    tracker.set_match_score(app_id, report.score)
    draft_id = tracker.add_draft(app_id, job_matcher.AGENT, "match_report", report.model_dump(), model)
    return report, model, draft_id


def latest_report(tracker: Tracker, app_id: int, include_rejected: bool = False) -> MatchReport | None:
    row = tracker.latest_draft(app_id, "match_report", include_rejected)
    return MatchReport.model_validate_json(row["content"]) if row else None


def _resume_content(result: TailorResult, report: MatchReport, bank: ExperienceBank, built) -> dict:
    before = keyword_coverage(report, baseline_text(bank))
    after = keyword_coverage(report, tailored_text(result))
    have = {m.skill for m in report.must_have + report.nice_to_have if m.matched}
    return {
        "summary": result.summary,
        "summary_fallback": result.summary_fallback,
        "sections": [{"key": s.key, "label": s.label, "bullets": [asdict(b) for b in bullets]}
                     for s, bullets in result.sections],
        "skills": result.skills,
        "violations": result.violations,
        "coverage": {
            "before": before.percent, "after": after.percent,
            "hidden": [k for k in after.missing if k in have],     # you have it, resume doesn't show it
            "gaps": [k for k in after.missing if k not in have],   # correctly not claimed
        },
        "tex": str(built.tex), "pdf": str(built.pdf) if built.pdf else None,
        "pages": built.pages, "compile_error": built.compile_error,
    }


def tailor(tracker: Tracker, config: Config, app_id: int) -> tuple[int, dict, str]:
    """Returns (draft id, draft content, model)."""
    app = tracker.get_application(app_id)
    bank = load_bank(config.experience_bank)
    report = latest_report(tracker, app_id) or match(tracker, config, app_id, bank)[0]
    result = resume_tailor.run(LLM(config), app["description"], report, bank)
    built = build(result, bank, config.resume_template,
                  config.output_dir / f"app-{app_id}", app["company"], config.pdflatex)
    content = _resume_content(result, report, bank, built)
    draft_id = tracker.add_draft(app_id, resume_tailor.AGENT, "resume", content, result.model)
    return draft_id, content, result.model


def outreach(tracker: Tracker, config: Config, app_id: int, contact_id: int, kind: str) -> tuple[int, dict, str]:
    """Draft a message to one contact. Returns (draft id, content, model). Never sends."""
    app = tracker.get_application(app_id)
    contact = dict(tracker.get_contact(contact_id))
    bank = load_bank(config.experience_bank)
    report = latest_report(tracker, app_id, include_rejected=True)

    previous = None
    if kind == "follow_up":
        earlier = [d for d in tracker.outreach_drafts(app_id)
                   if d["contact_id"] == contact_id and d["status"] == "approved" and d["kind"] != "follow_up"]
        if earlier:
            c = json.loads(earlier[-1]["edited_content"] or earlier[-1]["content"])
            previous = f"Subject: {c.get('subject', '')}\n{c['body']}"

    out, check, model = outreach_agent.run(
        LLM(config), kind, bank=bank, report=report, company=app["company"], role=app["title"],
        jd_text=app["description"], contact=contact, previous=previous,
        linkedin_note_chars=config.linkedin_note_chars,
    )
    content = {
        "kind": kind, "subject": out.subject, "body": out.body, "fact_ids": out.fact_ids,
        "contact_details_used": out.contact_details_used,
        "problems": check.problems, "warnings": check.warnings,
        "max_chars": outreach_agent.kinds(config.linkedin_note_chars)[kind].max_chars,
        "follow_up_without_earlier_message": kind == "follow_up" and previous is None,
    }
    draft_id = tracker.add_draft(app_id, outreach_agent.AGENT, kind, content, model, contact_id=contact_id)
    return draft_id, content, model


def outreach_edit_check(tracker: Tracker, config: Config, draft_id: int, subject: str, body: str) -> list[str]:
    """Re-run the outreach checks on a human edit. Returns problems + warnings as plain strings."""
    draft = tracker.get_draft(draft_id)
    content = json.loads(draft["content"])
    app = tracker.get_application(draft["application_id"])
    contact = tracker.get_contact(draft["contact_id"])
    out = outreach_agent.OutreachOutput(subject=subject, body=body, fact_ids=content["fact_ids"],
                                        contact_details_used=content["contact_details_used"])
    check = outreach_agent.check_message(
        out, outreach_agent.kinds(config.linkedin_note_chars)[content["kind"]],
        bank=load_bank(config.experience_bank),
        report=latest_report(tracker, draft["application_id"], include_rejected=True),
        contact_name=contact["name"], contact_notes=contact["notes"] or "",
        context=f"{app['description']} {app['company']} {app['title']}",
    )
    return check.problems + check.warnings


def interview_prep(tracker: Tracker, config: Config, app_id: int,
                   refresh_research: bool = False) -> tuple[int, dict, str]:
    """Research the company (cached per company), then write the prep sheet. Returns (draft id, content, model)."""
    app = tracker.get_application(app_id)
    bank = load_bank(config.experience_bank)
    report = latest_report(tracker, app_id, include_rejected=True)

    notes: list[str] = []
    cached = None if refresh_research else tracker.latest_research(app["company_id"])
    if cached:
        sources, researched_at = cached
    else:
        try:
            found, notes = Researcher(config.ollama_api_key).gather(app["company"], app["title"],
                                                                    app["company_website"])
            sources = [s.as_dict() for s in found]
            tracker.save_research(app["company_id"], sources)
            researched_at = "just now"
        except ResearchError as exc:
            sources, researched_at, notes = [], None, [str(exc)]

    llm = LLM(config)
    brief, dropped_brief, _ = prep_agent.write_brief(llm, app["company"], sources)
    prep, dropped_prep, model = prep_agent.write_prep(
        llm, bank=bank, report=report, company=app["company"], role=app["title"],
        jd_text=app["description"], brief=brief,
    )
    content = {
        "brief": brief.model_dump(),
        "sources": [{"id": i, "title": s["title"], "url": s["url"]} for i, s in enumerate(sources, 1)],
        "researched_at": researched_at,
        "research_notes": notes,
        "questions": [q.model_dump() for q in prep.questions],
        "star_outlines": [s.model_dump() for s in prep.star_outlines],
        "questions_to_ask": prep.questions_to_ask,
        "dropped": dropped_brief + dropped_prep,
    }
    draft_id = tracker.add_draft(app_id, prep_agent.AGENT, "interview_prep", content, model)
    return draft_id, content, model


def latest_prep(tracker: Tracker, app_id: int) -> dict | None:
    """The newest non-rejected prep sheet (edited version if you edited it)."""
    row = tracker.latest_draft(app_id, "interview_prep")
    return json.loads(row["edited_content"] or row["content"]) if row else None


def start_mock(tracker: Tracker, app_id: int, n: int, categories: list[str],
               seed: int | None = None) -> tuple[int, list[dict]]:
    prep = latest_prep(tracker, app_id)
    if not prep:
        raise ValueError("No interview prep sheet for this application yet - run Interview Prep first.")
    questions = mock_agent.pick_questions(prep, n, categories, seed)
    if not questions:
        raise ValueError("The prep sheet has no questions in the chosen categories.")
    return tracker.start_practice(app_id, questions), questions


def answer_mock(tracker: Tracker, config: Config, session_id: int, app_id: int, question: dict,
                answer: str, is_follow_up: bool = False) -> tuple[dict, float | None]:
    if not answer.strip():
        raise ValueError("Type an answer first.")
    app = tracker.get_application(app_id)
    feedback, score, model = mock_agent.evaluate(
        LLM(config), question=question, answer=answer, bank=load_bank(config.experience_bank),
        report=latest_report(tracker, app_id, include_rejected=True),
        company=app["company"], role=app["title"], jd_text=app["description"],
    )
    feedback["model"] = model
    tracker.add_practice_answer(session_id, question, answer, feedback, score, is_follow_up)
    return feedback, score


def save_story(config: Config, story: dict) -> Path:
    """Add a STAR story you completed to the Experience Bank (backed up first). Returns the backup path."""
    bank = load_bank(config.experience_bank)
    base = re.sub(r"[^a-z0-9]+", "-", story["title"].lower()).strip("-")[:40] or "story"
    existing = {s.id for s in bank.stories}
    story_id, n = f"story-{base}", 2
    while story_id in existing:
        story_id, n = f"story-{base}-{n}", n + 1
    updated = ExperienceBank.model_validate({
        **bank.model_dump(), "stories": [*[s.model_dump() for s in bank.stories], {**story, "id": story_id}],
    })
    backup = config.experience_bank.with_suffix(f".{datetime.now():%Y%m%d-%H%M%S}.bak.yaml")
    shutil.copy2(config.experience_bank, backup)
    config.experience_bank.write_text(
        "# MONDAY Experience Bank - only TRUE facts.\n" + resume_import.bank_yaml(updated), encoding="utf-8")
    return backup


def prep_markdown(app, content: dict) -> str:
    """The prep sheet as Markdown, to study from or print."""
    src = {s["id"]: s for s in content["sources"]}
    lines = [f"# Interview prep: {app['title']} @ {app['company']}", ""]
    titles = {"overview": "Overview", "products": "Products", "tech_and_engineering": "Tech & engineering",
              "recent_news": "Recent news", "interview_process": "Interview process",
              "culture_and_values": "Culture & values"}
    lines.append("## Company brief")
    for key, title in titles.items():
        points = content["brief"].get(key) or []
        if points:
            lines.append(f"\n### {title}")
            for p in points:
                refs = " ".join(f"[{i}]({src[i]['url']})" for i in p["source_ids"] if i in src)
                lines.append(f"- {p['text']} {refs}")
    groups = {"technical": "Technical", "resume": "About your resume", "behavioral": "Behavioral",
              "gap": "Gaps - answer honestly", "company": "Why this company"}
    lines.append("\n## Likely questions")
    for cat, title in groups.items():
        qs = [q for q in content["questions"] if q["category"] == cat]
        if qs:
            lines.append(f"\n### {title}")
            for q in qs:
                lines.append(f"\n**{q['question']}**  \n_{q['why_they_ask']}_")
                lines += [f"- {t}" for t in q["talking_points"]]
    if content["star_outlines"]:
        lines.append("\n## STAR outlines (fill in the blanks)")
        for s in content["star_outlines"]:
            lines.append(f"\n### {s['title']}")
            lines += [f"- **Situation:** {s['situation']}", f"- **Task:** {s['task']}",
                      f"- **Action:** {s['action']}", f"- **Result:** {s['result']}"]
            lines += [f"- [ ] {f}" for f in s["fill_in"]]
    if content["questions_to_ask"]:
        lines.append("\n## Questions to ask them")
        lines += [f"- {q}" for q in content["questions_to_ask"]]
    if content["sources"]:
        lines.append("\n## Sources")
        lines += [f"{s['id']}. [{s['title']}]({s['url']})" for s in content["sources"]]
    return "\n".join(lines) + "\n"


def _section(key: str, bank: ExperienceBank) -> Section:
    kind, index = key.split(":")
    source = (bank.roles if kind == "role" else bank.projects)[int(index)]
    return Section(key, source, len(source.facts))


def edit_warnings(content: dict, bank: ExperienceBank, report: MatchReport) -> list[str]:
    """Run the fabrication checks on human edits. Warnings only - you are the authority."""
    terms = resume_tailor.claimable_terms(bank, report)
    warnings = []
    for sec in content["sections"]:
        section = _section(sec["key"], bank)
        for b in sec["bullets"]:
            for problem in check_bullet(Bullet(fact_ids=b["fact_ids"], text=b["text"]), section, terms):
                warnings.append(f"{section.label}: \"{b['text'][:50]}...\" - {problem}")
    warnings += [f"summary: {p}" for p in resume_tailor.check_summary(content["summary"], [], bank)]
    return warnings


def rebuild_resume(config: Config, app_id: int, company: str, content: dict,
                   report: MatchReport) -> dict:
    """Re-render and recompile a resume after human edits. Returns updated content."""
    bank = load_bank(config.experience_bank)
    result = TailorResult(
        summary=content["summary"], summary_fallback=False,
        sections=[(_section(s["key"], bank), [TailoredBullet(**b) for b in s["bullets"]])
                  for s in content["sections"]],
        skills=content["skills"], violations=content.get("violations", []),
    )
    built = build(result, bank, config.resume_template,
                  config.output_dir / f"app-{app_id}" / "edited", company, config.pdflatex)
    return _resume_content(result, report, bank, built)
