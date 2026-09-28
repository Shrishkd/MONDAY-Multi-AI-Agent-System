"""Agent runs as tracked operations - shared by the CLI and the Streamlit app.

Each function runs an agent, stores its output as a pending draft, and returns it.
Nothing here sends anything anywhere; approval is a separate, human step.
"""

import json
from dataclasses import asdict

from monday.agents import job_matcher, resume_tailor
from monday.agents import outreach as outreach_agent
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
