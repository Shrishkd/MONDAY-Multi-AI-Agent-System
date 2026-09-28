import argparse
import shutil
import sys
from pathlib import Path

import ollama

from monday import pipeline
from monday.config import PROJECT_ROOT, load_config
from monday.db import APPLICATION_STATUSES, connect
from monday.experience import load_bank
from monday.latex import LatexError, find_pdflatex
from monday.llm import LLMError
from monday.tracker import Tracker

EXAMPLE_BANK = PROJECT_ROOT / "examples" / "experience_bank.example.yaml"


def cmd_init(args, config):
    config.database.parent.mkdir(parents=True, exist_ok=True)
    connect(config.database).close()
    print(f"Database ready: {config.database}")
    if config.experience_bank.exists():
        print(f"Experience Bank already exists: {config.experience_bank}")
    else:
        shutil.copy(EXAMPLE_BANK, config.experience_bank)
        print(f"Created {config.experience_bank} from the example - replace it with your real experience.")


def cmd_doctor(args, config):
    client = ollama.Client(host=config.ollama_host)
    try:
        installed = {m.model for m in client.list().models}
    except Exception as exc:  # any failure here means the daemon is unusable
        print(f"[x] Ollama not reachable at {config.ollama_host}: {exc}")
        return 1
    print(f"[ok] Ollama reachable at {config.ollama_host}")
    ok = True
    for agent, models in config.agents.items():
        marks = []
        for model in models:
            found = model in installed or f"{model}:latest" in installed
            ok &= found or model != models[0]
            marks.append(f"{model} {'ok' if found else 'MISSING'}")
        print(f"  {agent:15} " + ", ".join(marks))
    try:
        bank = load_bank(config.experience_bank)
        print(f"[ok] Experience Bank: {len(bank.facts())} facts, {len(bank.stories)} stories")
    except Exception as exc:
        ok = False
        print(f"[x] Experience Bank: {exc}")
    try:
        print(f"[ok] pdflatex: {find_pdflatex(config.pdflatex)}")
    except LatexError as exc:
        ok = False
        print(f"[x] {exc}")
    if config.resume_template.exists():
        print(f"[ok] Resume template: {config.resume_template}")
    else:
        ok = False
        print(f"[x] Resume template missing: {config.resume_template}")
    return 0 if ok else 1


def cmd_add_job(args, tracker):
    description = Path(args.jd_file).read_text(encoding="utf-8") if args.jd_file else sys.stdin.read()
    app_id = tracker.add_job(args.company, args.title, description,
                             url=args.url, location=args.location, source=args.source)
    print(f"Added application #{app_id}: {args.title} @ {args.company}")


def cmd_list(args, tracker):
    rows = tracker.list_applications(args.status)
    if not rows:
        print("No applications yet.")
        return
    print(f"{'id':>4}  {'status':12} {'score':>5}  {'company':20} title")
    for r in rows:
        score = f"{r['match_score']:.0f}" if r["match_score"] is not None else "-"
        print(f"{r['id']:>4}  {r['status']:12} {score:>5}  {r['company'][:20]:20} {r['title']}")


def cmd_show(args, tracker):
    app = tracker.get_application(args.id)
    print(f"#{app['id']} {app['title']} @ {app['company']}  [{app['status']}]")
    if app["url"]:
        print(app["url"])
    if app["next_action"]:
        print(f"Next: {app['next_action']}" + (f" (due {app['next_action_due']})" if app["next_action_due"] else ""))
    print("\nTimeline:")
    for e in tracker.timeline(args.id):
        print(f"  {e['created_at']}  {e['kind']:12} {e['detail'] or ''}")


def cmd_status(args, tracker):
    tracker.set_status(args.id, args.status, args.note)
    print(f"#{args.id} -> {args.status}")


def cmd_match(args, tracker, config):
    app = tracker.get_application(args.id)
    print(f"Matching #{app['id']} {app['title']} @ {app['company']} ...")
    report, model, _ = pipeline.match(tracker, config, app["id"])

    print(f"\nScore {report.score:.0f}/100 -> {report.recommendation.upper()}   (extracted by {model})")
    for label, matches in (("Must have", report.must_have), ("Nice to have", report.nice_to_have)):
        if matches:
            print(f"\n{label}:")
            for m in matches:
                proof = ", ".join(m.evidence[:3]) if m.matched else "GAP"
                print(f"  [{'x' if m.matched else ' '}] {m.skill:28} {proof}")
    for flag in report.flags:
        print(f"! {flag}")
    if report.dropped_skills:
        print(f"  ignored: {', '.join(report.dropped_skills)}")
    print(f"\nDecide: monday status {app['id']} preparing   |   monday status {app['id']} skipped")


def cmd_tailor(args, tracker, config):
    app = tracker.get_application(args.id)
    print(f"Tailoring resume for #{app['id']} {app['title']} @ {app['company']} (takes a minute) ...")
    draft_id, c, model = pipeline.tailor(tracker, config, app["id"])

    print(f"\nDraft #{draft_id} written by {model}")
    print(f"\nSummary:{'  (original kept - rewrite failed checks)' if c['summary_fallback'] else ''}\n  {c['summary']}")
    for section in c["sections"]:
        print(f"\n{section['label']}")
        for b in section["bullets"]:
            tag = "  [original]" if b["fallback"] else ""
            print(f"  - {b['text']}{tag}\n      cites: {', '.join(b['fact_ids'])}")
    cov = c["coverage"]
    print(f"\nATS keyword coverage: {cov['before']:.0f}% -> {cov['after']:.0f}%")
    if cov["hidden"]:
        print(f"  you have but the resume doesn't show: {', '.join(cov['hidden'])}")
    if cov["gaps"]:
        print(f"  real gaps, correctly not claimed: {', '.join(cov['gaps'])}")
    if c["violations"]:
        print(f"\nRejected by the fabrication check ({len(c['violations'])}):")
        for v in c["violations"]:
            print(f"  x {v}")
    if c["pdf"]:
        warn = "  <- more than one page!" if c["pages"] and c["pages"] > 1 else ""
        print(f"\nPDF: {c['pdf']} ({c['pages']} page{'s' if c['pages'] != 1 else ''}){warn}")
    else:
        print(f"\nPDF not built: {c['compile_error']}\n.tex written to {c['tex']}")
    print(f"\nReview and approve it in the app:  streamlit run app.py")


def cmd_add_contact(args, tracker):
    app = tracker.get_application(args.app_id)
    contact_id = tracker.add_contact(app["company"], args.name, role=args.role, email=args.email,
                                     linkedin_url=args.linkedin, notes=args.notes)
    print(f"Added contact #{contact_id}: {args.name} @ {app['company']}")


def cmd_outreach(args, tracker, config):
    app = tracker.get_application(args.id)
    contact = tracker.get_contact(args.contact)
    print(f"Drafting {args.kind} to {contact['name']} for #{app['id']} {app['title']} @ {app['company']} ...")
    draft_id, c, model = pipeline.outreach(tracker, config, app["id"], contact["id"], args.kind)
    print(f"\nDraft #{draft_id} by {model} ({len(c['body'])}/{c['max_chars']} characters)\n")
    if c["subject"]:
        print(f"Subject: {c['subject']}\n")
    print(c["body"])
    print(f"\ncites: {', '.join(c['fact_ids'])}")
    for p in c["problems"]:
        print(f"x FAILED CHECK: {p}")
    for w in c["warnings"]:
        print(f"! {w}")
    print("\nMONDAY never sends messages. Approve it in the app, then send it yourself.")


def cmd_funnel(args, tracker):
    print("Reached stage:")
    for stage, n in tracker.funnel().items():
        print(f"  {stage:12} {n}")
    print("\nDraft review:")
    for k, v in tracker.draft_stats().items():
        print(f"  {k:16} {v}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="monday", description="MONDAY - draft-only job search autopilot")
    p.add_argument("--config", help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the database and an Experience Bank to fill in")
    sub.add_parser("doctor", help="check Ollama, configured models and the Experience Bank")

    a = sub.add_parser("add-job", help="save a JD (from --jd-file or stdin) and open an application")
    a.add_argument("--company", required=True)
    a.add_argument("--title", required=True)
    a.add_argument("--jd-file")
    a.add_argument("--url")
    a.add_argument("--location")
    a.add_argument("--source", help="where you found it, e.g. linkedin, referral")

    ls = sub.add_parser("list", help="list applications")
    ls.add_argument("--status", choices=APPLICATION_STATUSES)

    sh = sub.add_parser("show", help="show one application and its timeline")
    sh.add_argument("id", type=int)

    st = sub.add_parser("status", help="move an application to a new stage")
    st.add_argument("id", type=int)
    st.add_argument("status", choices=APPLICATION_STATUSES)
    st.add_argument("--note")

    m = sub.add_parser("match", help="score a saved JD against your Experience Bank")
    m.add_argument("id", type=int)

    t = sub.add_parser("tailor", help="draft a tailored resume (.tex + .pdf) for an application")
    t.add_argument("id", type=int)

    ac = sub.add_parser("add-contact", help="add a person at an application's company")
    ac.add_argument("app_id", type=int)
    ac.add_argument("--name", required=True)
    ac.add_argument("--role")
    ac.add_argument("--email")
    ac.add_argument("--linkedin", help="profile URL")
    ac.add_argument("--notes", help="what you know about them - the only source of personal details")

    o = sub.add_parser("outreach", help="draft a message to a contact (never sent by MONDAY)")
    o.add_argument("id", type=int, help="application id")
    o.add_argument("--contact", type=int, required=True)
    o.add_argument("--kind", default="cold_email",
                   choices=["cold_email", "linkedin_note", "linkedin_message", "follow_up"])

    sub.add_parser("funnel", help="how many applications reached each stage")
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which can't print what models write (e.g. U+202F).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "init":
        return cmd_init(args, config) or 0
    if args.command == "doctor":
        return cmd_doctor(args, config)

    conn = connect(config.database)
    try:
        if args.command == "match":
            cmd_match(args, Tracker(conn), config)
        elif args.command == "tailor":
            cmd_tailor(args, Tracker(conn), config)
        elif args.command == "outreach":
            cmd_outreach(args, Tracker(conn), config)
        else:
            handler = {
                "add-job": cmd_add_job, "list": cmd_list, "show": cmd_show,
                "status": cmd_status, "funnel": cmd_funnel, "add-contact": cmd_add_contact,
            }[args.command]
            handler(args, Tracker(conn))
    except (KeyError, ValueError, FileNotFoundError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
