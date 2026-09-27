import argparse
import shutil
import sys
from pathlib import Path

import ollama

from monday.config import PROJECT_ROOT, load_config
from monday.db import APPLICATION_STATUSES, connect
from monday.experience import load_bank
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

    sub.add_parser("funnel", help="how many applications reached each stage")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "init":
        return cmd_init(args, config) or 0
    if args.command == "doctor":
        return cmd_doctor(args, config)

    conn = connect(config.database)
    try:
        handler = {
            "add-job": cmd_add_job, "list": cmd_list, "show": cmd_show,
            "status": cmd_status, "funnel": cmd_funnel,
        }[args.command]
        handler(args, Tracker(conn))
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
