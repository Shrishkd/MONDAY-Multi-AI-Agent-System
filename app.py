"""MONDAY review app.  Run:  streamlit run app.py

Every agent output lands here as a draft. Approving a draft marks it ready for you to use;
MONDAY never sends anything itself.
"""

import json
from datetime import date
from pathlib import Path

import pymupdf
import streamlit as st

import yaml

from monday import pipeline, resume_import
from monday.agents.job_matcher import MatchReport
from monday.config import load_config
from monday.db import APPLICATION_STATUSES, connect
from monday.experience import ExperienceBank, load_bank
from monday.latex import LatexError, compile_tex
from monday.llm import LLM, LLMError
from monday.tracker import Tracker

st.set_page_config(page_title="MONDAY", page_icon="📅", layout="wide")

config = load_config()
conn = connect(config.database)
tracker = Tracker(conn)


# --- helpers -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def pdf_page_png(path: str, mtime: float) -> bytes:
    with pymupdf.open(path) as doc:
        return doc[0].get_pixmap(dpi=120).tobytes("png")


def show_pdf(path: str | None, key: str):
    if not path or not Path(path).exists():
        st.warning("No PDF for this draft.")
        return
    p = Path(path)
    st.image(pdf_page_png(str(p), p.stat().st_mtime), width="stretch")
    st.download_button("Download PDF", p.read_bytes(), file_name=p.name, mime="application/pdf", key=f"pdf{key}")
    tex = p.with_suffix(".tex")
    if tex.exists():
        st.download_button("Download .tex", tex.read_bytes(), file_name=tex.name, key=f"tex{key}")


def app_label(app_id: int) -> str:
    a = tracker.get_application(app_id)
    return f"#{a['id']} {a['title']} @ {a['company']}"


def run_agent(label: str, fn):
    with st.spinner(f"{label} ..."):
        try:
            return fn()
        except LLMError as exc:
            st.error(f"{label} failed - no model could answer.\n\n{exc}")
        except Exception as exc:  # surface anything else instead of a blank page
            st.exception(exc)


# --- review queue ----------------------------------------------------------------

def review_match(draft, content: dict):
    report = MatchReport.model_validate(content)
    c1, c2, c3 = st.columns(3)
    c1.metric("Match score", f"{report.score:.0f}/100")
    c2.metric("Recommendation", report.recommendation.upper())
    c3.metric("Must-have gaps", len(report.gaps))
    rows = [{"requirement": m.skill, "type": kind, "you have it": "yes" if m.matched else "GAP",
             "evidence": ", ".join(m.evidence[:4])}
            for kind, ms in (("must", report.must_have), ("nice", report.nice_to_have)) for m in ms]
    st.dataframe(rows, hide_index=True, width="stretch")
    for flag in report.flags:
        st.warning(flag)

    b1, b2, b3 = st.columns(3)
    if b1.button("Pursue - start preparing", key=f"pursue{draft['id']}", type="primary"):
        tracker.review_draft(draft["id"], approve=True)
        tracker.set_status(draft["application_id"], "preparing", "decided to pursue")
        st.rerun()
    if b2.button("Skip this job", key=f"skip{draft['id']}"):
        tracker.review_draft(draft["id"], approve=True)
        tracker.set_status(draft["application_id"], "skipped", "decided to skip")
        st.rerun()
    if b3.button("Analysis is wrong - reject", key=f"rej{draft['id']}"):
        tracker.review_draft(draft["id"], approve=False)
        st.rerun()


def _texts(content: dict) -> tuple:
    return content["summary"], [[b["text"] for b in s["bullets"]] for s in content["sections"]]


def review_resume(draft, content: dict):
    did = draft["id"]
    state_key = f"edited{did}"
    shown = st.session_state.get(state_key, content)
    left, right = st.columns([1, 1])

    with left:
        if shown["compile_error"]:
            st.error(f"PDF build failed:\n\n{shown['compile_error']}")
        if shown["pages"] and shown["pages"] > 1:
            st.warning(f"Resume is {shown['pages']} pages - trim bullets before sending.")
        show_pdf(shown["pdf"], str(did))

    with right:
        cov = content["coverage"]
        c1, c2 = st.columns(2)
        c1.metric("ATS keyword coverage", f"{cov['after']:.0f}%", f"{cov['after'] - cov['before']:+.0f} pts")
        c2.metric("Rejected by fabrication check", len(content["violations"]))
        if cov["hidden"]:
            st.info("You have these but the resume doesn't show them: " + ", ".join(cov["hidden"]))
        if cov["gaps"]:
            st.caption("Real gaps, correctly not claimed: " + ", ".join(cov["gaps"]))
        if content["violations"]:
            with st.expander("What the fabrication check rejected"):
                for v in content["violations"]:
                    st.write(f"- {v}")

        st.markdown("**Edit before approving** - `**bold**` works. Each bullet shows the facts it cites.")
        summary = st.text_area("Summary", content["summary"], key=f"sum{did}", height=110)
        sections = []
        for si, sec in enumerate(content["sections"]):
            st.markdown(f"**{sec['label']}**")
            bullets = []
            for bi, b in enumerate(sec["bullets"]):
                tag = " · original wording (rewrite failed checks)" if b["fallback"] else ""
                text = st.text_area(f"cites {', '.join(b['fact_ids'])}{tag}", b["text"],
                                    key=f"b{did}-{si}-{bi}", height=80)
                bullets.append({**b, "text": text})
            sections.append({**sec, "bullets": bullets})
        edited = {**content, "summary": summary, "sections": sections}
        changed = summary != content["summary"] or sections != content["sections"]

        if changed:
            app = tracker.get_application(draft["application_id"])
            report = pipeline.latest_report(tracker, draft["application_id"], include_rejected=True)
            warnings = pipeline.edit_warnings(edited, load_bank(config.experience_bank), report)
            for w in warnings:
                st.warning(f"Your edit: {w}")
            if st.button("Rebuild PDF with my edits", key=f"rebuild{did}"):
                rebuilt = run_agent("Rebuilding PDF", lambda: pipeline.rebuild_resume(
                    config, draft["application_id"], app["company"], edited, report))
                if rebuilt:
                    st.session_state[state_key] = rebuilt
                    st.rerun()

        rebuilt = st.session_state.get(state_key)
        stale = rebuilt is not None and _texts(rebuilt) != _texts(edited)
        b1, b2 = st.columns(2)
        if changed and (not rebuilt or stale):
            b1.button("Approve", key=f"ok{did}", disabled=True, help="Rebuild the PDF with your edits first.")
        elif b1.button("Approve with edits" if rebuilt else "Approve as-is", key=f"ok{did}", type="primary"):
            tracker.review_draft(did, approve=True, edited_content=rebuilt)
            st.session_state.pop(state_key, None)
            st.rerun()
        if b2.button("Reject", key=f"no{did}"):
            tracker.review_draft(did, approve=False)
            st.session_state.pop(state_key, None)
            st.rerun()


def page_review():
    st.title("Review queue")
    st.caption("Nothing leaves MONDAY without your approval - and MONDAY never sends anything itself.")
    drafts = tracker.pending_drafts()
    if not drafts:
        st.success("Nothing waiting for review.")
        return
    for draft in drafts:
        content = json.loads(draft["content"])
        with st.container(border=True):
            st.subheader(f"{draft['kind'].replace('_', ' ').title()} · {app_label(draft['application_id'])}")
            st.caption(f"Draft #{draft['id']} by {draft['agent']} using {draft['model']} · {draft['created_at']} UTC")
            if draft["kind"] == "match_report":
                review_match(draft, content)
            elif draft["kind"] == "resume":
                review_resume(draft, content)
            else:
                st.json(content)


# --- applications ------------------------------------------------------------------

def page_applications():
    st.title("Applications")

    with st.expander("Add a job", expanded=not tracker.list_applications()):
        with st.form("add_job", clear_on_submit=True):
            c1, c2 = st.columns(2)
            company = c1.text_input("Company")
            title = c2.text_input("Role title")
            c3, c4, c5 = st.columns(3)
            url = c3.text_input("Job URL")
            location = c4.text_input("Location")
            source = c5.text_input("Found via", placeholder="LinkedIn, referral, careers page ...")
            jd = st.text_area("Job description (paste the full text)", height=220)
            match_now = st.checkbox("Score it against my Experience Bank right away", value=True)
            if st.form_submit_button("Add job", type="primary"):
                if not (company.strip() and title.strip() and jd.strip()):
                    st.error("Company, role title and job description are required.")
                else:
                    app_id = tracker.add_job(company, title, jd, url=url or None,
                                             location=location or None, source=source or None)
                    if match_now:
                        run_agent("Job Matcher is reading the JD", lambda: pipeline.match(tracker, config, app_id))
                    st.session_state["selected_app"] = app_id
                    st.rerun()

    status_filter = st.selectbox("Status", ["all", *APPLICATION_STATUSES])
    apps = tracker.list_applications(None if status_filter == "all" else status_filter)
    if not apps:
        st.info("No applications yet - add one above.")
        return
    st.dataframe(
        [{"id": a["id"], "company": a["company"], "title": a["title"], "status": a["status"],
          "score": None if a["match_score"] is None else round(a["match_score"]),
          "next action": a["next_action"], "due": a["next_action_due"], "updated (UTC)": a["updated_at"]}
         for a in apps],
        hide_index=True, width="stretch",
    )

    labels = {a["id"]: f"#{a['id']} {a['title']} @ {a['company']}" for a in apps}
    ids = list(labels)
    default = st.session_state.get("selected_app")
    selected = st.selectbox("Open application", ids, format_func=labels.get,
                            index=ids.index(default) if default in ids else 0)
    application_detail(selected)


def application_detail(app_id: int):
    app = tracker.get_application(app_id)
    st.divider()
    st.subheader(app_label(app_id))
    if app["url"]:
        st.markdown(f"[Job posting]({app['url']})")

    c1, c2, c3 = st.columns(3)
    if c1.button("Run Job Matcher", key=f"m{app_id}"):
        if run_agent("Job Matcher is reading the JD", lambda: pipeline.match(tracker, config, app_id)):
            st.rerun()
    if c2.button("Tailor resume", key=f"t{app_id}", type="primary"):
        if run_agent("Resume Tailor is drafting (about a minute)", lambda: pipeline.tailor(tracker, config, app_id)):
            st.success("Draft ready - see the Review queue.")
    c3.caption("Outreach and Interview Prep agents: coming next.")

    with st.form(f"status{app_id}"):
        s1, s2 = st.columns([1, 2])
        status = s1.selectbox("Status", APPLICATION_STATUSES, index=APPLICATION_STATUSES.index(app["status"]))
        note = s2.text_input("Note (optional)")
        n1, n2 = st.columns([2, 1])
        next_action = n1.text_input("Next action", app["next_action"] or "")
        due = n2.date_input("Due", value=date.fromisoformat(app["next_action_due"]) if app["next_action_due"] else None)
        if st.form_submit_button("Save"):
            tracker.set_status(app_id, status, note or None)
            if next_action != (app["next_action"] or "") or (due and due.isoformat() != app["next_action_due"]):
                tracker.set_next_action(app_id, next_action, due)
            st.rerun()

    with st.expander("Job description"):
        st.text(app["description"])
    with st.expander("Timeline", expanded=True):
        for e in tracker.timeline(app_id):
            st.write(f"`{e['created_at']}` **{e['kind']}** {e['detail'] or ''}")


# --- my resume ----------------------------------------------------------------------

def page_resume():
    st.title("My resume")
    st.caption("Upload a new version of your resume. The LaTeX file becomes the template every tailored "
               "resume is built from; its content becomes your Experience Bank - the only facts agents may use. "
               "Nothing changes until you press Apply, and the current version is backed up first.")

    try:
        current = load_bank(config.experience_bank)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Roles", len(current.roles))
        c2.metric("Projects", len(current.projects))
        c3.metric("Facts", len(current.facts()))
        c4.metric("Interview stories", len(current.stories))
    except Exception as exc:
        current = None
        st.warning(f"No usable Experience Bank yet: {exc}")
    st.caption(f"Template: `{config.resume_template}` · Bank: `{config.experience_bank}`")

    upload = st.file_uploader("Resume file", type=["tex", "pdf"],
                              help="LaTeX (.tex) updates both the template and the bank. PDF updates the bank only.")
    if upload is None:
        st.session_state.pop("import", None)
        return

    data = upload.getvalue()
    is_tex = upload.name.lower().endswith(".tex")
    upload_key = f"{upload.name}:{len(data)}"
    state = st.session_state.get("import")
    if not state or state["key"] != upload_key:
        state = st.session_state["import"] = {"key": upload_key}

    if is_tex:
        check = resume_import.check_template(data.decode("utf-8", errors="replace"))
        source = check.tex
        for fix in check.fixes:
            st.info(f"Fixed: {fix}")
        for problem in check.problems:
            st.warning(problem)
        if "preview" not in state:
            preview_dir = config.output_dir / "import-preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            tex_path = preview_dir / "preview.tex"
            tex_path.write_text(check.tex, encoding="utf-8")
            with st.spinner("Compiling a preview ..."):
                try:
                    state["preview"] = str(compile_tex(tex_path, config.pdflatex).pdf)
                except LatexError as exc:
                    state["preview"] = None
                    state["preview_error"] = str(exc)
        if state.get("preview_error"):
            st.error(f"This LaTeX doesn't compile, so it can't be used as the template:\n\n{state['preview_error']}")
    else:
        source = resume_import.pdf_text(data)
        st.warning("A PDF can update your Experience Bank, but not the LaTeX template - tailored resumes "
                   "will keep using the current template's layout, header and Education/Certifications. "
                   "Upload the .tex to change those too.")

    left, right = st.columns([1, 1])
    with left:
        if is_tex and state.get("preview"):
            show_pdf(state["preview"], "import")
        elif not is_tex:
            st.text_area("Text read from the PDF", source, height=500, disabled=True)

    with right:
        if st.button("Extract Experience Bank from this resume", type="primary"):
            result = run_agent("Reading your resume (about 20s)",
                               lambda: resume_import.extract_bank(LLM(config), source, current))
            if result:
                bank, warnings, model = result
                state.update(yaml=resume_import.bank_yaml(bank), warnings=warnings, model=model,
                             diff=resume_import.diff(current, bank))
                st.rerun()

        if "yaml" in state:
            d = state["diff"]
            st.markdown(f"**Proposed Experience Bank** (extracted by {state['model']}): "
                        f"{len(d['added'])} facts added · {len(d['reworded'])} reworded · {len(d['removed'])} removed")
            for kind in ("added", "reworded", "removed"):
                if d[kind]:
                    with st.expander(f"{kind.title()} ({len(d[kind])})", expanded=kind != "reworded"):
                        for line in d[kind]:
                            st.write(f"- {line}")
            for w in state["warnings"]:
                st.warning(f"Check this - the model may have invented it: {w}")
            edited_yaml = st.text_area("Review and edit before applying (YAML)", state["yaml"], height=420)

            can_template = is_tex and state.get("preview") and not check.problems
            replace_template = st.checkbox("Replace my LaTeX template", value=bool(can_template),
                                           disabled=not can_template)
            replace_bank = st.checkbox("Replace my Experience Bank", value=True)
            if st.button("Apply", type="primary", disabled=not (replace_template or replace_bank)):
                try:
                    bank = ExperienceBank.model_validate(yaml.safe_load(edited_yaml))
                except Exception as exc:
                    st.error(f"The YAML isn't a valid Experience Bank:\n\n{exc}")
                    return
                backup = resume_import.apply(
                    config,
                    bank if replace_bank else None,
                    check.tex if replace_template else None,
                    (upload.name, data),
                )
                st.session_state.pop("import", None)
                st.success(f"Applied. Previous version backed up to `{backup}`.")


# --- dashboard ----------------------------------------------------------------------

def page_dashboard():
    st.title("Dashboard")
    funnel = tracker.funnel()
    st.subheader("How far applications got")
    stages = ["discovered", "preparing", "applied", "responded", "interviewing", "offer"]
    cols = st.columns(len(stages))
    for col, stage in zip(cols, stages):
        col.metric(stage.title(), funnel[stage])
    st.caption(f"Skipped {funnel['skipped']} · rejected {funnel['rejected']} · withdrawn {funnel['withdrawn']}")

    st.subheader("Draft quality")
    stats = tracker.draft_stats()
    reviewed = stats["approved_as_is"] + stats["approved_edited"] + stats["rejected"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approved as-is", stats["approved_as_is"])
    c2.metric("Approved with edits", stats["approved_edited"])
    c3.metric("Rejected", stats["rejected"])
    c4.metric("Pending", stats["pending"])
    if reviewed:
        st.caption(f"{100 * stats['approved_as_is'] / reviewed:.0f}% of reviewed drafts needed no edits.")

    st.subheader("Due follow-ups")
    due = tracker.due_actions(date.today())
    if not due:
        st.write("Nothing due.")
    for d in due:
        st.write(f"**{d['next_action_due']}** · {d['company']} - {d['title']}: {d['next_action']}")


pending = len(tracker.pending_drafts())
nav = st.navigation([
    st.Page(page_review, title=f"Review queue ({pending})", icon="✅", default=True),
    st.Page(page_applications, title="Applications", icon="🗂️"),
    st.Page(page_resume, title="My resume", icon="📄"),
    st.Page(page_dashboard, title="Dashboard", icon="📊"),
])
nav.run()
conn.close()
