"""MONDAY review app.  Run:  streamlit run app.py

Every agent output lands here as a draft. Approving a draft marks it ready for you to use;
MONDAY never sends anything itself.
"""

import json
from datetime import date
from pathlib import Path
from urllib.parse import quote

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
            if exc.network:
                st.error("Couldn't reach Ollama's cloud (ollama.com) - your internet connection or DNS "
                         "dropped. MONDAY already retried a few times. Your work is saved; wait a moment and "
                         "try again. If this keeps happening, see *Network troubleshooting* in the README.")
                with st.expander("Technical details"):
                    st.code(str(exc), language=None)
            else:
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


OUTREACH_KINDS = {"cold_email": "Cold email", "linkedin_note": "LinkedIn connection note",
                  "linkedin_message": "LinkedIn message", "follow_up": "Follow-up email"}


def review_outreach(draft, content: dict):
    did = draft["id"]
    contact = tracker.get_contact(draft["contact_id"])
    st.markdown(f"**To:** {contact['name']}" + (f", {contact['role']}" if contact["role"] else "")
                + (f" · {contact['email']}" if contact["email"] else ""))
    if content.get("follow_up_without_earlier_message"):
        st.warning("No approved earlier message to this contact was found - this follow-up has nothing to refer to.")
    for p in content["problems"]:
        st.error(f"Failed check (after one retry): {p}")

    subject = st.text_input("Subject", content["subject"], key=f"subj{did}") if content["subject"] else ""
    body = st.text_area("Message", content["body"], key=f"body{did}", height=260)
    limit = content["max_chars"]
    st.caption(f"{len(body)} / {limit} characters" + ("  - over the limit!" if len(body) > limit else "")
               + f" · cites: {', '.join(content['fact_ids']) or 'nothing'}")

    edited = subject != content["subject"] or body != content["body"]
    issues = (pipeline.outreach_edit_check(tracker, config, did, subject, body) if edited
              else content["warnings"])
    for issue in issues:
        st.warning(("Your edit: " if edited else "") + issue)
    if content["contact_details_used"]:
        with st.expander("Details about the contact it used (from your notes)"):
            for d in content["contact_details_used"]:
                st.write(f"- {d}")

    b1, b2 = st.columns(2)
    if b1.button("Approve with edits" if edited else "Approve", key=f"ok{did}", type="primary"):
        tracker.review_draft(did, approve=True,
                             edited_content={**content, "subject": subject, "body": body} if edited else None)
        st.rerun()
    if b2.button("Reject", key=f"no{did}"):
        tracker.review_draft(did, approve=False)
        st.rerun()
    st.caption("Approving doesn't send anything. You'll send it yourself from the application page.")


PREP_GROUPS = {"technical": "Technical", "resume": "Your resume", "behavioral": "Behavioral",
               "gap": "Gaps", "company": "Why this company"}
BRIEF_TITLES = {"overview": "Overview", "products": "Products", "tech_and_engineering": "Tech & engineering",
                "recent_news": "Recent news", "interview_process": "Interview process",
                "culture_and_values": "Culture & values"}


def render_prep(content: dict, app, key: str):
    """A prep sheet: brief with sources, questions by group, STAR outlines, questions to ask."""
    src = {s["id"]: s for s in content["sources"]}
    st.download_button("Download as Markdown", pipeline.prep_markdown(app, content),
                       file_name=f"prep_{app['company']}_{app['title']}.md".replace(" ", "_"), key=f"md{key}")
    tabs = st.tabs(["Company brief", *PREP_GROUPS.values(), "STAR outlines", "Ask them"])

    with tabs[0]:
        if not content["sources"]:
            st.warning("No web research was available - this sheet is built from the JD and your facts only.")
        st.caption(f"{len(content['sources'])} sources · researched {content['researched_at'] or 'never'}"
                   " · every point links to where it came from")
        for section, title in BRIEF_TITLES.items():
            points = content["brief"].get(section) or []
            if points:
                st.markdown(f"**{title}**")
                for p in points:
                    refs = " ".join(f"[[{i}]]({src[i]['url']})" for i in p["source_ids"] if i in src)
                    st.markdown(f"- {p['text']} {refs}")
        with st.expander("Sources"):
            for s in content["sources"]:
                st.markdown(f"{s['id']}. [{s['title']}]({s['url']})")
        if content["research_notes"] or content["dropped"]:
            with st.expander("Filtered out or dropped by checks"):
                for n in content["research_notes"] + content["dropped"]:
                    st.write(f"- {n}")

    for tab, (cat, title) in zip(tabs[1:6], PREP_GROUPS.items()):
        with tab:
            qs = [q for q in content["questions"] if q["category"] == cat]
            if cat == "gap":
                st.caption("Questions about skills you don't have. Answer honestly: adjacent experience + how you'd ramp up.")
            if not qs:
                st.write("None.")
            for q in qs:
                st.markdown(f"**{q['question']}**")
                st.caption(q["why_they_ask"])
                for t in q["talking_points"]:
                    st.markdown(f"- {t}")
                if q["fact_ids"]:
                    st.caption(f"from your facts: {', '.join(q['fact_ids'])}")

    with tabs[6]:
        st.info("Drafts built from your facts. The situation and task are the model's framing - rewrite every "
                "part in your own words and answer the fill-in questions, then save it as a story you can reuse.")
        for i, s in enumerate(content["star_outlines"]):
            with st.form(f"star{key}-{i}"):
                st.markdown(f"**{s['title']}** · from {', '.join(s['fact_ids'])}")
                for f in s["fill_in"]:
                    st.markdown(f"- [ ] {f}")
                parts = {part: st.text_area(part.title(), s[part], height=70, key=f"{part}{key}-{i}")
                         for part in ("situation", "task", "action", "result")}
                if st.form_submit_button("Save as story in my Experience Bank"):
                    if all(parts[p] == s[p] for p in parts):
                        st.error("Rewrite it in your own words first - saving the model's draft as-is isn't a story.")
                    else:
                        backup = pipeline.save_story(config, {"title": s["title"], "fact_ids": s["fact_ids"],
                                                              "skills": [], **parts})
                        st.success(f"Saved. Previous bank backed up to {backup.name}.")

    with tabs[7]:
        for q in content["questions_to_ask"]:
            st.markdown(f"- {q}")


def review_prep(draft, content: dict):
    app = tracker.get_application(draft["application_id"])
    render_prep(content, app, f"r{draft['id']}")
    b1, b2 = st.columns(2)
    if b1.button("Approve", key=f"ok{draft['id']}", type="primary"):
        tracker.review_draft(draft["id"], approve=True)
        st.rerun()
    if b2.button("Reject", key=f"no{draft['id']}"):
        tracker.review_draft(draft["id"], approve=False)
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
            elif draft["kind"] in OUTREACH_KINDS:
                review_outreach(draft, content)
            elif draft["kind"] == "interview_prep":
                review_prep(draft, content)
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


def contacts_section(app_id: int, app):
    st.markdown("#### Contacts & outreach")
    st.caption("People at this company. Your notes are the ONLY source of personal details the Outreach "
               "agent may use - paste their About section, a recent post, how you found them.")

    for contact in tracker.contacts_for_application(app_id):
        cid = contact["id"]
        header = contact["name"] + (f" - {contact['role']}" if contact["role"] else "")
        with st.expander(header):
            with st.form(f"contact{cid}"):
                n1, n2 = st.columns(2)
                role = n1.text_input("Role", contact["role"] or "")
                email = n2.text_input("Email", contact["email"] or "")
                linkedin = st.text_input("LinkedIn URL", contact["linkedin_url"] or "")
                notes = st.text_area("Notes", contact["notes"] or "", height=100)
                if st.form_submit_button("Save contact"):
                    tracker.update_contact(cid, role=role or None, email=email or None,
                                           linkedin_url=linkedin or None, notes=notes or None)
                    st.rerun()
            cols = st.columns(len(OUTREACH_KINDS))
            for col, (kind, label) in zip(cols, OUTREACH_KINDS.items()):
                if col.button(f"Draft {label.lower()}", key=f"o{cid}{kind}"):
                    if run_agent(f"Outreach agent is drafting a {label.lower()}",
                                 lambda k=kind: pipeline.outreach(tracker, config, app_id, cid, k)):
                        st.success("Draft ready - see the Review queue.")

    with st.form(f"new_contact{app_id}", clear_on_submit=True):
        st.markdown("**Add a contact**")
        n1, n2 = st.columns(2)
        name = n1.text_input("Name")
        role = n2.text_input("Role", placeholder="Engineering Manager, recruiter, alum ...")
        e1, e2 = st.columns(2)
        email = e1.text_input("Email")
        linkedin = e2.text_input("LinkedIn URL")
        notes = st.text_area("Notes about them", height=90)
        if st.form_submit_button("Add contact"):
            if not name.strip():
                st.error("Name is required.")
            else:
                tracker.add_contact(app["company"], name.strip(), role=role or None, email=email or None,
                                    linkedin_url=linkedin or None, notes=notes or None)
                st.rerun()

    drafts = tracker.outreach_drafts(app_id)
    ready = [d for d in drafts if d["status"] == "approved" and not d["sent_at"]]
    if ready:
        st.markdown("**Approved - ready for you to send**")
    for d in ready:
        c = json.loads(d["edited_content"] or d["content"])
        with st.container(border=True):
            st.markdown(f"{OUTREACH_KINDS[d['kind']]} to **{d['contact_name']}** (draft #{d['id']})")
            if c.get("subject"):
                st.code(c["subject"], language=None)
            st.code(c["body"], language=None, wrap_lines=True)
            b1, b2, b3 = st.columns(3)
            if d["kind"] in ("cold_email", "follow_up") and d["contact_email"]:
                mailto = (f"mailto:{d['contact_email']}?subject={quote(c.get('subject', ''))}"
                          f"&body={quote(c['body'])}")
                b1.link_button("Open in my email app", mailto)
            elif d["contact_linkedin"]:
                b1.link_button("Open their LinkedIn", d["contact_linkedin"])
            if b2.button("I sent it", key=f"sent{d['id']}", type="primary"):
                tracker.mark_sent(d["id"], follow_up_in_days=config.follow_up_days)
                st.rerun()
            b3.caption(f"Marking it sent schedules a follow-up in {config.follow_up_days} days.")

    ready_ids = {d["id"] for d in ready}
    history = [d for d in drafts if d["id"] not in ready_ids]
    if history:
        st.dataframe(
            [{"draft": d["id"], "type": OUTREACH_KINDS.get(d["kind"], d["kind"]), "to": d["contact_name"],
              "status": d["status"], "sent (UTC)": d["sent_at"] or ""} for d in history],
            hide_index=True, width="stretch",
        )


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
    refresh = st.checkbox("Refresh company research (otherwise reuse what was found for this company)",
                          key=f"refresh{app_id}")
    if c3.button("Prepare for interview", key=f"p{app_id}"):
        if run_agent("Researching the company and writing your prep sheet (1-2 minutes)",
                     lambda: pipeline.interview_prep(tracker, config, app_id, refresh_research=refresh)):
            st.success("Prep sheet ready - see the Review queue.")

    with st.form(f"site{app_id}"):
        w1, w2 = st.columns([3, 1])
        website = w1.text_input("Company website (fetched directly as a trusted research source)",
                                app["company_website"] or "", placeholder="https://www.example.com")
        cached = tracker.latest_research(app["company_id"])
        w2.caption(f"Research cached: {len(cached[0])} sources, {cached[1]} UTC" if cached else "No research yet")
        if st.form_submit_button("Save website"):
            tracker.set_company_website(app["company_id"], website.strip() or None)
            st.rerun()

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

    contacts_section(app_id, app)

    approved_prep = tracker.latest_draft(app_id, "interview_prep")
    if approved_prep and approved_prep["status"] == "approved":
        with st.expander("Interview prep sheet", expanded=False):
            render_prep(json.loads(approved_prep["content"]), app, f"a{approved_prep['id']}")

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


# --- mock interview -----------------------------------------------------------------

CRITERIA_HELP = {"relevance": "answers the question asked", "structure": "easy to follow; STAR for behavioral",
                 "specificity": "concrete details and real numbers", "clarity": "concise, no rambling"}


def mock_setup():
    apps = [a for a in tracker.list_applications() if pipeline.latest_prep(tracker, a["id"])]
    if not apps:
        st.info("Mock interviews use an application's interview prep sheet. Open an application and click "
                "**Prepare for interview** first.")
        return
    labels = {a["id"]: f"#{a['id']} {a['title']} @ {a['company']}" for a in apps}
    with st.form("mock_setup"):
        app_id = st.selectbox("Practice for", list(labels), format_func=labels.get)
        n = st.slider("Questions", 3, 10, 5)
        cats = st.multiselect("Question types", list(PREP_GROUPS), default=list(PREP_GROUPS),
                              format_func=PREP_GROUPS.get)
        if st.form_submit_button("Start mock interview", type="primary"):
            try:
                session_id, questions = pipeline.start_mock(tracker, app_id, n, cats or list(PREP_GROUPS))
            except ValueError as exc:
                st.error(str(exc))
                return
            st.session_state["mock"] = {"session_id": session_id, "app_id": app_id, "questions": questions,
                                        "i": 0, "phase": "answer", "question": questions[0],
                                        "follow_up": False, "feedback": None, "answer": ""}
            st.rerun()


def show_feedback(m: dict):
    fb = m["feedback"]
    with st.expander("Your answer"):
        st.write(m["answer"])
    cols = st.columns(len(fb["scores"]) or 1)
    for col, c in zip(cols, fb["scores"]):
        col.metric(c["name"].title(), f"{c['score']}/5", help=CRITERIA_HELP.get(c["name"]))
    for c in fb["scores"]:
        st.caption(f"**{c['name'].title()}:** {c['comment']}")
    for note in fb["checks"]:
        st.warning(note)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**What worked**")
        for s in fb["strengths"]:
            st.markdown(f"- {s}")
    with s2:
        st.markdown("**Improve**")
        for s in fb["improvements"]:
            st.markdown(f"- {s}")
    if fb["stronger_answer"]:
        with st.expander("A stronger version of your answer (only your words + your facts)"):
            st.write(fb["stronger_answer"])
            if fb["fact_ids"]:
                st.caption(f"uses: {', '.join(fb['fact_ids'])}")
    elif fb["stronger_answer_withheld"]:
        st.caption("The suggested rewrite was withheld - it added things you didn't say: "
                   + "; ".join(fb["stronger_answer_withheld"]))
    if not m["follow_up"] and m["question"].get("talking_points"):
        with st.expander("What your prep sheet suggested"):
            for t in m["question"]["talking_points"]:
                st.markdown(f"- {t}")


def mock_summary(session_id: int):
    answers = tracker.practice_answers(session_id)
    st.subheader("Session summary")
    if not answers:
        st.write("No answers recorded.")
        return
    per: dict[str, list[int]] = {}
    for a in answers:
        for c in json.loads(a["feedback"])["scores"]:
            per.setdefault(c["name"], []).append(c["score"])
    cols = st.columns(len(per) or 1)
    for col, (name, vals) in zip(cols, per.items()):
        col.metric(name.title(), f"{sum(vals) / len(vals):.1f}/5")
    if per:
        weakest = min(per, key=lambda k: sum(per[k]) / len(per[k]))
        st.info(f"Weakest area this session: **{weakest}** - {CRITERIA_HELP.get(weakest, '')}.")
    for a in answers:
        q = json.loads(a["question"])
        tag = " (follow-up)" if a["is_follow_up"] else ""
        st.markdown(f"- **{a['score']:.1f}/5**{tag} {q['question']}" if a["score"] is not None
                    else f"- {q['question']}")


def page_mock():
    st.title("Mock interview")
    st.caption("Answer as you would out loud. You get scores, direct feedback, a follow-up question, and a "
               "stronger version of your answer built only from what you said and your Experience Bank.")
    m = st.session_state.get("mock")
    if not m:
        mock_setup()
        return

    if m["phase"] == "done":
        mock_summary(m["session_id"])
        if st.button("Start a new session", type="primary"):
            st.session_state.pop("mock")
            st.rerun()
        return

    total = len(m["questions"])
    st.progress(m["i"] / total, text=f"Question {m['i'] + 1} of {total}" + (" · follow-up" if m["follow_up"] else ""))
    q = m["question"]
    st.caption(PREP_GROUPS.get(q.get("category"), "Follow-up") if not m["follow_up"] else "Follow-up")
    st.markdown(f"### {q['question']}")

    if m["phase"] == "answer":
        answer = st.text_area("Your answer", key=f"ans{m['session_id']}-{m['i']}-{m['follow_up']}", height=220)
        st.caption(f"{len(answer.split())} words · ~{len(answer.split()) / 130 * 60:.0f}s spoken")
        a1, a2, a3 = st.columns(3)
        if a3.button("End session now", key="end_answer"):
            tracker.finish_practice(m["session_id"])
            m["phase"] = "done"
            st.rerun()
        if a1.button("Submit answer", type="primary"):
            result = run_agent("Interviewer is reviewing your answer", lambda: pipeline.answer_mock(
                tracker, config, m["session_id"], m["app_id"], q, answer, is_follow_up=m["follow_up"]))
            if result:
                m.update(feedback=result[0], answer=answer, phase="feedback")
                st.rerun()
        if a2.button("Skip this question"):
            _next_question(m)
            st.rerun()
    else:
        show_feedback(m)
        st.divider()
        follow = m["feedback"]["follow_up_question"]
        if follow and not m["follow_up"]:
            st.markdown(f"**The interviewer follows up:** {follow}")
        b1, b2, b3 = st.columns(3)
        if follow and not m["follow_up"] and b1.button("Answer the follow-up"):
            m.update(question={"question": follow, "category": "follow_up"}, follow_up=True, phase="answer")
            st.rerun()
        last = m["i"] + 1 >= total
        if b2.button("Finish" if last else "Next question", type="primary"):
            _next_question(m)
            st.rerun()
        if b3.button("End session now", key="end_feedback"):
            tracker.finish_practice(m["session_id"])
            m["phase"] = "done"
            st.rerun()


def _next_question(m: dict):
    m["i"] += 1
    if m["i"] >= len(m["questions"]):
        tracker.finish_practice(m["session_id"])
        m["phase"] = "done"
    else:
        m.update(question=m["questions"][m["i"]], follow_up=False, phase="answer", feedback=None, answer="")


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

    st.subheader("Interview practice")
    practice = tracker.practice_stats()
    if not practice["answers"]:
        st.write("No mock interviews yet.")
    else:
        cols = st.columns(2 + len(practice["criteria"]))
        cols[0].metric("Sessions", practice["sessions"])
        cols[1].metric("Answers", practice["answers"])
        for col, (name, avg) in zip(cols[2:], practice["criteria"].items()):
            col.metric(name.title(), f"{avg:.1f}/5")
        trend = practice["session_averages"]
        if len(trend) > 1:
            st.caption("Average score per session: " + " → ".join(f"{s:.1f}" for s in trend))

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
    st.Page(page_mock, title="Mock interview", icon="🎤"),
    st.Page(page_resume, title="My resume", icon="📄"),
    st.Page(page_dashboard, title="Dashboard", icon="📊"),
])
nav.run()
conn.close()
