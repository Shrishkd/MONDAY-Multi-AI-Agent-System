import json
from datetime import date

import pytest

from monday.db import connect
from monday.tracker import Tracker


@pytest.fixture
def tracker():
    conn = connect(":memory:")
    yield Tracker(conn)
    conn.close()


def test_add_job_opens_application_and_reuses_company(tracker):
    a = tracker.add_job("Acme", "Backend Engineer", "Python, SQL")
    b = tracker.add_job("acme", "Data Engineer", "SQL")
    assert tracker.get_application(a)["status"] == "discovered"
    assert tracker.conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 1
    assert {r["id"] for r in tracker.list_applications()} == {a, b}


def test_empty_description_rejected(tracker):
    with pytest.raises(ValueError):
        tracker.add_job("Acme", "Eng", "   ")


def test_status_changes_are_logged_and_validated(tracker):
    app = tracker.add_job("Acme", "Eng", "JD")
    tracker.set_status(app, "applied", "via referral")
    assert tracker.timeline(app)[-1]["detail"] == "discovered -> applied: via referral"
    with pytest.raises(ValueError):
        tracker.set_status(app, "hired-lol")


def test_funnel_counts_stages_reached_not_current(tracker):
    a = tracker.add_job("Acme", "Eng", "JD")
    b = tracker.add_job("Beta", "Eng", "JD")
    tracker.set_status(a, "applied")
    tracker.set_status(a, "interviewing")
    tracker.set_status(a, "rejected")
    tracker.set_status(b, "skipped")
    f = tracker.funnel()
    assert f["discovered"] == 2
    assert f["applied"] == 1 and f["interviewing"] == 1 and f["rejected"] == 1
    assert f["skipped"] == 1 and f["offer"] == 0


def test_draft_review_gate(tracker):
    app = tracker.add_job("Acme", "Eng", "JD")
    d1 = tracker.add_draft(app, "outreach", "cold_email", {"body": "hi"}, "gpt-oss:120b-cloud")
    d2 = tracker.add_draft(app, "outreach", "cold_email", {"body": "yo"}, "gpt-oss:120b-cloud")
    d3 = tracker.add_draft(app, "outreach", "cold_email", {"body": "hey"}, "gpt-oss:120b-cloud")
    assert len(tracker.pending_drafts()) == 3

    tracker.review_draft(d1, approve=True)
    tracker.review_draft(d2, approve=True, edited_content={"body": "hello"})
    tracker.review_draft(d3, approve=False)

    assert tracker.pending_drafts() == []
    edited = tracker.conn.execute("SELECT edited_content FROM drafts WHERE id = ?", (d2,)).fetchone()[0]
    assert json.loads(edited) == {"body": "hello"}
    assert tracker.draft_stats() == {"approved_as_is": 1, "approved_edited": 1, "rejected": 1, "pending": 0}
    with pytest.raises(ValueError, match="already"):
        tracker.review_draft(d1, approve=False)


def test_old_database_gets_new_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE drafts (id INTEGER PRIMARY KEY, application_id INTEGER, agent TEXT, kind TEXT,"
                " content TEXT, edited_content TEXT, model TEXT, status TEXT, created_at TEXT, reviewed_at TEXT)")
    old.close()
    conn = connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(drafts)")}
    assert {"contact_id", "sent_at"} <= cols
    conn.close()


def test_outreach_draft_sent_by_you_schedules_follow_up(tracker):
    app = tracker.add_job("Acme", "Eng", "JD")
    contact = tracker.add_contact("Acme", "Priya Shah", role="EM", notes="met at meetup")
    assert [c["name"] for c in tracker.contacts_for_application(app)] == ["Priya Shah"]

    d = tracker.add_draft(app, "outreach", "cold_email", {"body": "hi"}, "m", contact_id=contact)
    with pytest.raises(ValueError, match="only approved"):
        tracker.mark_sent(d)
    tracker.review_draft(d, approve=True)
    tracker.mark_sent(d, follow_up_in_days=5)

    assert tracker.get_draft(d)["sent_at"] is not None
    assert tracker.get_application(app)["next_action"].startswith("Follow up with Priya Shah")
    assert tracker.outreach_drafts(app)[0]["contact_name"] == "Priya Shah"
    with pytest.raises(ValueError, match="already"):
        tracker.mark_sent(d)


def test_update_contact_only_touches_allowed_fields(tracker):
    tracker.add_job("Acme", "Eng", "JD")
    c = tracker.add_contact("Acme", "Priya")
    tracker.update_contact(c, notes="new notes", company_id=999)
    row = tracker.get_contact(c)
    assert row["notes"] == "new notes" and row["company"] == "Acme"


def test_due_actions(tracker):
    app = tracker.add_job("Acme", "Eng", "JD")
    tracker.set_next_action(app, "follow up with recruiter", date(2026, 10, 1))
    assert tracker.due_actions(date(2026, 9, 30)) == []
    assert [r["id"] for r in tracker.due_actions(date(2026, 10, 1))] == [app]
