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


def test_due_actions(tracker):
    app = tracker.add_job("Acme", "Eng", "JD")
    tracker.set_next_action(app, "follow up with recruiter", date(2026, 10, 1))
    assert tracker.due_actions(date(2026, 9, 30)) == []
    assert [r["id"] for r in tracker.due_actions(date(2026, 10, 1))] == [app]
