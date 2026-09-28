"""The Tracker: a lightweight CRM over every application.

State changes are plain code, not model output - an LLM should never be the source of
truth for where an application stands. Every change is written to the events timeline.
"""

import json
import sqlite3
from datetime import date

from monday.db import APPLICATION_STATUSES


class Tracker:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # --- companies & jobs -------------------------------------------------

    def company_id(self, name: str) -> int:
        name = name.strip()
        row = self.conn.execute("SELECT id FROM companies WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        return self.conn.execute("INSERT INTO companies (name) VALUES (?)", (name,)).lastrowid

    def add_job(self, company: str, title: str, description: str, url: str | None = None,
                location: str | None = None, source: str | None = None) -> int:
        """Save a JD and open an application for it. Returns the application id."""
        if not description.strip():
            raise ValueError("Job description is empty")
        with self.conn:
            job_id = self.conn.execute(
                "INSERT INTO jobs (company_id, title, url, location, description, source)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (self.company_id(company), title.strip(), url, location, description.strip(), source),
            ).lastrowid
            app_id = self.conn.execute(
                "INSERT INTO applications (job_id) VALUES (?)", (job_id,)
            ).lastrowid
            self._log(app_id, "created", f"{title} @ {company}", status="discovered")
        return app_id

    # --- applications -----------------------------------------------------

    def get_application(self, app_id: int) -> sqlite3.Row:
        row = self.conn.execute(
            """SELECT a.*, j.title, j.url, j.location, j.description, c.name AS company
               FROM applications a
               JOIN jobs j ON j.id = a.job_id
               JOIN companies c ON c.id = j.company_id
               WHERE a.id = ?""",
            (app_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"No application with id {app_id}")
        return row

    def list_applications(self, status: str | None = None) -> list[sqlite3.Row]:
        sql = """SELECT a.id, a.status, a.match_score, a.next_action, a.next_action_due,
                        a.updated_at, j.title, c.name AS company
                 FROM applications a
                 JOIN jobs j ON j.id = a.job_id
                 JOIN companies c ON c.id = j.company_id"""
        params: tuple = ()
        if status:
            sql += " WHERE a.status = ?"
            params = (status,)
        return self.conn.execute(sql + " ORDER BY a.updated_at DESC, a.id DESC", params).fetchall()

    def set_status(self, app_id: int, status: str, note: str | None = None) -> None:
        if status not in APPLICATION_STATUSES:
            raise ValueError(f"Unknown status '{status}'. Use one of: {', '.join(APPLICATION_STATUSES)}")
        old = self.get_application(app_id)["status"]
        if old == status:
            return
        with self.conn:
            self.conn.execute(
                "UPDATE applications SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, app_id),
            )
            self._log(app_id, "status", f"{old} -> {status}" + (f": {note}" if note else ""),
                      status=status)

    def set_match_score(self, app_id: int, score: float) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE applications SET match_score = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (score, app_id),
            )
            self._log(app_id, "scored", f"{score:.0f}")

    def set_next_action(self, app_id: int, action: str, due: date | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE applications SET next_action = ?, next_action_due = ?,"
                " updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (action, due.isoformat() if due else None, app_id),
            )
            self._log(app_id, "next_action", action + (f" (due {due})" if due else ""))

    def due_actions(self, on_or_before: date) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT a.id, a.next_action, a.next_action_due, j.title, c.name AS company
               FROM applications a
               JOIN jobs j ON j.id = a.job_id
               JOIN companies c ON c.id = j.company_id
               WHERE a.next_action_due IS NOT NULL AND a.next_action_due <= ?
               ORDER BY a.next_action_due""",
            (on_or_before.isoformat(),),
        ).fetchall()

    # --- contacts ---------------------------------------------------------

    def add_contact(self, company: str, name: str, role: str | None = None, email: str | None = None,
                    linkedin_url: str | None = None, notes: str | None = None) -> int:
        with self.conn:
            return self.conn.execute(
                "INSERT INTO contacts (company_id, name, role, email, linkedin_url, notes)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (self.company_id(company), name, role, email, linkedin_url, notes),
            ).lastrowid

    # --- drafts: the human approval gate ----------------------------------

    def add_draft(self, app_id: int, agent: str, kind: str, content: dict, model: str) -> int:
        with self.conn:
            draft_id = self.conn.execute(
                "INSERT INTO drafts (application_id, agent, kind, content, model) VALUES (?, ?, ?, ?, ?)",
                (app_id, agent, kind, json.dumps(content), model),
            ).lastrowid
            self._log(app_id, "draft", f"{kind} #{draft_id} by {agent} ({model})")
        return draft_id

    def latest_draft(self, app_id: int, kind: str, include_rejected: bool = False) -> sqlite3.Row | None:
        exclude = "" if include_rejected else " AND status != 'rejected'"
        return self.conn.execute(
            f"SELECT * FROM drafts WHERE application_id = ? AND kind = ?{exclude} ORDER BY id DESC LIMIT 1",
            (app_id, kind),
        ).fetchone()

    def pending_drafts(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM drafts WHERE status = 'pending' ORDER BY created_at, id"
        ).fetchall()

    def review_draft(self, draft_id: int, approve: bool, edited_content: dict | None = None) -> None:
        """Record the human decision. Approving marks the draft usable - it never sends anything."""
        draft = self.conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        if draft is None:
            raise KeyError(f"No draft with id {draft_id}")
        if draft["status"] != "pending":
            raise ValueError(f"Draft {draft_id} was already {draft['status']}")
        status = "approved" if approve else "rejected"
        edited = json.dumps(edited_content) if edited_content is not None else None
        with self.conn:
            self.conn.execute(
                "UPDATE drafts SET status = ?, edited_content = ?, reviewed_at = CURRENT_TIMESTAMP"
                " WHERE id = ?",
                (status, edited, draft_id),
            )
            detail = f"{draft['kind']} #{draft_id} {status}" + (" with edits" if edited else "")
            self._log(draft["application_id"], "review", detail)

    # --- metrics ----------------------------------------------------------

    def funnel(self) -> dict[str, int]:
        """How many applications ever reached each stage (not just sit there now)."""
        reached = {s: 0 for s in APPLICATION_STATUSES}
        rows = self.conn.execute(
            "SELECT status, COUNT(DISTINCT application_id) AS n FROM events"
            " WHERE status IS NOT NULL GROUP BY status"
        ).fetchall()
        reached.update({row["status"]: row["n"] for row in rows})
        return reached

    def draft_stats(self) -> dict[str, int]:
        """How often drafts were approved as-is vs edited vs rejected - a quality signal per agent."""
        row = self.conn.execute(
            """SELECT
                 SUM(status = 'approved' AND edited_content IS NULL) AS approved_as_is,
                 SUM(status = 'approved' AND edited_content IS NOT NULL) AS approved_edited,
                 SUM(status = 'rejected') AS rejected,
                 SUM(status = 'pending') AS pending
               FROM drafts"""
        ).fetchone()
        return {k: row[k] or 0 for k in row.keys()}

    def timeline(self, app_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT kind, detail, created_at FROM events WHERE application_id = ? ORDER BY id",
            (app_id,),
        ).fetchall()

    def _log(self, app_id: int, kind: str, detail: str | None = None, status: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events (application_id, kind, detail, status) VALUES (?, ?, ?, ?)",
            (app_id, kind, detail, status),
        )
