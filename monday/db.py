import sqlite3
from pathlib import Path

APPLICATION_STATUSES = (
    "discovered",    # JD saved, not yet scored
    "skipped",       # decided not to pursue
    "preparing",     # tailoring / outreach / prep drafts in progress
    "applied",
    "responded",     # recruiter or hiring manager replied
    "interviewing",
    "offer",
    "rejected",
    "withdrawn",
)

DRAFT_STATUSES = ("pending", "approved", "rejected")

_statuses = lambda values: ", ".join(f"'{v}'" for v in values)  # noqa: E731

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS companies (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    website     TEXT,
    notes       TEXT,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY,
    company_id   INTEGER NOT NULL REFERENCES companies(id),
    title        TEXT NOT NULL,
    url          TEXT,
    location     TEXT,
    description  TEXT NOT NULL,
    source       TEXT,
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS applications (
    id               INTEGER PRIMARY KEY,
    job_id           INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
    status           TEXT NOT NULL DEFAULT 'discovered'
                     CHECK (status IN ({_statuses(APPLICATION_STATUSES)})),
    match_score      REAL,
    next_action      TEXT,
    next_action_due  TEXT,
    created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS contacts (
    id            INTEGER PRIMARY KEY,
    company_id    INTEGER NOT NULL REFERENCES companies(id),
    name          TEXT NOT NULL,
    role          TEXT,
    email         TEXT,
    linkedin_url  TEXT,
    notes         TEXT,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Every agent output lands here as a draft. Nothing leaves the system without a human
-- approving it, and nothing is ever sent by the system itself.
CREATE TABLE IF NOT EXISTS drafts (
    id              INTEGER PRIMARY KEY,
    application_id  INTEGER NOT NULL REFERENCES applications(id),
    agent           TEXT NOT NULL,
    kind            TEXT NOT NULL,
    content         TEXT NOT NULL,   -- JSON produced by the agent
    edited_content  TEXT,            -- JSON after human edits; NULL if approved as-is
    model           TEXT NOT NULL,   -- model that produced the draft
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ({_statuses(DRAFT_STATUSES)})),
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at     TEXT,
    contact_id      INTEGER REFERENCES contacts(id),  -- outreach drafts: who it's for
    sent_at         TEXT             -- set when YOU mark it sent; MONDAY never sends
);

-- Append-only timeline. Funnel and time-in-stage metrics are computed from it.
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY,
    application_id  INTEGER NOT NULL REFERENCES applications(id),
    kind            TEXT NOT NULL,
    detail          TEXT,
    status          TEXT,            -- set when the event moved the application to a stage
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_application ON events(application_id);
CREATE INDEX IF NOT EXISTS idx_drafts_status ON drafts(status);
"""


# Columns added after a table first shipped. CREATE TABLE IF NOT EXISTS never alters an
# existing table, so older databases get them here.
MIGRATIONS = {
    "drafts": {"contact_id": "INTEGER REFERENCES contacts(id)", "sent_at": "TEXT"},
}


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in MIGRATIONS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, decl in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    conn.commit()


def connect(path: Path | str) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn
