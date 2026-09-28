from monday.agents.outreach import OutreachOutput, check_message, kinds
from tests.test_resume_tailor import BANK, REPORT

NOTES = "Leads the platform team. Wrote a post about migrating to FastAPI last month."
EMAIL = kinds()["cold_email"]
NOTE = kinds(200)["linkedin_note"]


def check(body, subject="Backend engineer - FastAPI services", kind=EMAIL, fact_ids=("r2",), details=()):
    out = OutreachOutput(subject=subject, body=body, fact_ids=list(fact_ids), contact_details_used=list(details))
    return check_message(out, kind, bank=BANK, report=REPORT, contact_name="Priya Shah",
                         contact_notes=NOTES, context="AI Engineer at Globex")


GOOD = ("Hi Priya, I read your post about migrating to FastAPI. I recently cut p95 latency from 800ms "
        "to 120ms on a booking service. Would you be open to a 15-minute chat about the AI Engineer role? "
        "Thanks, A")


def test_grounded_message_passes():
    c = check(GOOD, details=["post about migrating to FastAPI"])
    assert c.problems == [] and c.warnings == []


def test_invented_number_is_a_problem():
    c = check(GOOD.replace("from 800ms to 120ms", "by 85%"))
    assert any("85" in p for p in c.problems)


def test_time_ask_is_not_a_claim():
    assert check(GOOD.replace("15-minute", "20 minute")).problems == []


def test_placeholders_and_missing_subject():
    c = check("Hi [Name], I cut latency from 800ms to 120ms. Thanks, A", subject="")
    assert any("placeholders" in p for p in c.problems)
    assert any("subject" in p for p in c.problems)


def test_linkedin_note_length_limit():
    c = check("Hi Priya, " + "x" * 200 + " A", subject="", kind=NOTE)
    assert any("too long" in p for p in c.problems)


def test_fact_id_leaking_into_text_is_a_problem():
    c = check(GOOD.replace("on a booking service", "on a booking service (r2)"))
    assert any("leaked" in p for p in c.problems)


def test_connection_note_needs_no_sign_off():
    c = check("Hi Priya, I read your FastAPI migration post and cut p95 latency from 800ms to 120ms "
              "myself. Would love to connect.", subject="", kind=NOTE)
    assert c.problems == [] and not any("signed" in w for w in c.warnings)


def test_unknown_fact_id():
    assert any("unknown fact" in p for p in check(GOOD, fact_ids=["nope"]).problems)


def test_warnings_for_gaps_invented_details_names_and_cliches():
    body = ("I hope this email finds you well. I loved your talk at PyCon. I know Kubernetes well and "
            "cut p95 latency from 800ms to 120ms. Regards")
    c = check(body, details=["talk at PyCon"])
    joined = " | ".join(c.warnings)
    assert "Kubernetes" in joined            # a JD skill you don't have
    assert "talk at PyCon" in joined         # not in the notes you wrote
    assert "Priya" in joined                 # not addressed by name
    assert "signed" in joined                # no sign-off with your name
    assert "cliché" in joined
