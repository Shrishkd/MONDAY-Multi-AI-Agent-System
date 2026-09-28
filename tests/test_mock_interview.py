import json
from types import SimpleNamespace

from monday.agents.mock_interview import Feedback, check_answer, check_stronger, evaluate, pick_questions
from monday.config import Config
from monday.db import connect
from monday.llm import LLM
from monday.tracker import Tracker
from tests.test_resume_tailor import BANK, REPORT

GOOD = ("At Acme I owned the booking service. p95 latency had crept up and checkout was timing out, so I "
        "profiled the endpoint, found repeated reads, and added a cache in front of them. That cut p95 "
        "latency from 800ms to 120ms, and the timeouts stopped. I also added a dashboard so the team "
        "could see latency regressions before users did.")


def test_clean_answer_has_no_notes():
    assert check_answer(GOOD, BANK, REPORT) == []


def test_short_and_long_answers():
    assert "Short" in check_answer("I made it faster.", BANK, REPORT)[0]
    assert "Long" in check_answer(GOOD + " more words" * 200, BANK, REPORT)[0]


def test_misremembered_number_flagged():
    notes = check_answer(GOOD.replace("to 120ms", "to 90ms"), BANK, REPORT)
    assert any("90" in n for n in notes)


def test_numbers_in_the_question_are_fine():
    answer = GOOD + " With 3 engineers on the team we shipped it in a sprint."
    assert check_answer(answer, BANK, REPORT, context="How did you work with 3 engineers?") == []


def test_gap_claim_flagged():
    notes = check_answer(GOOD + " I have deployed it on Kubernetes too.", BANK, REPORT)
    assert any("Kubernetes" in n for n in notes)


def feedback(stronger, fact_ids=("r2",)):
    return Feedback(scores=[], strengths=[], improvements=[], stronger_answer=stronger,
                    fact_ids=list(fact_ids), follow_up_question="How did you measure it?")


def test_stronger_answer_may_only_use_answer_and_facts():
    assert check_stronger(feedback("I cut p95 latency from 800ms to 120ms."), GOOD, BANK, REPORT) == []
    assert check_stronger(feedback("I cut latency by 85%."), GOOD, BANK, REPORT)
    assert check_stronger(feedback("I ran it on Kubernetes."), GOOD, BANK, REPORT)
    assert check_stronger(feedback("Fine.", ["nope"]), GOOD, BANK, REPORT)
    leaked = check_stronger(feedback("As shown in my r2 work, latency fell from 800ms to 120ms."), GOOD, BANK, REPORT)
    assert any("fact ids" in p for p in leaked)


def test_pick_questions_round_robin_and_limits():
    prep = {"questions": [{"category": "technical", "question": f"t{i}"} for i in range(5)]
            + [{"category": "behavioral", "question": f"b{i}"} for i in range(2)]}
    picked = pick_questions(prep, 4, ["technical", "behavioral"], seed=1)
    assert [q["category"] for q in picked] == ["technical", "behavioral", "technical", "behavioral"]
    assert len(pick_questions(prep, 10, ["behavioral"], seed=1)) == 2


class FakeClient:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, **kwargs):
        return SimpleNamespace(message=SimpleNamespace(content=self.reply))


def test_evaluate_scores_and_withholds_bad_rewrite():
    reply = json.dumps({
        "scores": [{"name": n, "score": s, "comment": "ok"} for n, s in
                   (("relevance", 5), ("structure", 4), ("specificity", 3), ("clarity", 4))],
        "strengths": ["concrete metric"], "improvements": ["say what you'd do differently"],
        "stronger_answer": "I cut p95 latency by 85%.", "fact_ids": ["r2"],
        "follow_up_question": "How did you pick what to cache?",
    })
    llm = LLM(Config(agents={"mock_interview": ["fake"]}), client=FakeClient(reply))
    content, mean, model = evaluate(llm, question={"question": "Tell me about a performance win",
                                                   "category": "behavioral"},
                                    answer=GOOD, bank=BANK, report=REPORT, company="Acme", role="Eng", jd_text="")
    assert mean == 4.0 and model == "fake"
    assert content["stronger_answer"] is None and content["stronger_answer_withheld"]
    assert content["follow_up_question"].startswith("How did you pick")


def test_practice_sessions_and_stats():
    t = Tracker(connect(":memory:"))
    app = t.add_job("Acme", "Eng", "JD")
    s1 = t.start_practice(app, [{"question": "q1"}])
    t.add_practice_answer(s1, {"question": "q1"}, "a", {"scores": [{"name": "clarity", "score": 2}]}, 2.0)
    t.finish_practice(s1)
    s2 = t.start_practice(app, [{"question": "q2"}])
    t.add_practice_answer(s2, {"question": "q2"}, "a", {"scores": [{"name": "clarity", "score": 4}]}, 4.0)
    stats = t.practice_stats()
    assert stats["answers"] == 2 and stats["sessions"] == 2
    assert stats["criteria"] == {"clarity": 3.0}
    assert stats["session_averages"] == [2.0, 4.0]
    assert "average 2.0/5" in t.timeline(app)[-2]["detail"]
