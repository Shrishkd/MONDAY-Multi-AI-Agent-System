import pytest
from pydantic import ValidationError

from monday.cli import EXAMPLE_BANK
from monday.experience import ExperienceBank, load_bank


def bank(**overrides):
    data = {
        "profile": {"name": "A", "email": "a@example.com"},
        "roles": [{
            "company": "X", "title": "Eng", "start": 2023,
            "facts": [{"id": "f1", "text": "did a thing", "skills": ["Python"]}],
        }],
    }
    data.update(overrides)
    return ExperienceBank.model_validate(data)


def test_example_bank_is_valid():
    b = load_bank(EXAMPLE_BANK)
    assert "excorp-latency" in b.fact_index()
    assert "Redis" in b.all_skills()


def test_numeric_dates_are_kept_as_text():
    assert bank().roles[0].start == "2023"


def test_duplicate_fact_ids_rejected():
    with pytest.raises(ValidationError, match="Duplicate fact id"):
        bank(projects=[{"name": "P", "facts": [{"id": "f1", "text": "again"}]}])


def test_story_must_cite_existing_facts():
    story = {"id": "s", "title": "t", "situation": "s", "task": "t", "action": "a",
             "result": "r", "fact_ids": ["nope"]}
    with pytest.raises(ValidationError, match="unknown fact ids"):
        bank(stories=[story])


def test_missing_bank_explains_how_to_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="monday init"):
        load_bank(tmp_path / "missing.yaml")
