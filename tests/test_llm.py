from types import SimpleNamespace

import ollama
import pytest
from pydantic import BaseModel

from monday.config import Config
from monday.llm import LLM, LLMError


class Skills(BaseModel):
    skills: list[str]


class FakeClient:
    """Returns scripted replies per model; an Exception reply is raised."""

    def __init__(self, replies: dict):
        self.replies = replies
        self.calls = []

    def chat(self, model, messages, format, options):
        self.calls.append(model)
        reply = self.replies[model]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(message=SimpleNamespace(content=reply))


def make_llm(replies):
    config = Config(agents={"matcher": list(replies)})
    return LLM(config, client=FakeClient(replies))


def test_returns_typed_output_and_model():
    llm = make_llm({"small": '{"skills": ["Python"]}'})
    result = llm.structured("matcher", "sys", "jd", Skills)
    assert result.output.skills == ["Python"]
    assert result.model == "small"


def test_falls_back_on_error_and_on_bad_output():
    llm = make_llm({
        "down": ollama.ResponseError("rate limited", 429),
        "garbled": '{"skills": "not a list"}',
        "good": '{"skills": ["SQL"]}',
    })
    result = llm.structured("matcher", "sys", "jd", Skills)
    assert result.model == "good"
    assert llm.client.calls == ["down", "garbled", "good"]


def test_raises_when_every_model_fails():
    llm = make_llm({"a": ConnectionError("offline")})
    with pytest.raises(LLMError, match="a: ConnectionError"):
        llm.structured("matcher", "sys", "jd", Skills)


def test_unknown_agent_is_a_config_error():
    with pytest.raises(KeyError, match="nope"):
        make_llm({"a": "{}"}).structured("nope", "sys", "p", Skills)
