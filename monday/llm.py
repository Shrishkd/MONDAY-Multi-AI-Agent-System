"""One door to every model. Agents ask for a typed result; this module picks the model.

Local and Ollama-cloud models are served by the same local Ollama daemon, so a single
client covers both. Output is validated with Pydantic, so agents never parse free text.

Local models honour Ollama's JSON-schema `format`; the cloud gpt-oss models ignore it.
So the schema is also spelled out in the system prompt, JSON is pulled out of any
surrounding prose, and a model that returns invalid JSON gets one chance to fix it.
"""

import json
from dataclasses import dataclass
from typing import Generic, TypeVar

import httpx
import ollama
from pydantic import BaseModel, ValidationError

from monday.config import Config

T = TypeVar("T", bound=BaseModel)

# Room for a long JD plus the Experience Bank. Ollama's default window is smaller and
# silently truncates the prompt. 8192 is also Llama 3's maximum.
NUM_CTX = 8192

_TRANSPORT_ERRORS = (ollama.ResponseError, ollama.RequestError, httpx.HTTPError, ConnectionError)


class LLMError(RuntimeError):
    """Every configured model for an agent failed."""


@dataclass
class LLMResult(Generic[T]):
    output: T
    model: str  # which model actually produced the output; stored on every draft


def extract_json(text: str) -> str:
    """The outermost {...} in a reply, ignoring markdown fences or prose around it."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def _with_schema(system: str, schema: type[BaseModel]) -> str:
    return (f"{system}\n\nRespond with ONLY a JSON object that matches this JSON schema. "
            f"No markdown, no explanation.\n{json.dumps(schema.model_json_schema())}")


class LLM:
    def __init__(self, config: Config, client: ollama.Client | None = None):
        self.config = config
        self.client = client or ollama.Client(host=config.ollama_host)

    def structured(
        self,
        agent: str,
        system: str,
        prompt: str,
        schema: type[T],
        temperature: float = 0.2,
    ) -> LLMResult[T]:
        """Ask the agent's models in order until one returns output matching `schema`."""
        failures = []
        for model in self.config.models_for(agent):
            messages = [
                {"role": "system", "content": _with_schema(system, schema)},
                {"role": "user", "content": prompt},
            ]
            for attempt in (1, 2):
                try:
                    response = self.client.chat(
                        model=model,
                        messages=messages,
                        format=schema.model_json_schema(),
                        options={"temperature": temperature, "num_ctx": NUM_CTX},
                    )
                except _TRANSPORT_ERRORS as exc:
                    failures.append(f"{model}: {type(exc).__name__}: {exc}")
                    break
                content = response.message.content
                try:
                    return LLMResult(schema.model_validate_json(extract_json(content)), model)
                except ValidationError as exc:
                    failures.append(f"{model} (attempt {attempt}): ValidationError: {exc}")
                    messages += [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": f"That reply was not valid:\n{exc}\n"
                                                    "Reply with only the corrected JSON object."},
                    ]
        raise LLMError(f"All models failed for agent '{agent}':\n  " + "\n  ".join(failures))
