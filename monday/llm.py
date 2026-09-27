"""One door to every model. Agents ask for a typed result; this module picks the model.

Local and Ollama-cloud models are served by the same local Ollama daemon, so a single
client covers both. Output is constrained with Ollama's JSON-schema `format` and then
validated with Pydantic, so agents never parse free text.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

import httpx
import ollama
from pydantic import BaseModel, ValidationError

from monday.config import Config

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Every configured model for an agent failed."""


@dataclass
class LLMResult(Generic[T]):
    output: T
    model: str  # which model actually produced the output; stored on every draft


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
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        failures = []
        for model in self.config.models_for(agent):
            try:
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    format=schema.model_json_schema(),
                    options={"temperature": temperature},
                )
                return LLMResult(schema.model_validate_json(response.message.content), model)
            except (ollama.ResponseError, ollama.RequestError, httpx.HTTPError,
                    ConnectionError, ValidationError) as exc:
                failures.append(f"{model}: {type(exc).__name__}: {exc}")
        raise LLMError(f"All models failed for agent '{agent}':\n  " + "\n  ".join(failures))
