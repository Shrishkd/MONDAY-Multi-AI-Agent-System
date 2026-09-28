"""One door to every model. Agents ask for a typed result; this module picks the model.

Local and Ollama-cloud models are served by the same local Ollama daemon, so a single
client covers both. Output is validated with Pydantic, so agents never parse free text.

Local models honour Ollama's JSON-schema `format`; the cloud gpt-oss models ignore it.
So the schema is also spelled out in the system prompt, JSON is pulled out of any
surrounding prose, and a model that returns invalid JSON gets one chance to fix it.

Network hiccups (DNS failures, timeouts, 502/503/504 from the cloud) are retried on the
same model with a short backoff before moving on - they usually clear in seconds.
"""

import json
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import httpx
import ollama
from pydantic import BaseModel, ValidationError

from monday.config import Config

T = TypeVar("T", bound=BaseModel)

# Room for a long JD plus the Experience Bank. Ollama's default window is smaller and
# silently truncates the prompt. 8192 is also Llama 3's maximum.
NUM_CTX = 8192

# Seconds to wait before each retry of a transient network failure on the same model.
RETRY_DELAYS = (2, 5)

_TRANSPORT_ERRORS = (ollama.ResponseError, ollama.RequestError, httpx.HTTPError, ConnectionError)
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_NETWORK_MARKERS = ("no such host", "dial tcp", "connection refused", "connection reset", "timed out",
                    "timeout", "failed to connect", "network is unreachable", "tls handshake")


def is_network_error(exc: Exception) -> bool:
    """A failure that is likely to clear on retry (as opposed to 402 no credits, 404 no model...)."""
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return True
    if isinstance(exc, ollama.ResponseError):
        return exc.status_code in _TRANSIENT_STATUS or any(m in str(exc).lower() for m in _NETWORK_MARKERS)
    return False


class LLMError(RuntimeError):
    """Every configured model for an agent failed."""

    def __init__(self, message: str, network: bool = False):
        super().__init__(message)
        self.network = network   # True -> every failure was a connectivity problem


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
    def __init__(self, config: Config, client: ollama.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.config = config
        self.client = client or ollama.Client(host=config.ollama_host)
        self.sleep = sleep

    def _chat(self, **kwargs):
        """One chat call, retrying transient network failures."""
        for delay in (*RETRY_DELAYS, None):
            try:
                return self.client.chat(**kwargs)
            except _TRANSPORT_ERRORS as exc:
                if delay is None or not is_network_error(exc):
                    raise
                self.sleep(delay)

    def structured(
        self,
        agent: str,
        system: str,
        prompt: str,
        schema: type[T],
        temperature: float = 0.2,
        num_ctx: int = NUM_CTX,
    ) -> LLMResult[T]:
        """Ask the agent's models in order until one returns output matching `schema`."""
        failures = []
        network_failures = 0
        for model in self.config.models_for(agent):
            messages = [
                {"role": "system", "content": _with_schema(system, schema)},
                {"role": "user", "content": prompt},
            ]
            for attempt in (1, 2):
                try:
                    response = self._chat(
                        model=model,
                        messages=messages,
                        format=schema.model_json_schema(),
                        options={"temperature": temperature, "num_ctx": num_ctx},
                    )
                except _TRANSPORT_ERRORS as exc:
                    failures.append(f"{model}: {type(exc).__name__}: {exc}")
                    network_failures += is_network_error(exc)
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
        raise LLMError(f"All models failed for agent '{agent}':\n  " + "\n  ".join(failures),
                       network=bool(failures) and network_failures == len(failures))
