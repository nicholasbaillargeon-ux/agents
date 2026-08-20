"""LLM access for every agent, with absence treated as a normal state.

Uses the official Anthropic SDK. The base URL is configurable because the
homelab fronts Claude with a LiteLLM gateway that speaks the native Messages
API — swapping that for api.anthropic.com is a one-line env change and no
code change, which is the point.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import anthropic

from .config import Config

log = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """No key, or the endpoint failed. Callers degrade; they do not crash."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens,
                     self.output_tokens + other.output_tokens)


class LLM:
    """Thin, synchronous wrapper. Two models: one that writes, one that sorts."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.usage = Usage()
        self._client: anthropic.Anthropic | None = None
        if cfg.llm_api_key:
            self._client = anthropic.Anthropic(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url,
                timeout=120.0,
                max_retries=3,
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4000,
        fast: bool = False,
    ) -> str:
        """Return the text of one completion, or raise LLMUnavailable."""
        if self._client is None:
            raise LLMUnavailable("no AGENTS_LLM_API_KEY configured")
        model = model or (self.cfg.fast_model if fast else self.cfg.write_model)
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        try:
            resp = self._client.messages.create(**kwargs)
        except anthropic.NotFoundError as e:
            raise LLMUnavailable(f"model {model!r} not served by {self.cfg.llm_base_url}: {e}") from e
        except anthropic.AuthenticationError as e:
            raise LLMUnavailable(f"rejected credentials for {self.cfg.llm_base_url}: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMUnavailable(f"rate limited after retries: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMUnavailable(f"HTTP {e.status_code} from LLM: {e}") from e
        except anthropic.APIConnectionError as e:
            raise LLMUnavailable(f"cannot reach {self.cfg.llm_base_url}: {e}") from e

        if getattr(resp, "usage", None):
            self.usage = self.usage + Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        if getattr(resp, "stop_reason", None) == "refusal":
            raise LLMUnavailable("model declined the request")
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if not text.strip():
            raise LLMUnavailable(f"empty completion from {model} (stop_reason={resp.stop_reason})")
        return text.strip()

    def json(self, prompt: str, *, default, **kw):
        """Completion parsed as JSON. Malformed output degrades to `default`.

        Models fence JSON in markdown often enough that stripping it is part of
        the contract, not a hack.
        """
        try:
            raw = self.complete(prompt, **kw)
        except LLMUnavailable as e:
            log.warning("llm.json unavailable: %s", e)
            return default
        return parse_json(raw, default=default)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(raw: str, *, default):
    """Best-effort JSON out of model prose. Never raises."""
    for candidate in _candidates(raw):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    log.warning("could not parse JSON from %d chars of model output", len(raw))
    return default


def _candidates(raw: str):
    raw = raw.strip()
    yield raw
    for m in _FENCE.finditer(raw):
        yield m.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = raw.find(opener), raw.rfind(closer)
        if 0 <= i < j:
            yield raw[i : j + 1]


class FakeLLM(LLM):
    """Deterministic stand-in for tests and for --no-llm runs.

    Records every prompt so tests can assert on what an agent actually asked,
    which is the part that regresses silently.
    """

    def __init__(self, cfg: Config, responses=None, *, available: bool = True) -> None:
        self.cfg = cfg
        self.usage = Usage()
        self._client = None
        self.prompts: list[str] = []
        self._available = available
        self._responses = list(responses or [])
        self.default_response = "FAKE"

    @property
    def available(self) -> bool:
        return self._available

    def complete(self, prompt: str, **kw) -> str:
        self.prompts.append(prompt)
        if not self._available:
            raise LLMUnavailable("FakeLLM configured as unavailable")
        self.usage = self.usage + Usage(len(prompt) // 4, 32)
        if self._responses:
            nxt = self._responses.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        return self.default_response
