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
        # Every distinct reason a call failed this run. `json()` swallows its
        # exception and callers are free to catch theirs, so without this the
        # only trace of a total outage was a log line nobody reads.
        self.failures: list[str] = []
        # Why the last reply ended. "max_tokens" means it was cut off mid-sentence
        # and whatever the caller asked for last is simply missing.
        self.last_stop_reason: str | None = None
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

    def _unavailable(self, msg: str) -> LLMUnavailable:
        """Record a failure, then hand back the exception for the caller to raise.

        Routing every raise through here is what makes an outage visible: the
        record outlives the exception, so a caller that degrades quietly — or
        `json()`, which catches on the caller's behalf — still leaves something
        the brief can report.
        """
        if msg not in self.failures:
            self.failures.append(msg)
        return LLMUnavailable(msg)

    @property
    def degradations(self) -> list[str]:
        """One line per distinct failure, short enough for the degraded banner."""
        return [f"LLM call failed: {_clip(m)}" for m in self.failures]

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
            raise self._unavailable("no AGENTS_LLM_API_KEY configured")
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
            raise self._unavailable(f"model {model!r} not served by {self.cfg.llm_base_url}: {e}") from e
        except anthropic.AuthenticationError as e:
            raise self._unavailable(f"rejected credentials for {self.cfg.llm_base_url}: {e}") from e
        except anthropic.RateLimitError as e:
            raise self._unavailable(f"rate limited after retries: {e}") from e
        except anthropic.APIStatusError as e:
            raise self._unavailable(f"HTTP {e.status_code} from LLM: {e}") from e
        except anthropic.APIConnectionError as e:
            raise self._unavailable(f"cannot reach {self.cfg.llm_base_url}: {e}") from e

        self.last_stop_reason = getattr(resp, "stop_reason", None)
        if self.last_stop_reason == "max_tokens":
            log.warning("completion from %s hit the %d-token cap; the reply is cut off",
                        model, max_tokens)
        if getattr(resp, "usage", None):
            self.usage = self.usage + Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        if getattr(resp, "stop_reason", None) == "refusal":
            raise self._unavailable("model declined the request")
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if not text.strip():
            raise self._unavailable(f"empty completion from {model} (stop_reason={resp.stop_reason})")
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
        parsed = parse_json(raw, default=default)
        if parsed is default:
            # The call succeeded and the output was still unusable. Same loss to
            # the reader as an outage, so it degrades the same way.
            msg = f"unparseable JSON from {kw.get('model') or 'the model'}"
            if msg not in self.failures:
                self.failures.append(msg)
        return parsed


def _clip(msg: str, limit: int = 140) -> str:
    """Gateway errors arrive with a JSON blob glued on; the banner wants a line.

    The blob restates the status code in three nested layers and is what makes
    the degraded banner unreadable, so it is cut at the brace rather than
    truncated mid-token.
    """
    one = " ".join(str(msg).split())
    head, sep, _ = one.partition(" - {")
    if sep:
        one = head
    return one if len(one) <= limit else one[: limit - 1] + "…"


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
        self.failures: list[str] = []
        self.prompts: list[str] = []
        self.last_stop_reason: str | None = None
        self._available = available
        self._responses = list(responses or [])
        self.default_response = "FAKE"

    @property
    def available(self) -> bool:
        return self._available

    def complete(self, prompt: str, **kw) -> str:
        self.prompts.append(prompt)
        if not self._available:
            raise self._unavailable("FakeLLM configured as unavailable")
        self.usage = self.usage + Usage(len(prompt) // 4, 32)
        if self._responses:
            nxt = self._responses.pop(0)
            if isinstance(nxt, Exception):
                # Injected failures must record like real ones, or a test that
                # stubs an outage would not exercise the degradation path.
                if isinstance(nxt, LLMUnavailable):
                    raise self._unavailable(str(nxt))
                raise nxt
            return nxt
        return self.default_response
