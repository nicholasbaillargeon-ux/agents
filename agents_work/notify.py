"""Push a brief to a phone, treating "not configured" as a normal state.

ntfy because it is the only free push path that needs no account, no OAuth and
no API key: a topic name is the entire credential. That property is what makes
it work from a systemd timer, which has no TTY to refresh a token with.

The topic name *is* the secret — anyone who knows it can read and post to it —
so it lives in .env next to the API keys and never in source or a brief.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

# ntfy drops a message whose body exceeds its limit rather than truncating it,
# so a brief that grows past this must be cut here, where the cut is visible.
MAX_BODY = 3800
DEFAULT_SERVER = "https://ntfy.sh"


@dataclass
class PushResult:
    sent: bool = False
    reason: str = ""

    @property
    def degradation(self) -> str:
        return "" if self.sent else f"push not delivered: {self.reason}"


def clip(text: str, limit: int = MAX_BODY) -> str:
    """Cut at a paragraph boundary if one is close, and say that it was cut.

    A phone notification that ends mid-number is worse than a short one: the
    reader cannot tell whether "the 10-year is at 4." is a truncation or a typo.
    """
    if len(text) <= limit:
        return text
    head = text[: limit - 40]
    cut = head.rfind("\n\n")
    if cut > limit * 0.6:
        head = head[:cut]
    return head.rstrip() + "\n\n…truncated — full brief on the dashboard."


class Push:
    def __init__(self, topic: str | None, *, server: str = DEFAULT_SERVER,
                 timeout: float = 15.0, enabled: bool = True) -> None:
        self.topic = (topic or "").strip()
        self.server = (server or DEFAULT_SERVER).rstrip("/")
        self.timeout = timeout
        self.enabled = enabled

    @property
    def available(self) -> bool:
        return bool(self.topic) and self.enabled

    def send(self, body: str, *, title: str = "", click: str = "",
             tags: str = "", priority: int = 3, markdown: bool = True) -> PushResult:
        """Post one notification. Never raises — a dead phone is not a dead brief."""
        if not self.topic:
            return PushResult(False, "no AGENTS_NTFY_TOPIC configured")
        if not self.enabled:
            return PushResult(False, "push disabled for this run")
        headers = {"Priority": str(priority)}
        if title:
            # ntfy headers are latin-1 on the wire; an em dash in a title is
            # enough to make httpx raise before the request is ever sent.
            headers["Title"] = title.encode("ascii", "replace").decode("ascii")
        if click:
            headers["Click"] = click
        if tags:
            headers["Tags"] = tags
        if markdown:
            headers["Markdown"] = "yes"
        try:
            r = httpx.post(f"{self.server}/{self.topic}",
                           content=clip(body).encode("utf-8"),
                           headers=headers, timeout=self.timeout)
        except httpx.HTTPError as e:
            log.warning("push failed: %s", e)
            return PushResult(False, f"{type(e).__name__} reaching {self.server}")
        if not (200 <= r.status_code < 300):
            return PushResult(False, f"HTTP {r.status_code} from {self.server}")
        return PushResult(True)


class FakePush(Push):
    """Records instead of sending, so a test can assert what would reach the phone."""

    def __init__(self, *, available: bool = True) -> None:
        super().__init__("test-topic" if available else "", enabled=available)
        self.sent: list[dict] = []

    def send(self, body: str, **kw) -> PushResult:
        if not self.available:
            return PushResult(False, "FakePush configured as unavailable")
        self.sent.append({"body": body, **kw})
        return PushResult(True)
