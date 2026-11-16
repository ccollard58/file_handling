"""Retry helper for transient LLM failures (model overloaded / high demand)."""
import logging
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, TypeVar

T = TypeVar("T")

_TRANSIENT_MARKERS = (
    "high demand",
    "overloaded",
    "unavailable",
    "temporarily",
    "try again",
    "rate limit",
    "too many requests",
    "resource_exhausted",
    "timed out",
    "timeout",
    "connection reset",
    "connection error",
    "bad gateway",
    "gateway",
    "status code: 429",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "status_code=429",
    "status_code=503",
    "429",
    "502",
    "503",
    "504",
)


def is_transient_error(exc: BaseException) -> bool:
    """Return True if the exception looks like a temporary capacity/network issue."""
    text = f"{type(exc).__name__}: {exc}".lower()
    # A CUDA kernel watchdog timeout is a local GPU execution failure, not a
    # transient server/network error. Retrying the same request usually repeats it.
    if "cuda error" in text and "launch timed out and was terminated" in text:
        return False

    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (429, 500, 502, 503, 504):
        if status == 429 and any(
            marker in text
            for marker in (
                "requests per day",
                "request per day",
                "daily quota",
                "per-day quota",
                "requests_per_day",
            )
        ):
            # Do not keep probing a daily cap unless Google supplied a reset delay.
            return _retry_after_seconds(exc) is not None
        return True
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _retry_after_seconds(exc: BaseException):
    """Read Retry-After or Google's textual retry delay when available."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass

    message = str(exc)
    patterns = (
        r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s(?:ec(?:ond)?s?)?",
        r"retryDelay[\"']?\s*[:=]\s*[\"']([0-9]+(?:\.[0-9]+)?)s[\"']",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return max(0.0, float(match.group(1)))
    return None


def call_with_retry(
    func: Callable[[], T],
    *,
    description: str = "LLM call",
    max_attempts: int = 6,
    base_delay: float = 3.0,
    max_delay: float = 90.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``func`` retrying transient errors with exponential backoff and jitter.

    Non-transient errors are raised immediately; after the final attempt the
    last transient error is re-raised.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            if attempt >= max_attempts or not is_transient_error(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= random.uniform(0.75, 1.25)
            server_delay = _retry_after_seconds(exc)
            if server_delay is not None:
                delay = max(delay, server_delay)
            logging.warning(
                "%s failed (attempt %d/%d, transient): %s. Retrying in %.1fs",
                description, attempt, max_attempts, exc, delay,
            )
            sleep(delay)
    raise RuntimeError("unreachable")
