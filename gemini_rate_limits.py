"""Project/model quota enforcement for Gemini Developer API generation calls.

Google assigns RPM, TPM, and RPD values per project and model. Those values
are not available from the public model-list endpoint; the defaults below are
a snapshot from the user's AI Studio project and can be updated there.
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


_WINDOW_SECONDS = 60.0
_DAY_SECONDS = 24.0 * 60.0 * 60.0
DEFAULT_GEMINI_PROJECT_ID = "gen-lang-client-0321985034"

# Snapshot supplied from Google AI Studio for project
# gen-lang-client-0321985034 on 2026-11-08. Pass a different map through the
# Settings panel or GEMINI_RATE_LIMITS_JSON if the project's limits change.
DEFAULT_GEMINI_RATE_LIMITS = {
    "gemini-3.8-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
    "gemini-3.7-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
    "gemini-3-flash-preview": {"rpm": 5, "tpm": 250000, "rpd": 20},
    "gemini-3-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
    "gemini-3.1-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 500},
    "gemini-2.5-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
    "gemini-2.5-pro": {"rpm": 0, "tpm": 0, "rpd": 0},
    "gemini-3.6-flash": {"rpm": 5, "tpm": 250000, "rpd": 20},
    "gemini-3.5-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 500},
    "gemini-2.5-flash-lite": {"rpm": 10, "tpm": 250000, "rpd": 20},
    "gemini-2.0-flash": {"rpm": 0, "tpm": 0, "rpd": 0},
    "gemini-2.0-flash-lite": {"rpm": 0, "tpm": 0, "rpd": 0},
    "gemini-3.1-pro-preview": {"rpm": 0, "tpm": 0, "rpd": 0},
    "gemma-4-26b-a4b-it": {"rpm": 30, "tpm": 16000, "rpd": 14400},
    "gemma-4-31b-it": {"rpm": 30, "tpm": 16000, "rpd": 14400},
}


class GeminiRateLimitConfigurationError(ValueError):
    """Raised when no usable project/model quota has been configured."""


@dataclass(frozen=True)
class ModelRateLimits:
    rpm: Optional[int]
    tpm: Optional[int]
    rpd: Optional[int]


def parse_rate_limits(value) -> Dict[str, ModelRateLimits]:
    """Convert a JSON-compatible model->quota mapping to validated limits."""
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise GeminiRateLimitConfigurationError(
                f"Gemini rate limits must be valid JSON: {exc}"
            ) from exc
    if not isinstance(value, dict):
        raise GeminiRateLimitConfigurationError(
            "Gemini rate limits must be a JSON object keyed by model name."
        )

    parsed = {}
    for model, raw_limits in value.items():
        if not isinstance(model, str) or not model.strip():
            raise GeminiRateLimitConfigurationError("Each rate-limit entry needs a model name.")
        if isinstance(raw_limits, ModelRateLimits):
            parsed[model.strip()] = raw_limits
            continue
        if not isinstance(raw_limits, dict):
            raise GeminiRateLimitConfigurationError(
                f"Rate limits for {model!r} must be an object with rpm, tpm, and rpd."
            )
        values = {}
        for metric in ("rpm", "tpm", "rpd"):
            if metric not in raw_limits:
                raise GeminiRateLimitConfigurationError(
                    f"Rate limits for {model!r} are missing {metric.upper()}."
                )
            raw = raw_limits[metric]
            if raw is None:
                values[metric] = None
                continue
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise GeminiRateLimitConfigurationError(
                    f"{metric.upper()} for {model!r} must be a non-negative integer or null."
                )
            values[metric] = raw
        parsed[model.strip()] = ModelRateLimits(**values)
    return parsed


def load_rate_limits(value=None) -> Dict[str, ModelRateLimits]:
    """Read limits from a mapping/JSON string or GEMINI_RATE_LIMITS_JSON."""
    if value is None:
        value = os.environ.get("GEMINI_RATE_LIMITS_JSON")
        if value is None:
            selected_project = os.environ.get("GEMINI_PROJECT_ID")
            if selected_project and selected_project != DEFAULT_GEMINI_PROJECT_ID:
                return {}
            value = DEFAULT_GEMINI_RATE_LIMITS
    return parse_rate_limits(value)


def count_input_tokens(client, model: str, contents) -> int:
    """Ask Gemini to count the exact text/image input tokens for this request."""
    result = client.models.count_tokens(model=model, contents=contents)
    total = getattr(result, "total_tokens", None)
    if total is None:
        raise RuntimeError(f"Gemini did not return an input token count for {model}.")
    return int(total)


class GeminiRateLimiter:
    """Persist per-project/model reservations across app and evaluator runs.

    TPM uses input tokens in a rolling 60-second window and RPD uses a rolling
    24-hour window, which is conservative around Google's quota-window resets.
    Failed attempts remain charged as reservations because Google may count
    requests that end in an error.
    """

    def __init__(self, api_key: str, limits, database_path=None, project_id=None):
        self.limits = parse_rate_limits(limits)
        key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        # Gemini Developer API keys do not expose their project ID. Set
        # GEMINI_PROJECT_ID to make separate keys in one project share state.
        selected_project = project_id or os.environ.get("GEMINI_PROJECT_ID")
        default_limits = parse_rate_limits(DEFAULT_GEMINI_RATE_LIMITS)
        self.project_scope = (
            selected_project
            or (DEFAULT_GEMINI_PROJECT_ID if self.limits == default_limits else key_fingerprint)
        )
        configured_path = database_path or os.environ.get("GEMINI_RATE_LIMIT_DB")
        self.database_path = str(
            Path(configured_path).expanduser()
            if configured_path
            else Path.home() / ".gemini_rate_limit_usage.sqlite3"
        )
        if (
            selected_project
            and selected_project != DEFAULT_GEMINI_PROJECT_ID
            and self.limits == default_limits
        ):
            raise GeminiRateLimitConfigurationError(
                "The built-in Gemini limits belong to project "
                f"{DEFAULT_GEMINI_PROJECT_ID}. Set GEMINI_RATE_LIMITS_JSON to the limits "
                "for GEMINI_PROJECT_ID before using a different project."
            )
        self._initialize_database()

    def _connect(self):
        conn = sqlite3.connect(self.database_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _initialize_database(self):
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS gemini_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_scope TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    reserved_tokens INTEGER NOT NULL
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS gemini_usage_lookup "
                "ON gemini_usage(project_scope, model, created_at)"
            )
        finally:
            conn.close()

    @staticmethod
    def _find_wait_time(events, now, limits, requested_tokens):
        wait_for = 0.0
        recent = [(created_at, tokens) for created_at, tokens in events if created_at > now - _WINDOW_SECONDS]
        daily = [(created_at, tokens) for created_at, tokens in events if created_at > now - _DAY_SECONDS]

        if limits.rpm is not None:
            if limits.rpm == 0:
                raise GeminiRateLimitConfigurationError("Gemini RPM is configured as zero.")
            if len(recent) >= limits.rpm:
                expiration_index = len(recent) - limits.rpm
                wait_for = max(wait_for, recent[expiration_index][0] + _WINDOW_SECONDS - now)

        if limits.tpm is not None:
            if requested_tokens > limits.tpm:
                raise GeminiRateLimitConfigurationError(
                    f"This request reserves {requested_tokens:,} tokens, exceeding the configured "
                    f"input TPM limit of {limits.tpm:,}. Reduce the text or image input."
                )
            recent_tokens = sum(tokens for _, tokens in recent)
            if recent_tokens + requested_tokens > limits.tpm:
                remaining = recent_tokens
                for created_at, tokens in recent:
                    remaining -= tokens
                    if remaining + requested_tokens <= limits.tpm:
                        wait_for = max(wait_for, created_at + _WINDOW_SECONDS - now)
                        break

        if limits.rpd is not None:
            if limits.rpd == 0:
                raise GeminiRateLimitConfigurationError("Gemini RPD is configured as zero.")
            if len(daily) >= limits.rpd:
                expiration_index = len(daily) - limits.rpd
                wait_for = max(wait_for, daily[expiration_index][0] + _DAY_SECONDS - now)

        return max(0.0, wait_for)

    @contextmanager
    def admit(self, model: str, input_tokens: int):
        """Wait until the model has quota, then reserve one request and input tokens."""
        quota_model = model.removeprefix("models/")
        # Google routes the retired 3.7 endpoint to 3.8, and the preview-less
        # 3 Flash name to its preview endpoint. Account both aliases together.
        quota_model = {
            "gemini-3.7-flash": "gemini-3.8-flash",
            "gemini-3-flash": "gemini-3-flash-preview",
        }.get(quota_model, quota_model)
        limits = self.limits.get(quota_model)
        if limits is None:
            raise GeminiRateLimitConfigurationError(
                f"No Gemini RPM/TPM/RPD limits are configured for {model!r}. "
                "Add the values shown for this model in Google AI Studio."
            )
        input_tokens = max(0, int(input_tokens))
        token_reservation = input_tokens
        if limits.tpm is not None and token_reservation > limits.tpm:
            raise GeminiRateLimitConfigurationError(
                f"The request needs up to {token_reservation:,} tokens, above the configured "
                f"{model} TPM limit of {limits.tpm:,}."
            )

        while True:
            now = time.time()
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "DELETE FROM gemini_usage WHERE created_at <= ?",
                    (now - _DAY_SECONDS,),
                )
                rows = conn.execute(
                    "SELECT created_at, reserved_tokens FROM gemini_usage "
                    "WHERE project_scope = ? AND model = ? AND created_at > ? "
                    "ORDER BY created_at",
                    (self.project_scope, quota_model, now - _DAY_SECONDS),
                ).fetchall()
                wait_for = self._find_wait_time(rows, now, limits, token_reservation)
                if wait_for <= 0:
                    conn.execute(
                        "INSERT INTO gemini_usage(project_scope, model, created_at, reserved_tokens) "
                        "VALUES (?, ?, ?, ?)",
                        (self.project_scope, quota_model, now, token_reservation),
                    )
                    conn.commit()
                    break
                conn.rollback()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

            logging.info(
                "Gemini %s quota is full; waiting %.1f seconds before the next request",
                model,
                wait_for,
            )
            time.sleep(max(wait_for, 0.1))

        # Retain the reservation for the full quota window on success or failure.
        yield


def image_contents(image_path: str, prompt: str):
    """Build Gemini contents for exact token counting of a single image request."""
    from google.genai import types

    import mimetypes

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/png"
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    return [types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt]
