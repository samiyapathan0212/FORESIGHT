"""
FORESIGHT — Scoring Service Client
==================================
Small, isolated helper layer the dashboard uses to talk to the D6 scoring
service (service/main.py). All HTTP access lives here so the dashboard pages
never make requests directly.

Design rules:
  * The service is READ-ONLY and authoritative for SKU scoring fields.
  * If the service is unreachable or returns anything unexpected, callers fall
    back to the local processed artifacts, so the dashboard keeps working.
  * No forecast or risk logic is recomputed here — values are passed through.

The base URL is read from the FORESIGHT_API_URL environment variable and
defaults to a local development server. No credentials are used or stored.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://foresight-scoring-api-a1w2.onrender.com"
DEFAULT_TIMEOUT = 2.0  # seconds — keep the UI responsive if the API is down

# Scoring fields the service is authoritative for. Inventory-position fields
# (Current_Stock, Safety_Stock, ...) are not returned by the API and continue
# to come from the local artifacts.
SCORING_FIELDS = (
    "Product_Name",
    "Category",
    "risk_action",
    "stockout_risk",
    "overstock_risk",
    "sales_at_risk_rs",
    "capital_locked_rs",
    "total_rupee_impact_rs",
    "priority_score",
)


def base_url() -> str:
    return os.environ.get("FORESIGHT_API_URL", DEFAULT_BASE_URL).rstrip("/")


class ScoringServiceError(RuntimeError):
    """Raised when the scoring service cannot be reached or returns bad data."""


def _request(path: str, payload: dict | None = None, timeout: float = DEFAULT_TIMEOUT) -> Any:
    url = f"{base_url()}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise ScoringServiceError(f"HTTP {exc.code} from {path}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ScoringServiceError(f"Scoring service unreachable at {base_url()}") from exc
    except json.JSONDecodeError as exc:
        raise ScoringServiceError(f"Invalid JSON from {path}") from exc


def get_health(timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """Return the /health payload, or None if the service is unavailable."""
    try:
        payload = _request("/health", timeout=timeout)
        return payload if isinstance(payload, dict) and payload.get("status") == "ok" else None
    except ScoringServiceError:
        return None


def score_sku(sku: str, timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """GET /score/{sku}. Returns the scoring payload, or None on 404/failure."""
    try:
        payload = _request(f"/score/{sku}", timeout=timeout)
    except ScoringServiceError:
        return None
    return payload if isinstance(payload, dict) and payload.get("SKU") else None


def score_batch(skus: list[str], timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """POST /score for a list of SKUs. Returns the batch payload, or None on failure."""
    if not skus:
        return None
    try:
        payload = _request("/score", payload={"skus": list(skus)}, timeout=timeout)
    except ScoringServiceError:
        return None
    return payload if isinstance(payload, dict) and "results" in payload else None


def apply_score_to_row(row, payload: dict | None):
    """Overlay service scoring fields onto an artifact row.

    The returned row keeps every artifact column (inventory position, severities,
    forecast internals) and replaces only the fields the service is authoritative
    for. If `payload` is None, the artifact row is returned unchanged.
    """
    if payload is None:
        return row
    updated = row.copy()
    for field in SCORING_FIELDS:
        if field in payload and payload[field] is not None:
            updated[field] = payload[field]
    return updated
