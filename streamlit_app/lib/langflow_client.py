"""Minimal, standalone Langflow run-API client (no backend.app dependency).

Contract (same as backend/app/services/langflow_client.py, reimplemented here so this
app never needs the backend's settings/DB stack): POST /api/v1/run/{flow_endpoint} ->
outputs[0].outputs[0].results.message.text holds the flow's final JSON as a string."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:7860"
DEFAULT_TIMEOUT = 600.0
KEY_FILE = Path(__file__).resolve().parents[2] / "langflow_integration" / ".langflow" / "api_key.txt"


def default_api_key() -> str:
    """$LANGFLOW_API_KEY, else the key created by import_flow.py --create-key."""
    key = os.environ.get("LANGFLOW_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    return key


def is_reachable(base_url: str = DEFAULT_BASE_URL) -> bool:
    try:
        with httpx.Client(base_url=base_url, timeout=3.0) as client:
            return client.get("/health").status_code == 200
    except httpx.HTTPError:
        return False


def run_flow(endpoint: str, input_value: str, base_url: str = DEFAULT_BASE_URL, api_key: str = "",
            timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    api_key = api_key or default_api_key()
    headers = {"x-api-key": api_key} if api_key else {}
    body = {"input_value": input_value, "input_type": "chat", "output_type": "chat", "tweaks": {}}
    with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as client:
        response = client.post(f"/api/v1/run/{endpoint}", params={"stream": "false"}, json=body)
    response.raise_for_status()
    return response.json()


def extract_json_payload(run_response: dict[str, Any]) -> dict[str, Any]:
    first = run_response["outputs"][0]["outputs"][0]
    text = (
        (first.get("results") or {}).get("message", {}).get("text")
        or ((first.get("results") or {}).get("message", {}).get("data") or {}).get("text")
        or (first.get("artifacts") or {}).get("message")
    )
    if not isinstance(text, str) or not text.strip():
        raise ValueError("La respuesta de Langflow no contiene texto de salida (¿el flujo llega hasta el Chat Output?)")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("La salida del flujo no es un objeto JSON")
    return payload
