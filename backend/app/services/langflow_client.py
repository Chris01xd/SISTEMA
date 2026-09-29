"""Minimal client for Langflow's run API (`POST /api/v1/run/{flow}`).

Contract used by the ML analysis flow:
  request   {"input_value": "<JSON parameters>", "input_type": "chat", "output_type": "chat",
             "tweaks": {...}, "session_id": "..."}
  response  outputs[0].outputs[0].results.message.text  ->  the flow's final JSON (string)
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LangflowError(RuntimeError):
    """Base error; `status_code` is the HTTP status the backend should answer with."""

    status_code = 502


class LangflowUnavailableError(LangflowError):
    status_code = 503


class LangflowResponseError(LangflowError):
    status_code = 502


class LangflowClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 600.0, transport: httpx.BaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._transport = transport  # tests inject httpx.MockTransport

    def _client(self) -> httpx.Client:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        return httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout, transport=self._transport)

    def health(self) -> bool:
        try:
            with self._client() as client:
                return client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def run_flow(self, flow: str, input_value: str, tweaks: dict[str, Any] | None = None, session_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"input_value": input_value, "input_type": "chat", "output_type": "chat", "tweaks": tweaks or {}}
        if session_id:
            body["session_id"] = session_id
        try:
            with self._client() as client:
                response = client.post(f"/api/v1/run/{flow}", params={"stream": "false"}, json=body)
        except httpx.TimeoutException as exc:
            raise LangflowUnavailableError(f"Langflow did not answer within {self.timeout:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise LangflowUnavailableError(f"Could not connect to Langflow at {self.base_url} ({type(exc).__name__})") from exc
        if response.status_code in (401, 403):
            raise LangflowUnavailableError("Langflow rejected the API key (check LANGFLOW_API_KEY)")
        if response.status_code == 404:
            raise LangflowUnavailableError(f"Flow '{flow}' not found in Langflow (check LANGFLOW_FLOW_ID / import the flow)")
        if response.status_code >= 400:
            logger.warning("langflow run failed: status=%s", response.status_code)
            raise LangflowResponseError(f"Langflow returned HTTP {response.status_code} while running the flow")
        try:
            return response.json()
        except ValueError as exc:
            raise LangflowResponseError("Langflow returned a non-JSON response") from exc


def extract_message_text(run_response: dict[str, Any]) -> str:
    """The text of the first chat output of a run response (tolerates the shape
    differences between Langflow versions: results.message.text / .data.text / artifacts)."""
    try:
        first = run_response["outputs"][0]["outputs"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise LangflowResponseError("Langflow response has no outputs (did the flow run to its Chat Output?)") from exc
    candidates = [
        (first.get("results") or {}).get("message", {}).get("text"),
        ((first.get("results") or {}).get("message", {}).get("data") or {}).get("text"),
        (first.get("artifacts") or {}).get("message"),
        ((first.get("outputs") or {}).get("message") or {}).get("message"),
    ]
    for text in candidates:
        if isinstance(text, str) and text.strip():
            return text
    raise LangflowResponseError("Langflow response contains no message text")


def parse_flow_json(run_response: dict[str, Any]) -> dict[str, Any]:
    text = extract_message_text(run_response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LangflowResponseError("The flow output is not valid JSON (is the ML Analysis Output component connected?)") from exc
    if not isinstance(payload, dict):
        raise LangflowResponseError("The flow output is not a JSON object")
    return payload
