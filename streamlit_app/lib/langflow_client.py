"""Minimal, standalone Langflow client (no backend.app dependency).

Contract (same as backend/app/services/langflow_client.py, reimplemented here so this
app never needs the backend's settings/DB stack): POST /api/v1/run/{flow_endpoint} ->
outputs[0].outputs[0].results.message.text holds the flow's final JSON as a string.

Besides running the flow it can diagnose the setup (server up? flow imported? API key
valid?), import / update the flow and start a local Langflow server, so the Streamlit
page can fix the usual problems itself."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:7860"
DEFAULT_TIMEOUT = 900.0  # the full pipeline takes ~3-4 min on the demo data; LLM retries can add a minute
LF_ROOT = Path(__file__).resolve().parents[2] / "langflow_integration"
KEY_FILE = LF_ROOT / ".langflow" / "api_key.txt"
FLOW_FILE = LF_ROOT / "flows" / "ml_model_comparison.json"
SERVER_LOG = LF_ROOT / ".langflow" / "streamlit_server.log"


class LangflowError(RuntimeError):
    """Failure with a message meant for the user (Spanish, actionable)."""


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


@dataclass
class FlowStatus:
    reachable: bool
    auth_ok: bool = False
    flow: dict[str, Any] | None = None  # the imported flow (id, name, endpoint_name, updated_at)
    version: str | None = None
    problem: str | None = None

    @property
    def ready(self) -> bool:
        return self.reachable and self.auth_ok and self.flow is not None


def flow_status(endpoint: str, base_url: str = DEFAULT_BASE_URL, api_key: str = "") -> FlowStatus:
    """Server up -> API key accepted -> flow with this endpoint name imported."""
    if not is_reachable(base_url):
        return FlowStatus(False, problem="No hay un servidor LangFlow respondiendo en esa URL.")
    api_key = api_key or default_api_key()
    if not api_key:
        return FlowStatus(True, problem="No hay API key de LangFlow (falta LANGFLOW_API_KEY o .langflow/api_key.txt).")
    headers = {"x-api-key": api_key}
    try:
        with httpx.Client(base_url=base_url, headers=headers, timeout=15.0) as client:
            version = None
            with contextlib.suppress(Exception):
                version = client.get("/api/v1/version").json().get("version")
            r = client.get("/api/v1/flows/", params={"get_all": "true", "remove_example_flows": "true", "header_flows": "true"})
    except httpx.HTTPError as exc:
        return FlowStatus(True, problem=f"No se pudo consultar LangFlow ({type(exc).__name__}).")
    if r.status_code in (401, 403):
        return FlowStatus(True, version=version, problem="LangFlow rechazó la API key (inválida o de otra instalación).")
    if r.status_code >= 400:
        return FlowStatus(True, version=version, problem=f"LangFlow respondió HTTP {r.status_code} al listar los flujos.")
    flows = r.json()
    flows = flows.get("items", flows) if isinstance(flows, dict) else flows
    match = next((f for f in flows if f.get("endpoint_name") == endpoint or f.get("id") == endpoint), None)
    if match is None:
        return FlowStatus(True, auth_ok=True, version=version, problem=f"El flujo '{endpoint}' no está importado en este servidor.")
    return FlowStatus(True, auth_ok=True, flow=match, version=version)


def import_flow(base_url: str = DEFAULT_BASE_URL, create_key: bool = False) -> dict[str, Any]:
    """Runs langflow_integration/scripts/import_flow.py in-process (creates the API key if needed).
    Returns its JSON summary; raises LangflowError with the script's message on failure."""
    spec = importlib.util.spec_from_file_location("lf_import_flow", LF_ROOT / "scripts" / "import_flow.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    out, err = io.StringIO(), io.StringIO()
    argv = ["--url", base_url] + (["--create-key"] if create_key else [])
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = module.main(argv)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403) and not create_key:
            return import_flow(base_url, create_key=True)  # stale key: make a new one through auto-login
        raise LangflowError(f"LangFlow respondió HTTP {exc.response.status_code} al importar el flujo.") from exc
    except httpx.HTTPError as exc:
        raise LangflowError(f"No se pudo conectar con LangFlow para importar el flujo ({type(exc).__name__}).") from exc
    if code != 0:
        raise LangflowError(f"La importación falló: {(err.getvalue() or out.getvalue()).strip()[:400]}")
    text = out.getvalue()
    return json.loads(text[text.index("{"):]) if "{" in text else {}


def start_server(port: int = 7860) -> Path:
    """Starts langflow_integration/scripts/start_langflow.(ps1|sh) detached; output goes to SERVER_LOG.
    Returns the log path. The server needs ~1-2 min before /health answers."""
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "LANGFLOW_PORT": str(port)}
    log = open(SERVER_LOG, "w", encoding="utf-8")  # noqa: SIM115 - handed to the child process
    if sys.platform == "win32":
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LF_ROOT / "scripts" / "start_langflow.ps1")]
        # CREATE_NO_WINDOW, not DETACHED_PROCESS: with no console at all the start script never launches langflow.exe
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=flags, cwd=str(LF_ROOT.parent))
    else:
        cmd = ["bash", str(LF_ROOT / "scripts" / "start_langflow.sh")]
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=str(LF_ROOT.parent))
    return SERVER_LOG


def server_log_tail(lines: int = 25) -> str:
    if not SERVER_LOG.exists():
        return ""
    return "\n".join(SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def langflow_installed() -> bool:
    return (LF_ROOT / ".venv" / ("Scripts/langflow.exe" if sys.platform == "win32" else "bin/langflow")).exists()


def _http_problem(response: httpx.Response) -> str:
    detail = ""
    with contextlib.suppress(Exception):
        body = response.json()
        detail = body.get("detail") or body.get("message") or ""
        if isinstance(detail, (dict, list)):
            detail = json.dumps(detail, ensure_ascii=False)[:300]
    hints = {
        401: "LangFlow rechazó la API key. Usá «Importar / actualizar flujo» para crear una nueva.",
        403: "LangFlow rechazó la API key. Usá «Importar / actualizar flujo» para crear una nueva.",
        404: "El flujo no existe en este servidor. Usá «Importar / actualizar flujo».",
        500: "Error interno de LangFlow al ejecutar el flujo (revisá la consola del servidor; "
             "si hay otra instancia vieja en el mismo puerto, detenela y reiniciá).",
    }
    msg = hints.get(response.status_code, f"LangFlow respondió HTTP {response.status_code}.")
    return f"{msg} Detalle: {str(detail)[:400]}" if detail else msg


def run_flow(endpoint: str, input_value: str, base_url: str = DEFAULT_BASE_URL, api_key: str = "",
             timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    api_key = api_key or default_api_key()
    headers = {"x-api-key": api_key} if api_key else {}
    body = {"input_value": input_value, "input_type": "chat", "output_type": "chat", "tweaks": {}}
    try:
        with httpx.Client(base_url=base_url, headers=headers, timeout=httpx.Timeout(timeout, connect=10.0)) as client:
            response = client.post(f"/api/v1/run/{endpoint}", params={"stream": "false"}, json=body)
    except httpx.TimeoutException as exc:
        raise LangflowError(f"El flujo no respondió en {int(timeout)} s. Puede seguir corriendo en el servidor; "
                            "probá de nuevo con menos folds o sin LLM.") from exc
    except httpx.HTTPError as exc:
        raise LangflowError(f"Se perdió la conexión con LangFlow ({type(exc).__name__}). ¿Se detuvo el servidor?") from exc
    if response.status_code >= 400:
        raise LangflowError(_http_problem(response))
    return response.json()


def extract_json_payload(run_response: dict[str, Any]) -> dict[str, Any]:
    try:
        first = run_response["outputs"][0]["outputs"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise LangflowError("La respuesta de LangFlow no tiene salidas (¿el flujo termina en un Chat Output?).") from exc
    results = first.get("results") or {}
    message = results.get("message") or {}
    text = (
        message.get("text")
        or (message.get("data") or {}).get("text")
        or (first.get("artifacts") or {}).get("message")
        or ((first.get("outputs") or {}).get("message") or {}).get("message")
    )
    if not isinstance(text, str) or not text.strip():
        raise LangflowError("La respuesta de LangFlow no contiene texto de salida (¿el flujo llega hasta el Chat Output?).")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LangflowError(f"La salida del flujo no es JSON válido: {text[:300]}") from exc
    if not isinstance(payload, dict):
        raise LangflowError("La salida del flujo no es un objeto JSON.")
    return payload
