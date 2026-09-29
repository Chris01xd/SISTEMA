"""Imports (or updates in place) flows/ml_model_comparison.json into a running Langflow.

    python langflow_integration/scripts/import_flow.py [--url http://127.0.0.1:7860] [--api-key KEY]

The flow is matched by its endpoint name ("ml-model-comparison"): an existing flow is
updated, never duplicated. The API key comes from --api-key, $LANGFLOW_API_KEY or
langflow_integration/.langflow/api_key.txt (created by --create-key on a local
auto-login instance)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parents[1]
FLOW_FILE = HERE / "flows" / "ml_model_comparison.json"
KEY_FILE = HERE / ".langflow" / "api_key.txt"


def create_key(client: httpx.Client) -> str:
    """Local development only: auto-login token -> new API key (Langflow with LANGFLOW_AUTO_LOGIN=true)."""
    token = client.get("/api/v1/auto_login").raise_for_status().json()["access_token"]
    key = client.post("/api/v1/api_key/", json={"name": "hospital-twin-backend"},
                      headers={"Authorization": f"Bearer {token}"}).raise_for_status().json()["api_key"]
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key, encoding="utf-8")
    return key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("LANGFLOW_URL", "http://127.0.0.1:7860"))
    parser.add_argument("--api-key", default=os.environ.get("LANGFLOW_API_KEY"))
    parser.add_argument("--create-key", action="store_true", help="Create an API key through auto-login (local dev).")
    args = parser.parse_args(argv)

    flow = json.loads(FLOW_FILE.read_text(encoding="utf-8"))
    with httpx.Client(base_url=args.url, timeout=120) as client:
        key = args.api_key or (KEY_FILE.read_text(encoding="utf-8").strip() if KEY_FILE.exists() else None)
        if args.create_key or not key:
            key = create_key(client)
            print(f"API key created and stored in {KEY_FILE}")
        headers = {"x-api-key": key}
        existing = client.get("/api/v1/flows/", params={"get_all": "true", "remove_example_flows": "true"}, headers=headers)
        existing.raise_for_status()
        flows = existing.json()
        flows = flows.get("items", flows) if isinstance(flows, dict) else flows
        match = next((f for f in flows if f.get("endpoint_name") == flow["endpoint_name"]), None)
        body = {"name": flow["name"], "description": flow["description"], "data": flow["data"], "endpoint_name": flow["endpoint_name"]}
        if match:
            response = client.patch(f"/api/v1/flows/{match['id']}", json=body, headers=headers)
            action = "updated"
        else:
            response = client.post("/api/v1/flows/", json=body, headers=headers)
            action = "created"
        if response.status_code >= 400:
            print(f"ERROR {response.status_code}: {response.text[:500]}", file=sys.stderr)
            return 1
        saved = response.json()
    print(json.dumps({"action": action, "flow_id": saved["id"], "endpoint_name": saved.get("endpoint_name"),
                      "run_url": f"{args.url}/api/v1/run/{saved.get('endpoint_name') or saved['id']}"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
