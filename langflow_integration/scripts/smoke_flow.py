"""Execute the exported Langflow graph locally against the prepared public demo.

Run with the integration's Python environment. No server or LLM is required.
The successful run persists real ML artifacts, just like the API.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lfx.graph import Graph


async def run(payload: dict, request: dict) -> dict:
    graph = Graph.from_payload(payload)
    results = await graph.arun(
        inputs=[{"input_value": json.dumps(request)}],
        outputs=["ChatOutput-ml"],
    )
    messages = [output.results.get("message") for result in results
                for output in result.outputs if output is not None]
    if not messages:
        raise AssertionError("The exported flow did not produce a chat message")
    message = messages[0]
    text = message.get("text") if isinstance(message, dict) else message.text
    return json.loads(text)


async def main() -> None:
    path = ROOT / "langflow_integration/flows/ml_model_comparison.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    invalid = await run(payload, {"dataset": "missing_dataset", "explain_with_llm": False})
    assert invalid["status"] == "error", invalid
    assert invalid["error"]["code"] == "invalid_request", invalid["error"]
    print("Invalid request: structured error reached Chat Output", flush=True)
    result = await run(json.loads(path.read_text(encoding="utf-8")), {
        "dataset": "mimic_icu", "explain_with_llm": False, "language": "es",
    })
    assert result["status"] == "success", result.get("error")
    assert result["models_evaluated"] == 7, result.get("errors")
    assert result["best_model"]["name"] == result["ranking"][0]["model"]
    assert result["analysis_source"] == "deterministic"
    print(json.dumps({key: result[key] for key in (
        "status", "run_id", "models_evaluated", "best_model", "duration_s",
    )}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
