"""Export the ML pipeline using the installed Langflow graph serializer."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lfx.components.input_output.chat import ChatInput
from lfx.components.input_output.chat_output import ChatOutput
from lfx.custom.custom_component.component import Component
from lfx.graph import Graph

STEPS = (
    "dataset_input", "data_validation", "eda", "preprocessing", "normal_models",
    "hybrid_models", "cross_validation", "model_evaluation", "cross_reference",
    "model_comparator", "statistical_tests", "best_model_selector", "explanation", "analysis_output",
)
# Parallel branches are drawn on a second row of the canvas.
SECOND_ROW = {"eda", "hybrid_models", "statistical_tests"}
EXPECTED_NODES = len(STEPS) + 2  # + Chat Input / Chat Output
EXPECTED_EDGES = 38


def build_flow() -> dict:
    graph = Graph()
    chat = ChatInput(should_store_message=False)
    output = ChatOutput(should_store_message=False)
    chat_id = graph.add_component(chat, "ChatInput-ml")
    producers = {}
    positions = {chat_id: (0, 0)}
    last_id = None
    for index, name in enumerate(STEPS):
        module = importlib.import_module(f"langflow_integration.components.ml_pipeline.{name}")
        cls = next(value for value in vars(module).values()
                   if isinstance(value, type) and issubclass(value, Component)
                   and value is not Component and value.__module__ == module.__name__)
        component = cls()
        node_id = graph.add_component(component, f"{component.name}-ml")
        positions[node_id] = ((index + 1) * 420, 300 if name in SECOND_ROW else 0)
        if name == "dataset_input":
            graph.add_component_edge(chat_id, ("message", "input_value"), node_id)
        for field in component.inputs:
            if field.name in producers:
                source_id, source_output = producers[field.name]
                graph.add_component_edge(source_id, (source_output, field.name), node_id)
        for port in component.outputs:
            producers[port.name] = (node_id, port.name)
        last_id = node_id
    output_id = graph.add_component(output, "ChatOutput-ml")
    graph.add_component_edge(last_id, ("message", "input_value"), output_id)
    positions[output_id] = ((len(STEPS) + 1) * 420, 0)
    graph.initialize()
    payload = graph.dump(
        name="ML model comparison",
        description=("CRISP-DM ML analysis on the public MIMIC-IV demo: EDA, 5 normal + 2 hybrid models with grid-search "
                     "tuning, nested grouped cross-validation, cross-reference, inferential tests, automatic best-model "
                     "selection and explanation. Input: JSON parameters. Output: JSON result."),
        endpoint_name="ml-model-comparison",
    )
    for node in payload["data"]["nodes"]:
        x, y = positions[node["id"]]
        node["position"] = {"x": x, "y": y}
    assert len(payload["data"]["nodes"]) == EXPECTED_NODES, len(payload["data"]["nodes"])
    assert len(payload["data"]["edges"]) == EXPECTED_EDGES, len(payload["data"]["edges"])
    return payload


if __name__ == "__main__":
    destination = ROOT / "langflow_integration/flows/ml_model_comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_flow()
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(payload['data']['nodes'])} nodes and {len(payload['data']['edges'])} edges to {destination}")
