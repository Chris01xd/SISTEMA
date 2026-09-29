"""Assembles the LangGraph ``StateGraph``:

    START -> receive_input -> validate_input -(invalid)-> handle_error -> END
                                  | valid
                           plan_twin_queries -> load_digital_twin_context -(twin error)-> handle_error
                                                         |
                                                 reason_or_analyze <-----------------+
                                                         |                           | ungrounded, retries left
                                                 generate_response -> validate_response
                                                                         | grounded          | retries exhausted
                                                                        END           fallback_response -> END
"""
from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from ai_module.graphs import nodes
from ai_module.graphs.nodes import GraphDependencies, instrument
from ai_module.graphs.state import GraphState

NODE_ORDER = (
    "receive_input", "validate_input", "plan_twin_queries", "load_digital_twin_context",
    "reason_or_analyze", "generate_response", "validate_response",
)


def _continue_or_fail(next_node: str) -> Callable[[GraphState], str]:
    def route(state: GraphState) -> str:
        return "handle_error" if state.get("status") in ("rejected", "failed") else next_node

    return route


def build_graph(deps: GraphDependencies):
    """Returns the compiled graph (a LangChain ``Runnable``: ``invoke``/``stream``/``ainvoke``)."""
    graph = StateGraph(GraphState)
    for name in (*NODE_ORDER, "fallback_response", "handle_error"):
        graph.add_node(name, instrument(name, getattr(nodes, name), deps))

    graph.add_edge(START, "receive_input")
    for current, following in zip(NODE_ORDER[:-2], NODE_ORDER[1:-1]):
        graph.add_conditional_edges(current, _continue_or_fail(following), [following, "handle_error"])
    graph.add_conditional_edges("generate_response", _continue_or_fail("validate_response"), ["validate_response", "handle_error"])

    def after_validation(state: GraphState) -> str:
        status = state.get("status")
        if status == "needs_regeneration":
            return "reason_or_analyze" if state.get("attempts", 0) <= deps.settings.graph_max_regenerations else "fallback_response"
        return "handle_error" if status in ("rejected", "failed") else END

    graph.add_conditional_edges("validate_response", after_validation, ["reason_or_analyze", "fallback_response", "handle_error", END])
    graph.add_edge("fallback_response", END)
    graph.add_edge("handle_error", END)
    return graph.compile()
