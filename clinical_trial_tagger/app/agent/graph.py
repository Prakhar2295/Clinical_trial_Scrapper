from langgraph.graph import END, START, StateGraph

from app.agent.nodes.classifier import classifier_node
from app.agent.nodes.deep_reader import deep_reader_node
from app.agent.nodes.extractor import extractor_node
from app.agent.nodes.finalizer import finalizer_node
from app.agent.nodes.retriever import retriever_node
from app.agent.state import AgentState
from app.core.config import settings


def _route_after_classifier(state: AgentState) -> str:
    if state.get("confidence", 0.0) >= settings.confidence_threshold:
        return "finalizer"
    if state.get("fallback_triggered"):
        # Already read up to MAX_PAGES_FALLBACK once; further deep-reads won't
        # surface more content, so stop looping and finalize with what we have.
        return "finalizer"
    return "deep_reader"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("extractor", extractor_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("deep_reader", deep_reader_node)
    graph.add_node("finalizer", finalizer_node)

    graph.add_edge(START, "extractor")
    graph.add_edge("extractor", "retriever")
    graph.add_edge("retriever", "classifier")
    graph.add_conditional_edges(
        "classifier",
        _route_after_classifier,
        {"finalizer": "finalizer", "deep_reader": "deep_reader"},
    )
    graph.add_edge("deep_reader", "retriever")
    graph.add_edge("finalizer", END)

    return graph.compile()


agent_graph = build_graph()
