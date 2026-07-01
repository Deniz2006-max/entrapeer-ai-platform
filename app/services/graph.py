import logging
import os

from langgraph.graph import END, StateGraph

from app.agents.discovery import discovery_agent_node
from app.agents.peer import peer_agent_node
from app.agents.structuring import structuring_agent_node
from app.models.state import AgentState

logger = logging.getLogger(__name__)

REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")


# ---------------------------------------------------------------------------
# Routing functions for conditional edges
# ---------------------------------------------------------------------------
def _route_from_peer(state: AgentState) -> str:
    """
    After Peer Agent runs, decide the next node.

    current_step values set by Peer Agent:
      "discovery"  → BUSINESS_CRISIS path  → DiscoveryAgent
      "completed"  → DIRECT_ANSWER or CODE_REQUEST handled → END
      "rejected"   → OUT_OF_SCOPE → END
      anything else → END (safe fallback)
    """
    step = state.get("current_step", "end")
    if step == "discovery":
        return "discovery"
    return END


def _route_from_discovery(state: AgentState) -> str:
    """
    After Discovery Agent runs, decide the next node.

    interview_turns semantics (set by discovery_agent_node):
      1 → Q1 just asked (yönelim sorusu)  → pause (turns < 4)
      2 → Q2 just asked (konu sorusu 1)   → pause (turns < 4)
      3 → Q3 just asked (konu sorusu 2)   → pause (turns < 4)
      4 → Q4 just asked (konu sorusu 3)   → interview complete → Structuring Agent
      "structuring" step  → explicit override  → Structuring Agent

    NOTE: For turns 2-4, the /respond endpoint calls discovery_agent_node
    or structuring_agent_node DIRECTLY (bypassing the graph), so this edge is
    only exercised on the initial graph invocation (turn 0 → turn 1).
    It is kept for conceptual completeness and future LangGraph-native resume.
    """
    step = state.get("current_step", "end")
    turns = int(state.get("interview_turns", 0))

    if step == "structuring" or turns >= 4:
        return "structuring"
    return END  # awaiting_response: pause until user answers via /respond


# ---------------------------------------------------------------------------
# Build workflow topology (no checkpointer – injected at compile time)
# ---------------------------------------------------------------------------
def _build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("peer", peer_agent_node)
    workflow.add_node("discovery", discovery_agent_node)
    workflow.add_node("structuring", structuring_agent_node)

    workflow.set_entry_point("peer")

    workflow.add_conditional_edges(
        source="peer",
        path=_route_from_peer,
        path_map={"discovery": "discovery", END: END},
    )
    # Discovery pauses here – user must answer questions via /respond.
    # Only continues to structuring if current_step == "structuring" (edge case).
    workflow.add_conditional_edges(
        source="discovery",
        path=_route_from_discovery,
        path_map={"structuring": "structuring", END: END},
    )
    workflow.add_edge("structuring", END)

    return workflow


# ---------------------------------------------------------------------------
# Compile helpers – called from the FastAPI lifespan context
# ---------------------------------------------------------------------------
def compile_with_memory() -> object:
    """Compile graph backed by an in-process MemorySaver (dev / fallback)."""
    from langgraph.checkpoint.memory import MemorySaver
    graph = _build_workflow().compile(checkpointer=MemorySaver())
    logger.warning(
        "Using MemorySaver – sessions will NOT persist across restarts. "
        "Start the app with a running Redis to enable persistence."
    )
    return graph


async def compile_with_redis() -> object:
    """
    Compile graph with AsyncRedisSaver.

    Must be called inside an async context (e.g. FastAPI lifespan) because
    AsyncRedisSaver.from_conn_string() is an async context manager that must
    remain open for the lifetime of the graph.

    Returns (graph, checkpointer_ctx) where checkpointer_ctx must be kept
    alive (i.e. not exited) while the graph is in use.
    """
    from langgraph.checkpoint.redis import AsyncRedisSaver

    checkpointer_ctx = AsyncRedisSaver.from_conn_string(REDIS_URL)
    checkpointer = await checkpointer_ctx.__aenter__()
    await checkpointer.asetup()

    graph = _build_workflow().compile(checkpointer=checkpointer)
    logger.info(
        "LangGraph compiled with AsyncRedisSaver – url=%s cp_type=%s",
        REDIS_URL,
        type(checkpointer).__name__,
    )
    return graph, checkpointer_ctx


# ---------------------------------------------------------------------------
# Module-level placeholder – replaced during FastAPI lifespan startup
# ---------------------------------------------------------------------------
app_graph = compile_with_memory()
