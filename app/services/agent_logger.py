"""
AgentLogger – structured I/O logging for sub-agents.
Persists each agent invocation (inputs + outputs) to MongoDB so every
LLM call is auditable, reproducible, and visible in analytics dashboards.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

import app.services.mongodb as mongo_svc

logger = logging.getLogger(__name__)

AGENT_LOGS_COLLECTION = "agent_logs"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _db() -> AsyncIOMotorDatabase:
    return mongo_svc._db()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def log_agent_run(
    *,
    agent_name: str,
    thread_id: str,
    user_id: str | None,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    duration_ms: int | None = None,
    error: str | None = None,
) -> str:
    """
    Persist a single agent invocation record to the `agent_logs` collection.

    Fields stored:
      agent_name   – "ContentAgent" | "CodeAgent" | …
      thread_id    – LangGraph session identifier
      user_id      – optional end-user identifier
      input_data   – serialised prompt payload (ProblemTree, etc.)
      output_data  – serialised LLM response
      duration_ms  – wall-clock time of the LLM call
      error        – exception message if the call failed
      created_at   – UTC timestamp

    Returns the inserted document's string ID.
    """
    doc: dict[str, Any] = {
        "agent_name": agent_name,
        "thread_id": thread_id,
        "user_id": user_id,
        "input_data": input_data,
        "output_data": output_data,
        "duration_ms": duration_ms,
        "error": error,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        result = await _db()[AGENT_LOGS_COLLECTION].insert_one(doc)
        inserted_id = str(result.inserted_id)
        logger.info(
            "AgentLogger – persisted %s run thread_id=%s id=%s",
            agent_name,
            thread_id,
            inserted_id,
        )
        return inserted_id
    except Exception as exc:
        # Logging must never break the main flow
        logger.warning(
            "AgentLogger – MongoDB write failed agent=%s thread_id=%s: %s",
            agent_name,
            thread_id,
            exc,
        )
        return ""


async def get_agent_logs(
    thread_id: str,
    agent_name: str | None = None,
) -> list[dict]:
    """
    Retrieve agent log records for a given thread, optionally filtered by agent name.
    Results are sorted newest-first.
    """
    query: dict[str, Any] = {"thread_id": thread_id}
    if agent_name:
        query["agent_name"] = agent_name

    try:
        cursor = _db()[AGENT_LOGS_COLLECTION].find(
            query,
            {"_id": 0},
        ).sort("created_at", -1)
        docs = await cursor.to_list(length=100)
        for doc in docs:
            if isinstance(doc.get("created_at"), datetime):
                doc["created_at"] = doc["created_at"].isoformat()
        return docs
    except Exception as exc:
        logger.warning(
            "AgentLogger – MongoDB read failed thread_id=%s: %s", thread_id, exc
        )
        return []
