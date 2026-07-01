import logging
import os
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

MONGO_DETAILS: str = os.getenv("MONGO_DETAILS", "mongodb://mongodb:27017")
DB_NAME = "entrapeer"
COLLECTION = "analyses"

# ---------------------------------------------------------------------------
# Module-level client – initialised once during FastAPI lifespan startup
# ---------------------------------------------------------------------------
_client: AsyncIOMotorClient | None = None


def init_mongo() -> None:
    """Create the motor client. Call once at application startup."""
    global _client
    _client = AsyncIOMotorClient(MONGO_DETAILS)
    logger.info("MongoDB client initialised – url=%s", MONGO_DETAILS)


async def close_mongo() -> None:
    """Close the motor client. Call at application shutdown."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB client closed.")


def _db() -> AsyncIOMotorDatabase:
    if _client is None:
        raise RuntimeError(
            "MongoDB client is not initialised. "
            "Call init_mongo() during application startup."
        )
    return _client[DB_NAME]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
async def save_analysis(
    user_id: str,
    thread_id: str,
    structured_problem: dict,
) -> str:
    """
    Persist a completed analysis report.
    Returns the inserted document's string ID.
    """
    doc = {
        "user_id": user_id,
        "thread_id": thread_id,
        "created_at": datetime.now(timezone.utc),
        "structured_problem": structured_problem,
    }
    result = await _db()[COLLECTION].insert_one(doc)
    logger.info("Analysis saved – user_id=%s thread_id=%s", user_id, thread_id)
    return str(result.inserted_id)


async def get_analyses(user_id: str) -> list[dict]:
    """
    Return all analysis reports for a given user_id, newest first.
    MongoDB `_id` is excluded; `created_at` is serialised to ISO string.
    """
    cursor = _db()[COLLECTION].find(
        {"user_id": user_id},
        {"_id": 0},
    ).sort("created_at", -1)

    docs = await cursor.to_list(length=200)

    # Convert datetime objects to ISO strings for JSON serialisation
    for doc in docs:
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()

    return docs
