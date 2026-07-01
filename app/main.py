import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router, alias_router, v1_router
import app.services.graph as graph_module
import app.services.mongodb as mongo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s – %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── MongoDB ────────────────────────────────────────────────────────────
    mongo.init_mongo()

    # ── LangGraph + Redis checkpointer ────────────────────────────────────
    _redis_ctx = None
    try:
        graph, _redis_ctx = await graph_module.compile_with_redis()
        graph_module.app_graph = graph
        logger.info("Redis checkpointer active – sessions will persist.")
    except Exception as exc:
        logger.warning(
            "Redis checkpointer unavailable (%s) – keeping MemorySaver.", exc
        )

    yield  # ── application runs ──────────────────────────────────────────

    # ── Shutdown ───────────────────────────────────────────────────────────
    await mongo.close_mongo()

    if _redis_ctx is not None:
        try:
            await _redis_ctx.__aexit__(None, None, None)
            logger.info("Redis checkpointer connection closed.")
        except Exception as exc:
            logger.warning("Error closing Redis checkpointer: %s", exc)


app = FastAPI(
    title="ENTRAPEER",
    description="Multi-agent business problem analysis API powered by LangGraph.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
# Unified Smart Entry-Point: POST /v1/agent/execute (no /api prefix).
# Mounted AFTER canonical router so /api/v1/... paths take doc precedence,
# but BEFORE alias_router so the smart routing logic takes priority over
# the old dumb aliases.
app.include_router(v1_router)
# Universal catch-all aliases — must be mounted last.
app.include_router(alias_router)


@app.get("/health", tags=["health"])
async def health() -> dict:
    cp_type = type(graph_module.app_graph.checkpointer).__name__
    return {"status": "ok", "checkpointer": cp_type}
