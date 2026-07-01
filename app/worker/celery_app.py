"""
celery_app.py — Celery application factory for ENTRAPEER.

Broker  : Redis  (same instance used by LangGraph checkpointer)
Backend : Redis  (task results stored here; polled by /v1/agent/status)

The module is importable by both the FastAPI process (to call .delay())
and the Celery worker process (to discover and execute tasks).
"""
import os

from celery import Celery

REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")

celery_app = Celery(
    "entrapeer",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Result TTL: 1 hour is plenty for interactive sessions
    result_expires=3600,
    # Worker concurrency: each task spawns its own asyncio event loop via
    # asyncio.run(), so prefork concurrency == number of parallel pipeline runs.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    # Visibility timeout slightly longer than the maximum expected LLM call
    # chain (~3 min for 3 discovery turns + structuring + sub-agents).
    broker_transport_options={"visibility_timeout": 600},
)
