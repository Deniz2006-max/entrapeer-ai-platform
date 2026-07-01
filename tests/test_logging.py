"""
test_logging.py — Verifies that POST /v1/agent/execute emits the expected
INFO-level log entries to stdout/logger whenever a task is accepted.

Test categories
───────────────
  T901  Basic smoke: HTTP 200 + "V1/EXECUTE queued" log present
  T902  Log contains the routing decision (new_session) and task preview
  T903  Celery task_id is traceable in the log (observability)

All external I/O is mocked through the shared `client` fixture (conftest.py).
Celery's `run_agent_pipeline_task.delay()` is additionally mocked per-test so
no broker connection is required.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_TASK = "Satışlarımız son çeyrekte %40 düştü"


def _mock_celery_task(task_id: str = "mock-celery-id-001") -> MagicMock:
    """Return a MagicMock that mimics a Celery AsyncResult stub."""
    stub = MagicMock()
    stub.id = task_id
    return stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExecuteLogging:
    """T9xx — Logging coverage for the unified Celery entry-point."""

    # ── T901 ──────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_emits_queued_log(
        self,
        client: AsyncClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        T901 – POST /v1/agent/execute with a valid business task must:
          • Return HTTP 200 with {task_id, thread_id, status='queued'}
          • Emit an INFO log line containing 'V1/EXECUTE queued'
        """
        celery_stub = _mock_celery_task("mock-celery-id-001")

        with patch("app.worker.tasks.run_agent_pipeline_task") as mock_task:
            mock_task.delay.return_value = celery_stub

            with caplog.at_level(logging.INFO, logger="app.api.router"):
                response = await client.post(
                    "/v1/agent/execute",
                    json={"task": _VALID_TASK},
                )

        # ── HTTP contract ─────────────────────────────────────────────
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        body = response.json()
        assert "task_id" in body,   "Response must contain task_id"
        assert "thread_id" in body, "Response must contain thread_id"
        assert body.get("status") == "queued", "status must be 'queued'"

        # ── Logging contract ──────────────────────────────────────────
        assert "V1/EXECUTE queued" in caplog.text, (
            "Expected 'V1/EXECUTE queued' in log output.\n"
            f"Actual caplog:\n{caplog.text}"
        )

    # ── T902 ──────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_log_contains_route_and_task_preview(
        self,
        client: AsyncClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        T902 – The queued log entry must include:
          • The routing decision (new_session when no thread_id is given)
          • A recognisable snippet of the original task text
        """
        celery_stub = _mock_celery_task("mock-celery-id-002")

        with patch("app.worker.tasks.run_agent_pipeline_task") as mock_task:
            mock_task.delay.return_value = celery_stub

            with caplog.at_level(logging.INFO, logger="app.api.router"):
                response = await client.post(
                    "/v1/agent/execute",
                    json={"task": _VALID_TASK},
                )

        assert response.status_code == 200

        # No thread_id supplied → router always picks new_session
        assert "new_session" in caplog.text, (
            "Expected routing decision 'new_session' in log.\n"
            f"Actual caplog:\n{caplog.text}"
        )

        # At least the first 20 chars of the task must appear in the preview
        assert _VALID_TASK[:20] in caplog.text, (
            f"Task preview '{_VALID_TASK[:20]}' must be logged for traceability.\n"
            f"Actual caplog:\n{caplog.text}"
        )

    # ── T903 ──────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_log_contains_celery_task_id(
        self,
        client: AsyncClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        T903 – The Celery task_id generated at queuing time must appear
        verbatim in the INFO log so that operators can correlate requests
        in log aggregators (e.g. CloudWatch, Datadog).
        """
        unique_id = "trace-xyz-99999"
        celery_stub = _mock_celery_task(unique_id)

        with patch("app.worker.tasks.run_agent_pipeline_task") as mock_task:
            mock_task.delay.return_value = celery_stub

            with caplog.at_level(logging.INFO, logger="app.api.router"):
                response = await client.post(
                    "/v1/agent/execute",
                    json={"task": _VALID_TASK},
                )

        assert response.status_code == 200

        # The response body must echo the same task_id
        assert response.json().get("task_id") == unique_id, (
            "Response task_id must match the Celery stub id"
        )

        # The exact task_id must be present in the log for traceability
        assert unique_id in caplog.text, (
            f"Celery task_id '{unique_id}' must be emitted in the log.\n"
            f"Actual caplog:\n{caplog.text}"
        )
