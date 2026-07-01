"""
router.py — ENTRAPEER API thin layer.

This file contains ONLY transport logic.  No classification, no guardrails,
no keyword lists.  All routing decisions are made inside the LangGraph pipeline
by the Peer Agent node (app/agents/peer.py).

Endpoint responsibilities:
  POST /v1/agent/execute      – UNIFIED smart entry-point (no /api prefix)
                                auto-routes: new session → analyze flow
                                             continuation → respond flow
  POST /api/v1/analyze        – open a new analysis session (canonical)
  POST /api/v1/respond        – continue an existing session (canonical)
  POST /api/v1/agent/execute  – canonical execute (always new session)
  GET  /api/v1/history/:id    – fetch past analyses for a user
"""
import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import app.services.graph as graph_module
import app.services.mongodb as mongo
from app.agents.discovery import discovery_agent_node, discovery_synthesis_node
from app.agents.structuring import structuring_agent_node
from app.services.content_agent import generate_action_plan
from app.services.code_agent import generate_code_template
from app.services.agent_logger import log_agent_run

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])

# ---------------------------------------------------------------------------
# Shared response schemas
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str = Field(..., examples=["user", "assistant"])
    content: str


class AnalyzeResponse(BaseModel):
    thread_id: str
    messages: list[Message]
    structured_problem: dict[str, Any]
    current_step: str


# ---------------------------------------------------------------------------
# /respond answer-length validator
#
# Intentionally minimal — only checks that the user typed more than 2 words.
# All topic / intent decisions live in the Peer Agent node.
# ---------------------------------------------------------------------------
_INSUFFICIENT_MSG = (
    "Verdiğiniz yanıt, kriz senaryosunu analiz etmek ve problem ağacını oluşturmak "
    "için yeterli bilgi içermiyor. Lütfen sorduğum sorulara (özellikle organizasyon "
    "yapısı ve finansal göstergelerle ilgili olanlara) daha detaylı yanıtlar vermeye "
    "çalışın."
)


def _validate_response(text: str) -> tuple[bool, str]:
    """
    Returns (is_valid, rejection_reason).
    Rule: fewer than 3 words → rejected as too short.
    """
    word_count = len(text.split())
    if word_count < 3:
        return False, f"too_short ({word_count} words)"
    return True, ""


# ---------------------------------------------------------------------------
# Sub-agent runner — ContentAgent + CodeAgent after structuring completes
# ---------------------------------------------------------------------------
async def _run_sub_agents(
    *,
    structured: dict,
    thread_id: str,
    user_id: str | None,
    final_messages: list,
) -> None:
    """
    Run ContentAgent (always) and CodeAgent (Technology problems) after a
    ProblemTree is ready.  Errors are caught so the main response is unaffected.
    """
    t0 = time.monotonic()
    try:
        action_plan = await generate_action_plan(structured, thread_id=thread_id)
        duration_ms = int((time.monotonic() - t0) * 1000)
        await log_agent_run(
            agent_name="ContentAgent",
            thread_id=thread_id,
            user_id=user_id,
            input_data={"problem_tree": structured},
            output_data=action_plan.model_dump(),
            duration_ms=duration_ms,
        )
        logger.info("ContentAgent done thread_id=%s duration_ms=%d", thread_id, duration_ms)
    except Exception as exc:
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.warning("ContentAgent failed thread_id=%s: %s", thread_id, exc)
        await log_agent_run(
            agent_name="ContentAgent",
            thread_id=thread_id,
            user_id=user_id,
            input_data={"problem_tree": structured},
            output_data={},
            duration_ms=duration_ms,
            error=str(exc),
        )

    if structured.get("problem_type") in {"Technology", "Hybrid"}:
        t0 = time.monotonic()
        try:
            code_tmpl = await generate_code_template(structured, thread_id=thread_id)
            duration_ms = int((time.monotonic() - t0) * 1000)
            if code_tmpl:
                await log_agent_run(
                    agent_name="CodeAgent",
                    thread_id=thread_id,
                    user_id=user_id,
                    input_data={"problem_tree": structured},
                    output_data=code_tmpl.model_dump(),
                    duration_ms=duration_ms,
                )
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.warning("CodeAgent failed thread_id=%s: %s", thread_id, exc)
            await log_agent_run(
                agent_name="CodeAgent",
                thread_id=thread_id,
                user_id=user_id,
                input_data={"problem_tree": structured},
                output_data={},
                duration_ms=duration_ms,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# POST /api/v1/analyze
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Latest user message")
    thread_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    user_profile: dict[str, Any] = Field(default_factory=dict)
    history: list[Message] = Field(default_factory=list)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Open a new analysis session.

    The message is passed directly to the LangGraph pipeline without any
    pre-filtering.  The Peer Agent node handles all routing decisions
    (BUSINESS_CRISIS / DIRECT_ANSWER / CODE_REQUEST / OUT_OF_SCOPE).
    """
    thread_id: str = request.thread_id or str(uuid.uuid4())

    messages: list[dict] = [m.model_dump() for m in request.history]
    messages.append({"role": "user", "content": request.message})

    initial_state = {
        "messages": messages,
        "current_step": "start",
        "user_profile": request.user_profile,
        "structured_problem": {},
    }

    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        "ANALYZE – thread_id=%s user_id=%s preview=%r",
        thread_id, request.user_id, request.message[:80],
    )

    try:
        final_state = await graph_module.app_graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("Graph failed thread_id=%s: %s", thread_id, exc)
        raise HTTPException(status_code=500, detail="Agent pipeline failed.") from exc

    structured = final_state.get("structured_problem", {})

    if request.user_id and structured.get("main_problem"):
        try:
            await mongo.save_analysis(request.user_id, thread_id, structured)
        except Exception as exc:
            logger.warning("MongoDB save failed (non-fatal): %s", exc)

    return AnalyzeResponse(
        thread_id=thread_id,
        messages=[Message(**m) for m in final_state["messages"]],
        structured_problem=structured,
        current_step=final_state.get("current_step", "end"),
    )


# ---------------------------------------------------------------------------
# Checkpoint persistence helper
# ---------------------------------------------------------------------------
async def _save_checkpoint(config: dict, state: dict, label: str = "") -> None:
    """
    Write updated state back to the LangGraph checkpointer.

    Why this is necessary:
      /respond calls discovery_agent_node and structuring_agent_node DIRECTLY
      (without going through graph.ainvoke).  LangGraph only updates the
      checkpoint when a node runs inside ainvoke.  Calling nodes directly leaves
      the checkpoint with the PRE-call state, so the next /respond reads stale
      interview_turns and the interview never advances.

    This helper calls graph.aupdate_state so the checkpoint reflects the latest
    interview_turns, current_question, interview_history, and messages.
    """
    try:
        await graph_module.app_graph.aupdate_state(config, state)
        logger.info("CHECKPOINT saved label=%s", label)
    except Exception as exc:
        # Non-fatal: worst case the interview cannot advance past turn 1.
        # Log prominently so ops can detect stale-checkpoint issues.
        logger.warning("CHECKPOINT save failed (%s): %s", label, exc)


# ---------------------------------------------------------------------------
# POST /api/v1/respond
# ---------------------------------------------------------------------------
class RespondRequest(BaseModel):
    thread_id: str = Field(..., description="Session ID returned by /analyze")
    message: str = Field(..., min_length=1, description="User's answers to discovery questions")
    user_id: str | None = Field(default=None)


@router.post("/respond", response_model=AnalyzeResponse)
async def respond(request: RespondRequest) -> AnalyzeResponse:
    """
    Continue a session by submitting answers to the discovery questions.

    This endpoint is specifically for the BUSINESS_CRISIS → Discovery flow.
    It validates the answer length, then calls the Structuring Agent directly
    (Peer and Discovery do not re-run for mid-session answers).
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    # ── 1. Retrieve existing checkpoint state ─────────────────────────────
    try:
        snapshot = await graph_module.app_graph.aget_state(config)
    except Exception as exc:
        logger.exception("Checkpoint read failed thread_id=%s: %s", request.thread_id, exc)
        raise HTTPException(status_code=500, detail="Failed to read session state.") from exc

    if snapshot is None or not snapshot.values:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{request.thread_id}' not found. "
                "Start a new conversation with POST /analyze."
            ),
        )

    existing_state: dict = dict(snapshot.values)
    conversation_history: list[dict] = list(existing_state.get("messages", []))

    logger.info(
        "RESPOND – thread_id=%s msg_count=%d preview=%r",
        request.thread_id,
        len(conversation_history) + 1,
        request.message[:80],
    )

    # ── 2. Validate answer length ─────────────────────────────────────────
    is_valid, reason = _validate_response(request.message)
    if not is_valid:
        logger.info(
            "RESPOND – rejected insufficient answer thread_id=%s reason=%s",
            request.thread_id, reason,
        )
        return AnalyzeResponse(
            thread_id=request.thread_id,
            messages=[Message(**m) for m in conversation_history],
            structured_problem=existing_state.get("structured_problem", {}),
            current_step="awaiting_response",
        )

    # ── 3. Append user answer to messages ─────────────────────────────────
    messages: list[dict] = list(conversation_history)
    messages.append({"role": "user", "content": request.message})

    # interview_turns counts questions already ASKED by the Discovery Agent.
    # When the user is answering, interview_turns reflects how many questions
    # have been posed so far.
    #   turns < 4  → interview still ongoing → Discovery generates next question
    #   turns >= 4 → all 4 questions answered  → Structuring builds ProblemTree
    existing_turns: int = int(existing_state.get("interview_turns", 1))

    updated_state = {
        **existing_state,
        "messages": messages,
    }

    logger.info(
        "RESPOND – thread_id=%s interview_turns=%d route=%s",
        request.thread_id,
        existing_turns,
        "structuring" if existing_turns >= 4 else "discovery",
    )

    # ── 4a. Interview complete → Discovery Synthesis → Structuring Agent ─────
    if existing_turns >= 4:
        # Step 1: Discovery Agent synthesises the completed interview into the
        #         4 mandatory hand-off fields (DiscoverySummary).
        #         This MUST run before structuring so that customer_stated_problem,
        #         identified_business_problem, hidden_root_risk and
        #         customer_chat_summary are available to the Structuring Agent.
        updated_state["current_step"] = "synthesis"
        try:
            synthesis_state = await discovery_synthesis_node(updated_state)
        except Exception as exc:
            logger.exception(
                "Discovery synthesis failed thread_id=%s: %s", request.thread_id, exc
            )
            # Non-fatal: continue with empty summary rather than crashing
            synthesis_state = {
                **updated_state,
                "discovery_summary": {},
            }
            logger.warning(
                "RESPOND – discovery synthesis skipped, continuing without summary"
            )

        logger.info(
            "RESPOND – synthesis complete thread_id=%s fields=%s",
            request.thread_id,
            list(synthesis_state.get("discovery_summary", {}).keys()),
        )

        # Step 2: Structuring Agent builds the ProblemTree from full context
        #         (conversation + discovery_summary now embedded in state).
        synthesis_state["current_step"] = "structuring"
        try:
            final_state = await structuring_agent_node(synthesis_state)
        except Exception as exc:
            logger.exception("Structuring failed thread_id=%s: %s", request.thread_id, exc)
            raise HTTPException(status_code=500, detail="Structuring agent failed.") from exc

        # ── Persist completed state to checkpoint ─────────────────────────
        await _save_checkpoint(config, final_state, label="structuring")

        structured = final_state.get("structured_problem", {})

        if request.user_id and structured.get("main_problem"):
            try:
                await mongo.save_analysis(request.user_id, request.thread_id, structured)
            except Exception as exc:
                logger.warning("MongoDB save failed (non-fatal): %s", exc)

        if structured.get("main_problem"):
            await _run_sub_agents(
                structured=structured,
                thread_id=request.thread_id,
                user_id=request.user_id,
                final_messages=final_state["messages"],
            )

        return AnalyzeResponse(
            thread_id=request.thread_id,
            messages=[Message(**m) for m in final_state["messages"]],
            structured_problem=structured,
            current_step=final_state.get("current_step", "end"),
        )

    # ── 4b. Interview ongoing → Discovery Agent (next follow-up question) ──
    #
    # CRITICAL: after calling discovery_agent_node directly (bypassing graph.ainvoke)
    # the updated state (new interview_turns, current_question, interview_history) MUST
    # be written back to the checkpointer.  Without this, the next /respond call reads
    # the stale old state and the interview never advances past turn 2.
    updated_state["current_step"] = "discovery"
    try:
        final_state = await discovery_agent_node(updated_state)
    except Exception as exc:
        logger.exception("Discovery failed thread_id=%s: %s", request.thread_id, exc)
        raise HTTPException(status_code=500, detail="Discovery agent failed.") from exc

    # ── Persist new interview_turns + current_question to checkpoint ──────
    await _save_checkpoint(config, final_state, label="discovery")

    new_turns = final_state.get("interview_turns", existing_turns + 1)
    logger.info(
        "RESPOND – thread_id=%s interview_turns %d → %d question_preview=%r",
        request.thread_id,
        existing_turns,
        new_turns,
        final_state.get("current_question", "")[:60],
    )

    return AnalyzeResponse(
        thread_id=request.thread_id,
        messages=[Message(**m) for m in final_state["messages"]],
        structured_problem=final_state.get("structured_problem", {}),
        current_step=final_state.get("current_step", "awaiting_response"),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/agent/execute — unified entry-point (alias for /analyze)
# ---------------------------------------------------------------------------
class AgentExecuteRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=1,
        description="A natural-language task description.",
        examples=[
            "Fintech sektöründeki önemli oyuncuları analiz et.",
            "Satışlarımız son çeyrekte %30 düştü, neden olabilir?",
        ],
    )
    thread_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    user_profile: dict[str, Any] = Field(default_factory=dict)


class AgentExecuteResponse(BaseModel):
    thread_id: str
    task: str
    messages: list[Message]
    structured_problem: dict[str, Any]
    current_step: str
    status: str = Field(
        ...,
        description=(
            "'completed' – result ready; "
            "'awaiting_response' – discovery questions pending; "
            "'rejected' – out of scope."
        ),
    )


@router.post("/agent/execute", response_model=AgentExecuteResponse)
async def agent_execute(request: AgentExecuteRequest) -> AgentExecuteResponse:
    """
    Unified task entry-point.  Passes the task directly to the LangGraph
    pipeline — the Peer Agent decides routing.
    """
    thread_id: str = request.thread_id or str(uuid.uuid4())

    initial_state = {
        "messages": [{"role": "user", "content": request.task}],
        "current_step": "start",
        "user_profile": request.user_profile,
        "structured_problem": {},
    }

    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        "AGENT/EXECUTE – thread_id=%s user_id=%s task_preview=%r",
        thread_id, request.user_id, request.task[:80],
    )

    try:
        final_state = await graph_module.app_graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("agent/execute graph failed thread_id=%s: %s", thread_id, exc)
        raise HTTPException(status_code=500, detail="Agent pipeline failed.") from exc

    structured = final_state.get("structured_problem", {})
    current_step = final_state.get("current_step", "end")

    logger.info("AGENT/EXECUTE done thread_id=%s current_step=%s", thread_id, current_step)

    if request.user_id and structured.get("main_problem"):
        try:
            await mongo.save_analysis(request.user_id, thread_id, structured)
        except Exception as exc:
            logger.warning("MongoDB save failed (non-fatal): %s", exc)

    if structured.get("main_problem"):
        await _run_sub_agents(
            structured=structured,
            thread_id=thread_id,
            user_id=request.user_id,
            final_messages=final_state["messages"],
        )

    if current_step == "awaiting_response":
        status = "awaiting_response"
    elif current_step == "rejected":
        status = "rejected"
    else:
        status = "completed"

    return AgentExecuteResponse(
        thread_id=thread_id,
        task=request.task,
        messages=[Message(**m) for m in final_state["messages"]],
        structured_problem=structured,
        current_step=current_step,
        status=status,
    )


# ---------------------------------------------------------------------------
# POST /v1/agent/execute  +  GET /v1/agent/status/{task_id}
#
# Unified Queue-Backed Entry-Point  (no /api prefix)
# ─────────────────────────────────────────────────────────────────────────
#
#  POST /v1/agent/execute
#    1. Checks Redis checkpoint to decide routing (new_session / continuation)
#    2. Posts the payload to the Celery queue via run_agent_pipeline_task.delay()
#    3. Returns IMMEDIATELY: {"task_id": "...", "status": "queued", "thread_id": "..."}
#
#  GET /v1/agent/status/{task_id}
#    Polls the Celery AsyncResult.  When SUCCESS, returns the full agent result
#    (next discovery question OR final problem tree) inside the response.
#
#  Smart routing logic (evaluated synchronously before queuing)
#  ─────────────────────────────────────────────────────────────
#   thread_id absent                           → new_session
#   thread_id present, no active checkpoint    → new_session
#   thread_id present, current_step==
#     "awaiting_response"                      → continuation
# ---------------------------------------------------------------------------
v1_router = APIRouter(prefix="/v1", tags=["Unified Entry-Point (v1)"])


# ── Response schemas ──────────────────────────────────────────────────────
class QueuedResponse(BaseModel):
    """Immediate acknowledgement returned by POST /v1/agent/execute."""
    task_id: str = Field(..., description="Celery task ID — use with GET /v1/agent/status/{task_id}")
    status: str = Field(default="queued", description="Always 'queued' on initial submission.")
    thread_id: str = Field(..., description="Session ID — include in follow-up requests.")


class TaskStatusResponse(BaseModel):
    """Response returned by GET /v1/agent/status/{task_id}."""
    task_id: str
    status: str = Field(
        ...,
        description=(
            "Celery state: PENDING | STARTED | SUCCESS | FAILURE | RETRY. "
            "When SUCCESS, 'result' contains the full agent response."
        ),
    )
    result: AgentExecuteResponse | None = Field(
        default=None,
        description="Populated when status == SUCCESS.",
    )
    error: str | None = Field(
        default=None,
        description="Populated when status == FAILURE.",
    )


# ── Session check helper ──────────────────────────────────────────────────
async def _has_active_session(thread_id: str) -> bool:
    """
    Return True iff the given thread_id has a Redis/MemorySaver checkpoint
    whose current_step is "awaiting_response" (interview in progress).

    Any exception is silently swallowed — treated as "no active session"
    so callers are never 500'd by a transient checkpointer error.
    """
    try:
        snapshot = await graph_module.app_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        if not snapshot or not snapshot.values:
            return False
        return snapshot.values.get("current_step") == "awaiting_response"
    except Exception as exc:
        logger.debug("_has_active_session check failed (non-fatal): %s", exc)
        return False


# ── POST /v1/agent/execute ────────────────────────────────────────────────
@v1_router.post(
    "/agent/execute",
    response_model=QueuedResponse,
    summary="Queue a new or ongoing agent pipeline task",
)
async def v1_unified_execute(request: AgentExecuteRequest) -> QueuedResponse:
    """
    **Unified Queue-Backed Entry-Point** — `POST /v1/agent/execute`

    Submit a single payload for the *entire* conversation lifecycle.
    The endpoint returns **immediately** with a `task_id`; the actual
    agent work runs asynchronously in a Celery worker.

    ```json
    { "task": "...", "thread_id": "(optional)", "user_id": "(optional)" }
    ```

    **Routing (decided server-side before queuing)**

    | Condition | Route |
    |---|---|
    | `thread_id` absent | new session |
    | `thread_id` present, no active checkpoint | new session |
    | `thread_id` present, checkpoint `awaiting_response` | continuation |

    Poll `GET /v1/agent/status/{task_id}` to retrieve the result.
    When `status == SUCCESS`, the response contains the next discovery
    question *or* the final `structured_problem` (ProblemTree).
    """
    from app.worker.tasks import run_agent_pipeline_task

    # Determine thread_id (preserve caller-supplied or generate new)
    thread_id: str = request.thread_id or str(uuid.uuid4())

    # Check Redis checkpoint to decide route (only if caller supplied thread_id)
    is_continuation = bool(
        request.thread_id
        and await _has_active_session(request.thread_id)
    )
    route = "continuation" if is_continuation else "new_session"

    payload = {
        "route": route,
        "thread_id": thread_id,
        "task": request.task,
        "user_id": request.user_id,
        "user_profile": request.user_profile or {},
    }

    celery_task = run_agent_pipeline_task.delay(payload)

    logger.info(
        "V1/EXECUTE queued – task_id=%s thread_id=%s route=%s task_preview=%r",
        celery_task.id, thread_id, route, request.task[:80],
    )

    return QueuedResponse(
        task_id=celery_task.id,
        status="queued",
        thread_id=thread_id,
    )


# ── GET /v1/agent/status/{task_id} ───────────────────────────────────────
@v1_router.get(
    "/agent/status/{task_id}",
    response_model=TaskStatusResponse,
    summary="Poll the status of a queued pipeline task",
)
async def v1_agent_status(task_id: str) -> TaskStatusResponse:
    """
    **Task Status Poll** — `GET /v1/agent/status/{task_id}`

    Check the current state of a previously queued pipeline task.

    | `status` value | Meaning |
    |---|---|
    | `PENDING` | Waiting in the queue |
    | `STARTED` | Worker has picked it up |
    | `SUCCESS` | Completed — `result` field is populated |
    | `FAILURE` | Failed — `error` field is populated |
    | `RETRY`   | Worker is retrying after a transient error |

    When `result.status == "awaiting_response"`, send the user's answer
    back via another `POST /v1/agent/execute` with the same `thread_id`.

    When `result.status == "completed"`, the `result.structured_problem`
    contains the final ProblemTree (also persisted to MongoDB).
    """
    from celery.result import AsyncResult
    from app.worker.celery_app import celery_app as _celery

    async_result = AsyncResult(task_id, app=_celery)
    state: str = async_result.state  # PENDING / STARTED / SUCCESS / FAILURE / RETRY

    if state == "SUCCESS":
        raw: dict = async_result.result
        agent_result = AgentExecuteResponse(
            thread_id=raw["thread_id"],
            task=raw["task"],
            messages=[Message(**m) for m in raw["messages"]],
            structured_problem=raw["structured_problem"],
            current_step=raw["current_step"],
            status=raw["status"],
        )
        return TaskStatusResponse(task_id=task_id, status="SUCCESS", result=agent_result)

    if state == "FAILURE":
        err = async_result.result
        return TaskStatusResponse(
            task_id=task_id,
            status="FAILURE",
            error=str(err),
        )

    # PENDING, STARTED, RETRY — result not ready yet
    return TaskStatusResponse(task_id=task_id, status=state)


# ---------------------------------------------------------------------------
# GET /api/v1/history/{user_id}
# ---------------------------------------------------------------------------
class AnalysisRecord(BaseModel):
    thread_id: str
    created_at: str
    structured_problem: dict[str, Any]


class HistoryResponse(BaseModel):
    user_id: str
    count: int
    analyses: list[AnalysisRecord]


@router.get("/history/{user_id}", response_model=HistoryResponse)
async def get_history(user_id: str) -> HistoryResponse:
    """Return all completed analysis reports for a given user, newest first."""
    try:
        docs = await mongo.get_analyses(user_id)
    except Exception as exc:
        logger.exception("MongoDB read failed user_id=%s: %s", user_id, exc)
        raise HTTPException(status_code=500, detail="Failed to read analysis history.") from exc

    analyses = [
        AnalysisRecord(
            thread_id=doc.get("thread_id", ""),
            created_at=doc.get("created_at", ""),
            structured_problem=doc.get("structured_problem", {}),
        )
        for doc in docs
    ]

    logger.info("HISTORY – user_id=%s count=%d", user_id, len(analyses))
    return HistoryResponse(user_id=user_id, count=len(analyses), analyses=analyses)


# ---------------------------------------------------------------------------
# Universal catch-all alias router
# ---------------------------------------------------------------------------
alias_router = APIRouter(tags=["analysis"])


@alias_router.post("/v1/agent/analyze", response_model=AnalyzeResponse)
async def _alias_v1_agent_analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return await analyze(request)


@alias_router.post("/v1/agent/respond", response_model=AnalyzeResponse)
async def _alias_v1_agent_respond(request: RespondRequest) -> AnalyzeResponse:
    return await respond(request)


@alias_router.post("/analyze", response_model=AnalyzeResponse)
async def _alias_analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return await analyze(request)


@alias_router.post("/respond", response_model=AnalyzeResponse)
async def _alias_respond(request: RespondRequest) -> AnalyzeResponse:
    return await respond(request)


@alias_router.post("/api/analyze", response_model=AnalyzeResponse)
async def _alias_api_analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return await analyze(request)


@alias_router.post("/api/respond", response_model=AnalyzeResponse)
async def _alias_api_respond(request: RespondRequest) -> AnalyzeResponse:
    return await respond(request)
