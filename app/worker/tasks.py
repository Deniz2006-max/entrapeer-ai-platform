"""
tasks.py — Celery task definitions for ENTRAPEER.

Single exported task
────────────────────
  run_agent_pipeline_task(payload)

    Executes the full LangGraph agent pipeline in a background worker process.
    Accepts a JSON-serialisable payload dict and returns a JSON-serialisable
    result dict that mirrors AgentExecuteResponse.

Async inside Celery
───────────────────
    Celery workers are sync by default.  We bridge the gap by calling
    asyncio.run() once per task invocation.  Each task creates its own
    AsyncRedisSaver connection so that the LangGraph checkpoint store
    (shared with the FastAPI process via Redis) is accessible and consistent.

MongoDB
───────
    Initialised lazily inside the worker process on first use via
    _ensure_mongo().  The motor client is cheap to construct and safe to
    share across sequential tasks in the same worker process.
"""
import asyncio
import logging
import os

from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_mongo() -> None:
    """Lazily initialise the MongoDB motor client inside the worker process."""
    import app.services.mongodb as mongo
    if mongo._client is None:
        mongo.init_mongo()


async def _build_graph():
    """
    Build a LangGraph CompiledGraph with AsyncRedisSaver for this task.

    Returns (graph, checkpointer_ctx) — the caller is responsible for
    calling ``await checkpointer_ctx.__aexit__(None, None, None)`` when done.

    Falls back to MemorySaver when Redis is unavailable so unit tests and
    local runs without Redis still work (state won't persist between tasks).
    """
    from app.services.graph import _build_workflow

    try:
        from langgraph.checkpoint.redis import AsyncRedisSaver

        ctx = AsyncRedisSaver.from_conn_string(REDIS_URL)
        checkpointer = await ctx.__aenter__()
        await checkpointer.asetup()
        graph = _build_workflow().compile(checkpointer=checkpointer)
        return graph, ctx
    except Exception as exc:
        logger.warning(
            "AsyncRedisSaver unavailable (%s) – falling back to MemorySaver.", exc
        )
        from langgraph.checkpoint.memory import MemorySaver

        graph = _build_workflow().compile(checkpointer=MemorySaver())
        return graph, None


def _derive_status(current_step: str) -> str:
    if current_step == "awaiting_response":
        return "awaiting_response"
    if current_step == "rejected":
        return "rejected"
    return "completed"


# ---------------------------------------------------------------------------
# Context-switch detection
# ---------------------------------------------------------------------------

# Full AgentState reset template — all interview fields zeroed out.
# Used by _execute_pipeline when a context switch is detected on a completed
# thread so that LangGraph starts with a clean slate instead of merging on
# top of the old interview_turns / interview_history.
_CLEAN_STATE: dict = {
    "messages": [],
    "current_step": "start",
    "user_profile": {},
    "structured_problem": {},
    "interview_turns": 0,
    "current_question": "",
    "interview_history": [],
    "discovery_summary": {},
    "pending_new_crisis": "",
}

# Guardrail message patterns to strip from state.
# Covers: Peer Agent OUT_OF_SCOPE (REJECTION_MSG), off-topic warning, and
# context-switch confirmation prompts.
_GUARDRAIL_PATTERNS: tuple[str, ...] = (
    "Bu talep **ENTRAPEER**",           # peer.py REJECTION_MSG
    "Verdiğiniz yanıt platformumuzun",  # tasks.py off-topic (Scenario 2)
    "konusundan ayrı bir konu olan",    # tasks.py context-switch confirmation (Scenario 1)
)


def _strip_guardrail_messages(messages: list[dict]) -> list[dict]:
    """Return a copy of *messages* with all guardrail/warning entries removed."""
    return [
        m for m in messages
        if not any(pat in m.get("content", "") for pat in _GUARDRAIL_PATTERNS)
    ]

_CONTEXT_SWITCH_SYSTEM = """\
You are a topic classification assistant for a business crisis analysis platform.

Your task: decide whether two business problem descriptions refer to the SAME
underlying corporate issue or to COMPLETELY DIFFERENT topics.

Respond with EXACTLY ONE WORD — nothing else:
  SAME        → same company department, same root theme, follow-up / deeper dive
  DIFFERENT   → different business domain, different problem category, unrelated issues

Examples
─────────────────────────────────────────────────────────────────
Previous: "Software subscription costs exceeded budget"
New:      "Raw material shortage halted production"
Answer:   DIFFERENT

Previous: "Sales team missed Q3 targets"
New:      "Can you elaborate on how the CRM gap affects pipeline?"
Answer:   SAME
─────────────────────────────────────────────────────────────────
"""


async def _detect_context_switch(
    new_task: str,
    prev_structured: dict,
) -> bool:
    """
    Return True when ``new_task`` is about a significantly different business
    problem than the previous completed analysis.

    Uses a fast gpt-4o-mini call (temperature=0) for binary classification.
    Falls back to True (force reset) on any error so the UX is never broken.

    Bypassed (always True) in MOCK_MODE to avoid real LLM calls during tests.
    """
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        # In mock mode we can't call OpenAI — always treat as new context.
        return True

    prev_summary: dict = prev_structured.get("discovery_summary", {})
    prev_context: str = (
        prev_summary.get("customer_stated_problem")
        or prev_structured.get("main_problem")
        or ""
    ).strip()

    if not prev_context:
        # No usable previous context — treat as new session.
        return True

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        human_text = (
            f"Previous analysis topic: {prev_context}\n"
            f"New user message: {new_task}\n\n"
            "Are these about the SAME or DIFFERENT business problem?"
        )
        response = await llm.ainvoke([
            SystemMessage(content=_CONTEXT_SWITCH_SYSTEM),
            HumanMessage(content=human_text),
        ])
        verdict = response.content.strip().upper()
        is_different = "DIFFERENT" in verdict
        logger.info(
            "CONTEXT-SWITCH check verdict=%r is_different=%s prev_context=%.60r new_task=%.60r",
            verdict, is_different, prev_context, new_task,
        )
        return is_different
    except Exception as exc:
        logger.warning("CONTEXT-SWITCH LLM check failed (%s) – defaulting to reset.", exc)
        return True


# ---------------------------------------------------------------------------
# Interview input classifier — 3-way (ANSWER / NEW_CRISIS / OFF_TOPIC)
# ---------------------------------------------------------------------------

_INTERVIEW_INPUT_CLASSIFIER_SYSTEM = """\
You are a classification assistant for a business crisis analysis platform.

During a structured interview about a specific corporate crisis, classify
the user's latest message into EXACTLY ONE of three categories:

  ANSWER      → User is answering (directly or indirectly) the current
                interview question. Even vague, short, or partial attempts
                to address the question count as ANSWER.

  NEW_CRISIS  → User is describing a COMPLETELY DIFFERENT corporate crisis
                or business problem with no connection to the ongoing topic.
                The user appears to be starting a fresh problem, not expanding
                or clarifying the current one.

  OFF_TOPIC   → Message has zero connection to business or the interview.
                (personal chat, food, hobbies, greetings, random unrelated text)

Rules:
  • When in doubt between ANSWER and NEW_CRISIS → choose ANSWER.
  • Only use NEW_CRISIS when the new problem is clearly in a different
    business domain or department from the current crisis.
  • Only use OFF_TOPIC for clearly non-business content.
  • Respond with EXACTLY ONE WORD: ANSWER, NEW_CRISIS, or OFF_TOPIC.

Examples
─────────────────────────────────────────────────────────────────
Current crisis: "Digital sales dropped 40%"
Current question: "Which channel is most affected?"
User: "E-commerce down, social media fine"
→ ANSWER

Current crisis: "Digital sales dropped 40%"
Current question: "Which channel is most affected?"
User: "Tedarik zincirimiz çöktü, hammadde bulamıyoruz"
→ NEW_CRISIS

Current crisis: "Digital sales dropped 40%"
Current question: "Which channel is most affected?"
User: "havaalanına gidiyorum bugün"
→ OFF_TOPIC
─────────────────────────────────────────────────────────────────
"""

# MOCK_MODE keyword lists for rule-based classification (no LLM)
_MOCK_OOS_WORDS: tuple[str, ...] = (
    "pizza", "dondurma", "kebap", "burger", "köfte", "tarifi",
    "nasılsın", "naber", "günaydın", "film öner", "hava durumu",
    "futbol", "spor haberi", "netflix", "havaalanı", "havalimanı",
    "tatil", "yiyeceğim", "yiyorum", "gidiyorum",
)
_MOCK_CRISIS_WORDS: tuple[str, ...] = (
    "şirketimiz", "satışlarımız", "ekibimiz", "bütçemiz", "maliyetimiz",
    "tedarik", "hammadde", "personel", "devir oranı", "churn",
)


async def _classify_interview_input(
    user_input: str,
    crisis_context: str,
    current_question: str,
) -> str:
    """
    Return "ANSWER", "NEW_CRISIS", or "OFF_TOPIC".

    Uses gpt-4o-mini (temperature=0).  Defaults to "ANSWER" on error.
    MOCK_MODE uses lightweight keyword heuristics (no LLM).
    """
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        lower = user_input.lower()
        if any(w in lower for w in _MOCK_OOS_WORDS):
            return "OFF_TOPIC"
        return "ANSWER"  # Never trigger NEW_CRISIS in tests

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        human_text = (
            f"Current crisis context: {crisis_context}\n"
            f"Current interview question: {current_question}\n"
            f"User message: {user_input}\n\n"
            "Classify: ANSWER, NEW_CRISIS, or OFF_TOPIC?"
        )
        response = await llm.ainvoke([
            SystemMessage(content=_INTERVIEW_INPUT_CLASSIFIER_SYSTEM),
            HumanMessage(content=human_text),
        ])
        verdict = response.content.strip().upper()
        for label in ("ANSWER", "NEW_CRISIS", "OFF_TOPIC"):
            if label in verdict:
                logger.info(
                    "INTERVIEW-CLASS verdict=%r label=%s input=%.60r",
                    verdict, label, user_input,
                )
                return label
        logger.warning("Unexpected classification %r — defaulting to ANSWER", verdict)
        return "ANSWER"
    except Exception as exc:
        logger.warning("INTERVIEW-CLASS failed (%s) — defaulting to ANSWER", exc)
        return "ANSWER"  # Fail open — never accidentally block a valid answer


def _is_affirmative(text: str) -> bool:
    """Return True when the user's text signals agreement / confirmation."""
    lower = text.lower().strip()
    return any(k in lower for k in (
        "evet", "yes", "tamam", "kabul", "devam", "geç", "geçelim",
        "yeni", "onayla", "onaylıyorum", "olur", "başlat", "switch",
    ))


# ---------------------------------------------------------------------------
# Core async pipeline logic
# ---------------------------------------------------------------------------

async def _execute_pipeline(payload: dict) -> dict:
    """
    Run the agent pipeline for a single task/turn.

    payload keys
    ─────────────
      route        : "new_session" | "continuation"
      thread_id    : str
      task         : str   — initial crisis text OR the user's discovery answer
      user_id      : str | None
      user_profile : dict
    """
    from app.agents.discovery import discovery_agent_node, discovery_synthesis_node
    from app.agents.structuring import structuring_agent_node
    import app.services.mongodb as mongo

    route: str = payload["route"]
    thread_id: str = payload["thread_id"]
    task_text: str = payload["task"]
    user_id: str | None = payload.get("user_id")
    user_profile: dict = payload.get("user_profile") or {}

    graph, checkpointer_ctx = await _build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # ── Path A: New session ───────────────────────────────────────────
        if route == "new_session":
            # ── Context-Switch Guard ──────────────────────────────────────
            # When the previous session for this thread_id has already
            # completed (current_step == "end"), we must decide whether to
            # wipe the old state before starting fresh.
            #
            # Why this matters: LangGraph merges ainvoke's input onto the
            # existing checkpoint.  Fields absent from initial_state
            # (interview_turns, interview_history, …) keep their old values,
            # causing the new analysis to inherit stale interview state.
            #
            # Decision logic:
            #   • previous session ended  AND topic is DIFFERENT  → reset
            #   • previous session ended  AND topic is SAME       → carry on
            #     (user may be asking follow-up on the finished analysis)
            #   • no previous checkpoint (new thread)             → skip check
            try:
                prev_snapshot = await graph.aget_state(config)
            except Exception:
                prev_snapshot = None

            if prev_snapshot and prev_snapshot.values:
                prev_step = prev_snapshot.values.get("current_step", "")

                # ── Context-switch detection (observability only) ─────────────
                # Only meaningful when a full analysis previously completed.
                # The result is LOGGED but does NOT change the reset decision —
                # we always wipe the checkpoint for every new session so that
                # DIRECT_ANSWER search results, OUT_OF_SCOPE rejection notices,
                # and stale Q-A pairs can NEVER contaminate the new interview.
                #
                # Without this unconditional reset, discovery_agent_node's
                # _extract_problem_context() would pick the *first* user message
                # in history (e.g. an old search query) as the crisis context,
                # causing the LLM to hallucinate by mixing two unrelated topics.
                if prev_step == "end":
                    prev_structured = prev_snapshot.values.get("structured_problem", {})
                    is_switch = await _detect_context_switch(task_text, prev_structured)
                    logger.info(
                        "TASK – context-switch check: %s thread_id=%s",
                        "DIFFERENT" if is_switch else "SAME (clean slate anyway)",
                        thread_id,
                    )

                # Always wipe — covers: "end", "completed", "rejected",
                # "discovery", "synthesis", "structuring", or any partial state.
                clean = {**_CLEAN_STATE, "user_profile": user_profile}
                await graph.aupdate_state(config, clean)
                logger.info(
                    "TASK – checkpoint wiped for clean new session "
                    "thread_id=%s prev_step=%r",
                    thread_id, prev_step,
                )

            # Always include all resettable fields in initial_state so they
            # override any checkpoint residue even if no explicit reset ran.
            initial_state = {
                "messages": [{"role": "user", "content": task_text}],
                "current_step": "start",
                "user_profile": user_profile,
                "structured_problem": {},
                "interview_turns": 0,
                "current_question": "",
                "interview_history": [],
                "discovery_summary": {},
                "pending_new_crisis": "",
            }
            final_state = await graph.ainvoke(initial_state, config=config)

        # ── Path B: Continuation — task is the user's discovery answer ────
        else:
            snapshot = await graph.aget_state(config)
            if not snapshot or not snapshot.values:
                raise ValueError(
                    f"No active checkpoint found for thread_id={thread_id!r}. "
                    "Cannot continue a session that does not exist."
                )

            existing_state: dict = dict(snapshot.values)
            existing_turns: int = int(existing_state.get("interview_turns", 1))

            # Validate answer length (mirrors /respond guardrail)
            word_count = len(task_text.split())
            if word_count < 3:
                logger.info(
                    "TASK – answer too short (%d words) thread_id=%s",
                    word_count, thread_id,
                )
                current_step = existing_state.get("current_step", "awaiting_response")
                return {
                    "thread_id": thread_id,
                    "task": task_text,
                    "messages": existing_state.get("messages", []),
                    "structured_problem": existing_state.get("structured_problem", {}),
                    "current_step": current_step,
                    "status": "awaiting_response",
                }

            # ── Shared helpers for guards below ───────────────────────────────
            # Derive crisis context (first user message = original crisis text)
            crisis_context: str = ""
            for m in existing_state.get("messages", []):
                if isinstance(m, dict) and m.get("role") == "user":
                    crisis_context = m.get("content", "")
                    break
            current_question: str = existing_state.get("current_question", "")

            def _prior_msgs() -> list[dict]:
                return [
                    {"role": m.get("role", ""), "content": m.get("content", "")}
                    for m in existing_state.get("messages", [])
                ]

            # ── SCENARIO 1a: Pending context-switch confirmation ───────────────
            # User previously proposed a new crisis; we asked for confirmation.
            # Now we handle their "Evet / Hayır" response.
            pending_crisis: str = existing_state.get("pending_new_crisis", "")
            if pending_crisis:
                if _is_affirmative(task_text):
                    # ── Confirmed → wipe state, start new interview ───────────
                    logger.info(
                        "TASK – context-switch CONFIRMED, starting new interview "
                        "thread_id=%s pending_crisis=%.60r",
                        thread_id, pending_crisis,
                    )
                    clean = {**_CLEAN_STATE, "user_profile": user_profile}
                    await graph.aupdate_state(config, clean)

                    new_initial_state = {
                        **_CLEAN_STATE,
                        "messages": [{"role": "user", "content": pending_crisis}],
                        "user_profile": user_profile,
                    }
                    final_state = await graph.ainvoke(new_initial_state, config=config)

                else:
                    # ── Declined → clear pending flag, re-ask current Q ───────
                    logger.info(
                        "TASK – context-switch DECLINED, resuming interview "
                        "thread_id=%s turns=%d",
                        thread_id, existing_turns,
                    )
                    await graph.aupdate_state(config, {"pending_new_crisis": ""})
                    resume_msg = (
                        "Anlaşıldı, mevcut analizimize kaldığımız yerden devam edelim.\n\n"
                        f"{current_question}"
                    )
                    return {
                        "thread_id": thread_id,
                        "task": task_text,
                        "messages": [*_prior_msgs(), {"role": "assistant", "content": resume_msg}],
                        "structured_problem": existing_state.get("structured_problem", {}),
                        "current_step": "awaiting_response",
                        "status": "awaiting_response",
                    }

            # ── SCENARIO 1b + 2: Input classification (interview turns < 4) ───
            # Skip classification when all 4 turns are done (synthesis path).
            elif existing_turns < 4 and not pending_crisis:
                input_class = await _classify_interview_input(
                    task_text, crisis_context, current_question
                )

                # ── SCENARIO 1b: New crisis detected mid-interview ────────────
                if input_class == "NEW_CRISIS":
                    crisis_title = (
                        crisis_context[:70] + "…"
                        if len(crisis_context) > 70
                        else crisis_context
                    )
                    new_title = (
                        task_text[:70] + "…"
                        if len(task_text) > 70
                        else task_text
                    )
                    confirm_msg = (
                        f"Şu anda aktif olarak tartıştığımız **\"{crisis_title}\"** "
                        f"konusundan ayrı bir konu olan **\"{new_title}\"** hakkında "
                        "fikir yürütmeye geçmek üzeresiniz.\n\n"
                        "Bu yeni krize geçmek istediğinizden emin misiniz?\n\n"
                        "✅ **Evet** → Mevcut mülakatı kapatır, yeni kriz analizine başlarız.\n"
                        "❌ **Hayır** → Mevcut mülakata kaldığımız yerden devam ederiz."
                    )
                    logger.info(
                        "TASK – NEW_CRISIS detected mid-interview, requesting confirmation "
                        "thread_id=%s turns=%d new_crisis=%.60r",
                        thread_id, existing_turns, task_text,
                    )
                    # Persist new crisis text as pending (do NOT increment turns)
                    await graph.aupdate_state(config, {"pending_new_crisis": task_text})
                    return {
                        "thread_id": thread_id,
                        "task": task_text,
                        "messages": [*_prior_msgs(), {"role": "assistant", "content": confirm_msg}],
                        "structured_problem": existing_state.get("structured_problem", {}),
                        "current_step": "awaiting_response",
                        "status": "awaiting_response",
                    }

                # ── SCENARIO 2: Off-topic / out-of-scope response ─────────────
                elif input_class == "OFF_TOPIC":
                    off_topic_msg = (
                        "Verdiğiniz yanıt platformumuzun iş analizi kapsamı dışındadır. "
                        "Mülakatımızın bölünmemesi için şu sorumuza odaklanabilir miyiz:\n\n"
                        f"{current_question}"
                    )
                    logger.info(
                        "TASK – OFF_TOPIC input, re-asking Q without incrementing turns "
                        "thread_id=%s turns=%d input=%.60r",
                        thread_id, existing_turns, task_text,
                    )
                    return {
                        "thread_id": thread_id,
                        "task": task_text,
                        "messages": [*_prior_msgs(), {"role": "assistant", "content": off_topic_msg}],
                        "structured_problem": existing_state.get("structured_problem", {}),
                        "current_step": "awaiting_response",
                        "status": "awaiting_response",
                    }

            # Append user's answer to messages.
            # Strip stale guardrail / warning messages first so they never
            # accumulate in Redis and never appear in the final report.
            messages: list[dict] = _strip_guardrail_messages(
                list(existing_state.get("messages", []))
            )
            messages.append({"role": "user", "content": task_text})

            updated_state = {
                **existing_state,
                "messages": messages,
            }

            logger.info(
                "TASK – thread_id=%s interview_turns=%d route=%s",
                thread_id,
                existing_turns,
                "structuring" if existing_turns >= 4 else "discovery",
            )

            # ── All 4 questions answered → Synthesis → Structuring ────────
            if existing_turns >= 4:
                updated_state["current_step"] = "synthesis"
                try:
                    synthesis_state = await discovery_synthesis_node(updated_state)
                except Exception as exc:
                    logger.exception(
                        "Discovery synthesis failed thread_id=%s: %s", thread_id, exc
                    )
                    synthesis_state = {**updated_state, "discovery_summary": {}}

                synthesis_state["current_step"] = "structuring"
                final_state = await structuring_agent_node(synthesis_state)

                # Persist to MongoDB if user_id present
                structured = final_state.get("structured_problem", {})
                if user_id and structured.get("main_problem"):
                    _ensure_mongo()
                    try:
                        await mongo.save_analysis(user_id, thread_id, structured)
                    except Exception as exc:
                        logger.warning("MongoDB save failed (non-fatal): %s", exc)

            # ── Interview still ongoing → next Discovery question ─────────
            else:
                updated_state["current_step"] = "discovery"
                final_state = await discovery_agent_node(updated_state)

            # Persist updated checkpoint back to Redis
            try:
                await graph.aupdate_state(config, final_state)
                logger.info("TASK – checkpoint saved thread_id=%s", thread_id)
            except Exception as exc:
                logger.warning("TASK – checkpoint save failed: %s", exc)

        # ── Build result dict ─────────────────────────────────────────────
        current_step = final_state.get("current_step", "end")
        structured = final_state.get("structured_problem", {})

        # MongoDB save for new_session path (only when structuring completed)
        if route == "new_session" and user_id and structured.get("main_problem"):
            _ensure_mongo()
            try:
                await mongo.save_analysis(user_id, thread_id, structured)
            except Exception as exc:
                logger.warning("MongoDB save failed (non-fatal): %s", exc)

        # Strip any residual guardrail messages before sending to the client.
        # This is the last safety net: even if a REJECTION_MSG somehow survived
        # into final_state, it will not reach the frontend.
        clean_messages = [
            {"role": m.get("role", ""), "content": m.get("content", "")}
            for m in _strip_guardrail_messages(list(final_state.get("messages", [])))
        ]

        return {
            "thread_id": thread_id,
            "task": task_text,
            "messages": clean_messages,
            "structured_problem": structured,
            "current_step": current_step,
            "status": _derive_status(current_step),
        }

    finally:
        if checkpointer_ctx is not None:
            try:
                await checkpointer_ctx.__aexit__(None, None, None)
            except Exception as exc:
                logger.debug("Redis checkpointer close error: %s", exc)


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="run_agent_pipeline",
    max_retries=0,          # Do not auto-retry; let the caller re-submit
    acks_late=True,         # Ack only after the task finishes (crash safety)
    reject_on_worker_lost=True,
)
def run_agent_pipeline_task(self, payload: dict) -> dict:
    """
    Execute the full ENTRAPEER agent pipeline as a background Celery task.

    Expected payload
    ─────────────────
    {
        "route":        "new_session" | "continuation",
        "thread_id":    "<uuid>",
        "task":         "<user message>",
        "user_id":      "<optional>",
        "user_profile": {}
    }

    Returns a JSON-serialisable dict matching AgentExecuteResponse fields:
    {
        "thread_id":          "...",
        "task":               "...",
        "messages":           [...],
        "structured_problem": {...},
        "current_step":       "...",
        "status":             "awaiting_response | completed | rejected"
    }
    """
    logger.info(
        "CELERY TASK START – route=%s thread_id=%s",
        payload.get("route"), payload.get("thread_id"),
    )
    try:
        result = asyncio.run(_execute_pipeline(payload))
        logger.info(
            "CELERY TASK DONE – thread_id=%s status=%s",
            result.get("thread_id"), result.get("status"),
        )
        return result
    except Exception as exc:
        logger.exception("CELERY TASK FAILED – thread_id=%s: %s", payload.get("thread_id"), exc)
        raise
