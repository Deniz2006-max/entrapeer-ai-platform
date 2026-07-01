"""
test_agent.py – Integration & unit tests for the ENTRAPEER agent API.

Test categories:
  T1xx – /health              (smoke tests)
  T2xx – /api/v1/agent/execute (happy path + guardrail)
  T3xx – /api/v1/analyze       (happy path)
  T4xx – /api/v1/respond       (guardrail / validation)
  T5xx – Input validation      (schema & boundary cases)
  T6xx – Code-request routing
  T7xx – SystemGuardrail rejection
  T8xx – Edge cases & boundary conditions

All external I/O is mocked via the `client` fixture defined in conftest.py.
Real LLM/DB/Redis calls never happen; MOCK_MODE=true keeps agent nodes fast.
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from tests.conftest import DISCOVERY_STATE, MOCK_ACTION_PLAN, MOCK_SNAPSHOT, STRUCTURED_STATE

# ===========================================================================
# T1xx – Smoke / health
# ===========================================================================

class TestHealth:
    """Basic connectivity and server health checks."""

    async def test_health_returns_200(self, client: AsyncClient):
        """GET /health must return HTTP 200 and a valid status payload."""
        response = await client.get("/health")
        assert response.status_code == 200

    async def test_health_payload_structure(self, client: AsyncClient):
        """Response body must contain 'status' and 'checkpointer' keys."""
        data = (await client.get("/health")).json()
        assert data["status"] == "ok"
        assert "checkpointer" in data


# ===========================================================================
# T2xx – POST /api/v1/agent/execute
# ===========================================================================

class TestAgentExecute:
    """
    Tests for the unified agent task entry-point.
    """

    # ── T201 ─────────────────────────────────────────────────────────────────
    async def test_valid_task_returns_200(self, client: AsyncClient):
        """
        Happy Path #1 – Valid business task execution.

        Verifies that a legitimate crisis description:
          - returns HTTP 200
          - sets `status` to either "awaiting_response" or "completed"
          - echoes back the original task string
          - returns a non-empty `thread_id`
        """
        payload = {"task": "Bulut altyapı maliyetlerimiz %50 arttı"}

        response = await client.post("/api/v1/agent/execute", json=payload)

        assert response.status_code == 200, response.text

        data = response.json()
        assert data["status"] in ("awaiting_response", "completed"), (
            f"Unexpected status: {data['status']!r}"
        )
        assert data["task"] == payload["task"]
        assert data["thread_id"]  # must be non-empty string
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) >= 1

    # ── T202 ─────────────────────────────────────────────────────────────────
    async def test_valid_task_discovery_flow(self, client: AsyncClient):
        """
        Happy Path #1b – Discovery flow check.

        When the graph returns `current_step = "awaiting_response"` (Discovery),
        the API must surface that as status="awaiting_response" so the frontend
        knows to render the question-answer UI.
        """
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "Satışlarımız son çeyrekte ciddi düştü, nedenini anlayamıyoruz"},
        )
        data = response.json()

        assert response.status_code == 200
        assert data["current_step"] == "awaiting_response"
        assert data["status"] == "awaiting_response"
        # Discovery questions should appear in the last assistant message
        last_assistant = next(
            (m for m in reversed(data["messages"]) if m["role"] == "assistant"), None
        )
        assert last_assistant is not None, "Expected at least one assistant message"

    # ── T203 ─────────────────────────────────────────────────────────────────
    async def test_completed_flow_returns_structured_problem(self, client: AsyncClient):
        """
        Happy Path #1c – Completed structuring flow.

        When the graph resolves fully (current_step = "end") the response must
        contain a non-empty `structured_problem` with the expected ProblemTree fields.
        """
        with patch(
            "app.services.graph.app_graph.ainvoke",
            new_callable=AsyncMock,
            return_value=STRUCTURED_STATE,
        ):
            response = await client.post(
                "/api/v1/agent/execute",
                json={"task": "Bulut altyapı maliyetlerimiz %50 arttı"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

        sp = data["structured_problem"]
        assert sp.get("main_problem"), "structured_problem.main_problem must not be empty"
        assert sp.get("problem_type"), "structured_problem.problem_type must not be empty"
        assert isinstance(sp.get("root_causes"), list) and len(sp["root_causes"]) > 0

    # ── T204 ─────────────────────────────────────────────────────────────────
    async def test_thread_id_passthrough(self, client: AsyncClient):
        """
        When a caller supplies a `thread_id`, the response must echo it back
        unchanged so session continuity is preserved.
        """
        custom_thread = "my-custom-session-abc123"
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "Rakiplerimizin fiyatlandırma stratejisi nedir?", "thread_id": custom_thread},
        )
        assert response.status_code == 200
        assert response.json()["thread_id"] == custom_thread


# ===========================================================================
# T3xx – POST /api/v1/analyze
# ===========================================================================

class TestAnalyze:
    """Tests for the primary analysis endpoint."""

    async def test_analyze_happy_path(self, client: AsyncClient):
        """
        Happy Path #2 – /analyze with a well-formed business crisis message.

        Checks: HTTP 200, non-empty thread_id, messages list, current_step present.
        """
        response = await client.post(
            "/api/v1/analyze",
            json={"message": "İnsan kaynakları departmanında yüksek çalışan sirkülasyonu yaşıyoruz"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"]
        assert data["current_step"] in ("awaiting_response", "discovery", "end", "search")
        assert len(data["messages"]) >= 1

    async def test_analyze_empty_message_rejected(self, client: AsyncClient):
        """Schema validation must reject an empty message string."""
        response = await client.post("/api/v1/analyze", json={"message": ""})
        # FastAPI returns 422 Unprocessable Entity for Pydantic validation failures
        assert response.status_code == 422


# ===========================================================================
# T4xx – POST /api/v1/respond  (Guardrail / Response Validation)
# ===========================================================================

class TestRespondGuardrail:
    """
    Guardrail tests verifying that the _validate_response() filter in router.py
    blocks irrelevant or insufficient user answers from triggering the
    Structuring Agent.
    """

    # ── T401 ─────────────────────────────────────────────────────────────────
    async def test_irrelevant_answer_blocked(self, client: AsyncClient):
        """
        Happy Path #2 (Guardrail Control) – "pizza yicem" must be blocked.

        "pizza yicem" matches _REJECT_STEMS ("pizza") and has no business
        override, so the SystemGuardrail fires before _validate_response.

        Expected behaviour:
          - HTTP 200 (graceful rejection, not a server error)
          - current_step = "rejected" (SystemGuardrail) OR "awaiting_response"
            (fallback validation) – both signal the session stays paused
          - response contains a rejection / warning message (not a ProblemTree)
        """
        response = await client.post(
            "/api/v1/respond",
            json={"thread_id": "test-thread-guardrail-001", "message": "pizza yicem"},
        )

        assert response.status_code == 200, response.text
        data = response.json()

        assert data["current_step"] in ("rejected", "awaiting_response"), (
            f"Guardrail failed – current_step should be 'rejected' or "
            f"'awaiting_response', got {data['current_step']!r}"
        )
        # There must be an assistant message (either rejection or warning)
        last_assistant = next(
            (m for m in reversed(data["messages"]) if m["role"] == "assistant"), None
        )
        assert last_assistant is not None, "Expected an assistant message"
        assert len(last_assistant["content"]) > 10

    # ── T402 ─────────────────────────────────────────────────────────────────
    async def test_single_word_answer_blocked(self, client: AsyncClient):
        """
        Single-word answers ("evet", "hayır", "tamam") must be rejected
        as they provide no analytical value.
        """
        for short_answer in ("evet", "tamam", "ok"):
            response = await client.post(
                "/api/v1/respond",
                json={"thread_id": "test-thread-guardrail-002", "message": short_answer},
            )
            assert response.status_code == 200
            assert response.json()["current_step"] == "awaiting_response", (
                f"Single-word answer {short_answer!r} should be blocked"
            )

    # ── T403 ─────────────────────────────────────────────────────────────────
    async def test_valid_detailed_answer_accepted(self, client: AsyncClient):
        """
        A substantive, detailed answer must pass validation and trigger
        the Structuring Agent (current_step transitions away from awaiting_response).
        """
        detailed_answer = (
            "Evet, son 3 ayda 3 yeni takım eklendi ve auto-scaling kuralları "
            "henüz güncellenmedi. Ayrıca staging ortamı kapanmadan bırakıldı."
        )
        with patch(
            "app.agents.structuring.structuring_agent_node",
            new_callable=AsyncMock,
            return_value=STRUCTURED_STATE,
        ):
            response = await client.post(
                "/api/v1/respond",
                json={"thread_id": "test-thread-valid-answer", "message": detailed_answer},
            )

        assert response.status_code == 200
        data = response.json()
        # Valid answer should NOT remain in awaiting_response
        assert data["current_step"] != "awaiting_response", (
            "Detailed answer should have passed validation and triggered structuring"
        )

    # ── T404 ─────────────────────────────────────────────────────────────────
    async def test_missing_session_returns_404(self, client: AsyncClient):
        """
        Responding to a non-existent thread_id must return HTTP 404.
        """
        with patch(
            "app.services.graph.app_graph.aget_state",
            new_callable=AsyncMock,
            return_value=None,  # simulate missing session
        ):
            response = await client.post(
                "/api/v1/respond",
                json={"thread_id": "non-existent-session-xyz", "message": "Herhangi bir cevap"},
            )
        assert response.status_code == 404


# ===========================================================================
# T6xx – Code-request routing (direct CodeAgent path)
# ===========================================================================

class TestCodeRequestRouting:
    """
    Verifies that technical coding requests bypass the LangGraph pipeline and
    are routed directly to CodeAgent, returning status="completed" immediately.
    """

    async def test_python_code_request_routed_to_code_agent(self, client: AsyncClient):
        """
        'bana python ile basit bir yazı tura oyunu yap' must be detected as a
        code request and return HTTP 200 with status="completed" (no discovery loop).
        """
        with patch(
            "app.services.code_agent.generate_code_from_task",
            new_callable=AsyncMock,
        ) as mock_gen:
            from app.services.code_agent import _MOCK_GENEL
            mock_gen.return_value = _MOCK_GENEL

            response = await client.post(
                "/api/v1/agent/execute",
                json={"task": "bana python ile basit bir yazı tura oyunu yap"},
            )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "completed", (
            f"Code requests must return 'completed' immediately, got {data['status']!r}"
        )
        # Peer Agent sets current_step="completed" for CODE_REQUEST
        assert data["current_step"] in ("completed", "end"), (
            f"Code path must end in 'completed' or 'end', got {data['current_step']!r}"
        )
        # Response must contain code content in the assistant message
        messages = data["messages"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert assistant_msgs, "Expected an assistant message with code"
        assert len(assistant_msgs[0]["content"]) > 50

    async def test_javascript_code_request_detected(self, client: AsyncClient):
        """JavaScript code requests must also be detected and routed correctly."""
        from app.services.code_agent import _MOCK_GENEL
        with patch(
            "app.services.code_agent.generate_code_from_task",
            new_callable=AsyncMock,
            return_value=_MOCK_GENEL,
        ):
            response = await client.post(
                "/api/v1/agent/execute",
                json={"task": "javascript ile basit bir todo list uygulaması oluştur"},
            )
        assert response.status_code == 200
        assert response.json()["status"] == "completed"

    async def test_irrelevant_request_still_blocked(self, client: AsyncClient):
        """
        'pizza yicem' must NOT be treated as a code request.
        The SystemGuardrail fires first (pizza matches _REJECT_STEMS)
        and returns current_step='rejected'.
        """
        response = await client.post(
            "/api/v1/respond",
            json={"thread_id": "test-guardrail-still-works", "message": "pizza yicem"},
        )
        assert response.status_code == 200
        assert response.json()["current_step"] in ("rejected", "awaiting_response"), (
            "pizza yicem must be blocked — expected 'rejected' or 'awaiting_response'"
        )

    async def test_business_crisis_not_mistaken_for_code(self, client: AsyncClient):
        """
        A genuine business crisis message must NOT hit the code fast-path.
        It should still go through LangGraph (mocked to return DISCOVERY_STATE).
        """
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "Satışlarımız bu çeyrekte %30 düştü, sebebini anlayamıyoruz"},
        )
        assert response.status_code == 200
        # Business crises go through Discovery and return awaiting_response
        assert response.json()["status"] in ("awaiting_response", "completed")


# ===========================================================================
# T7xx – SystemGuardrail rejection (off-topic requests)
# ===========================================================================

class TestSystemGuardrail:
    """
    Verifies that clearly off-topic requests are blocked by the SystemGuardrail
    before touching the LangGraph pipeline, and that the response satisfies
    all 4 rejection rules:
      1. Out-of-scope explanation
      2. Business-oriented system statement
      3. At least 3 example questions in Markdown
      4. current_step="rejected", status="rejected"
    """

    async def test_food_request_rejected(self, client: AsyncClient):
        """'pizza yicem' must return status='rejected' via agent/execute."""
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "bugün akşam pizza yicem hangisini önerirsin"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected", f"Expected 'rejected', got {data['status']!r}"
        assert data["current_step"] == "rejected"

    async def test_greeting_request_rejected(self, client: AsyncClient):
        """A casual greeting must be rejected without starting discovery."""
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "merhaba nasılsın bugün"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert data["current_step"] == "rejected"

    async def test_rejection_message_contains_example_questions(self, client: AsyncClient):
        """
        Rule 3 – The rejection assistant message must contain at least 3
        example business questions formatted as Markdown blockquotes (">").
        """
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "bana güzel bir film öner"},
        )
        data = response.json()
        assert data["status"] == "rejected"
        assistant_content = next(
            m["content"] for m in data["messages"] if m["role"] == "assistant"
        )
        # Must contain Markdown blockquotes for example questions
        blockquote_count = assistant_content.count(">")
        assert blockquote_count >= 3, (
            f"Expected ≥3 example questions ('>'), found {blockquote_count}"
        )

    async def test_rejection_message_mentions_scope(self, client: AsyncClient):
        """
        Rules 1 & 2 – Rejection message must explain the request is out of
        scope AND state the system is business-oriented.
        """
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "hava durumu nasıl yarın için"},
        )
        content = next(
            m["content"]
            for m in response.json()["messages"]
            if m["role"] == "assistant"
        )
        # Rule 1: mention out-of-scope concept
        assert any(w in content.lower() for w in ("kapsam", "dışında", "alanı")), (
            "Rejection message must explain the request is out of scope"
        )
        # Rule 2: mention business focus
        assert any(w in content.lower() for w in ("iş", "business", "uzmanlaşmış")), (
            "Rejection message must state the system is business-oriented"
        )

    async def test_dondurma_suffix_rejected(self, client: AsyncClient):
        """
        'dondurma alayımmı' must be caught by the guardrail.
        Tests Turkish agglutinative suffix tolerance: stem 'dondurma' is found
        as a substring even though it is followed by a suffixed word.
        """
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "dondurma alayımmı"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected", (
            f"'dondurma alayımmı' must be rejected, got status={data['status']!r}"
        )
        assert data["current_step"] == "rejected"

    async def test_kebapci_suffix_rejected(self, client: AsyncClient):
        """'kebapçıya gidiyorum' must be caught (stem: kebap)."""
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "kebapçıya gidiyorum hangisini önerirsin"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "rejected"

    async def test_business_request_with_food_word_not_rejected(self, client: AsyncClient):
        """
        A business sentence that happens to contain a food-related word must
        NOT be rejected — the business signal overrides the rejection pattern.
        E.g. "restoran sektöründe maliyet optimizasyonu nasıl yapılır"
        """
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "restoran sektöründe maliyet optimizasyonu nasıl yapılır"},
        )
        assert response.status_code == 200
        assert response.json()["status"] != "rejected", (
            "Business request with 'restoran' keyword must NOT be rejected"
        )

    # ── T712 – /respond is a thin layer: all messages go to Structuring ─────
    async def test_respond_any_message_passes_length_check_to_structuring(
        self, client: AsyncClient
    ):
        """
        /respond no longer classifies intent — it only validates message length.
        Any message ≥ 3 words (including a coding request) is forwarded to
        Structuring Agent and returns current_step='end'.
        The Peer Agent guardrail applies only to /analyze and /agent/execute.
        """
        response = await client.post(
            "/api/v1/respond",
            json={
                "thread_id": "test-thread-respond-thin",
                "message": "aslında python ile tic tac toe oyunu yaz",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # 8-word message passes length check → structuring runs → current_step="end"
        assert data["current_step"] != "awaiting_response", (
            "/respond thin layer: a valid-length message must advance past awaiting_response"
        )
        messages = data["messages"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert assistant_msgs, "Expected at least one assistant message from Structuring"

    async def test_respond_off_topic_passes_to_structuring_not_rejected(
        self, client: AsyncClient
    ):
        """
        /respond is a THIN LAYER — it does NOT classify OOS topics.
        An off-topic message with ≥ 3 words passes length validation and is
        forwarded to Structuring Agent (current_step='end'), NOT rejected.
        Guardrail enforcement lives exclusively in the Peer Agent node, which
        runs only on fresh requests to /analyze and /agent/execute.
        """
        response = await client.post(
            "/api/v1/respond",
            json={
                "thread_id": "test-thread-respond-oos",
                "message": "bugün akşam restoran önerir misin",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # 5-word message → length check passes → structuring runs → "end"
        assert data["current_step"] != "awaiting_response", (
            "/respond thin layer must forward valid-length messages to Structuring"
        )
        assert data["current_step"] != "rejected", (
            "/respond no longer blocks OOS — guardrail is Peer Agent's responsibility"
        )


# ===========================================================================
# T8xx – Edge cases & boundary conditions
# ===========================================================================

class TestEdgeCases:
    """
    Verifies graceful degradation for unusual, malformed, or extreme inputs.

    Key questions:
      • Does the system crash or return 5xx for weird inputs?  (it must not)
      • Does whitespace-only text pass Pydantic but get blocked downstream?
      • Does gibberish fall back safely to BUSINESS_CRISIS?
      • Are business context overrides working for food-adjacent words?
      • Does a language mention without a code verb avoid the CodeAgent path?
    """

    # ── EC1 – Whitespace-only task ────────────────────────────────────────
    async def test_whitespace_only_task_handled_gracefully(self, client: AsyncClient):
        """
        "   " passes Pydantic min_length=1 but carries zero semantic signal.
        _mock_classify finds no code / OOS / business stems → BUSINESS_CRISIS.
        Pipeline must not crash; returns a known status.
        """
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "   "},
        )
        assert response.status_code == 200
        assert response.json()["status"] in ("awaiting_response", "completed", "rejected"), (
            "Whitespace task must return a known status — never 5xx or unknown"
        )

    # ── EC2 – Gibberish / unrecognizable input ────────────────────────────
    async def test_gibberish_task_falls_back_to_business_crisis(self, client: AsyncClient):
        """
        Completely random text matches no classifier signal → safe BUSINESS_CRISIS
        fallback. The pipeline must continue without crashing.
        """
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "asdfghjkl qwerty zxcvbnm 12345"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("awaiting_response", "completed"), (
            f"Gibberish must fall back to BUSINESS_CRISIS pipeline, got {data['status']!r}"
        )
        assert data["status"] != "rejected"

    # ── EC3 – Numeric / symbol-only task ─────────────────────────────────
    async def test_numeric_only_task_handled_without_crash(self, client: AsyncClient):
        """Pure numbers and symbols carry no topic signal — graceful fallback required."""
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "123456 !@#$%^&*()"},
        )
        assert response.status_code == 200
        assert response.json()["status"] in ("awaiting_response", "completed", "rejected")

    # ── EC4 – Very long task (stress / token-limit boundary) ──────────────
    async def test_very_long_task_no_crash(self, client: AsyncClient):
        """
        A task > 3 000 characters must be accepted and processed without a
        500 error.  In MOCK_MODE the LLM is never called, so no token quota
        is consumed — this purely tests pipeline robustness.
        """
        long_task = ("Şirketimizde ciddi operasyonel sorunlar yaşıyoruz. " * 70).strip()
        assert len(long_task) > 3000, "Precondition: task must exceed 3000 chars"

        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": long_task},
        )
        assert response.status_code in (200, 422), (
            f"Very long task must NOT return 5xx — got {response.status_code}"
        )

    # ── EC5 – /respond with whitespace-only message ───────────────────────
    async def test_respond_whitespace_only_message_blocked(self, client: AsyncClient):
        """
        "   " passes Pydantic min_length but word_count = 0.
        _validate_response must reject it as 'too_short',
        keeping current_step = "awaiting_response".
        """
        response = await client.post(
            "/api/v1/respond",
            json={"thread_id": "test-edge-ws-respond", "message": "   "},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_step"] in ("awaiting_response", "rejected"), (
            f"Whitespace-only respond message must be blocked, got {data['current_step']!r}"
        )

    # ── EC6 – /respond with empty message → 422 ──────────────────────────
    async def test_respond_empty_message_returns_422(self, client: AsyncClient):
        """Empty string in /respond must be rejected by Pydantic with 422."""
        response = await client.post(
            "/api/v1/respond",
            json={"thread_id": "test-edge-empty-respond", "message": ""},
        )
        assert response.status_code == 422

    # ── EC7 – /agent/execute with completely empty JSON body ─────────────
    async def test_execute_empty_json_body_returns_422(self, client: AsyncClient):
        """An empty JSON object {} is missing the required `task` field → 422."""
        response = await client.post("/api/v1/agent/execute", json={})
        assert response.status_code == 422

    # ── EC8 – /agent/execute with no body at all ─────────────────────────
    async def test_execute_no_body_returns_422(self, client: AsyncClient):
        """A request with no body must return 422 (Pydantic cannot parse None)."""
        response = await client.post(
            "/api/v1/agent/execute",
            content=b"",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    # ── EC9 – Language keyword without code-action verb ───────────────────
    async def test_language_mention_without_code_action_is_business(self, client: AsyncClient):
        """
        Mentioning a programming language as a business/trend topic (no code
        action verb like 'yaz', 'yap') must NOT route to CodeAgent.
        It should be treated as BUSINESS_CRISIS.

        E.g. "Python ekosistemindeki şirketlerin büyüme trendi nedir"
             → has_lang=True  but  has_action=False  → not CODE_REQUEST
        """
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "Python ekosistemindeki şirketlerin büyüme trendi nedir"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] != "rejected", (
            "Language-as-business-topic must NOT be rejected"
        )
        # Pipeline (not CodeAgent) handles this
        assert data["status"] in ("awaiting_response", "completed")

    # ── EC10 – Food word inside a genuine business question ───────────────
    async def test_food_word_inside_business_context_not_rejected(self, client: AsyncClient):
        """
        "pizza zinciri franchising modelinde büyüme stratejisi nasıl kurulur"
        contains 'pizza' (OOS stem) but also 'büyüme' (business signal).
        Business signal must override — result must NOT be 'rejected'.
        """
        response = await client.post(
            "/api/v1/agent/execute",
            json={"task": "pizza zinciri franchising modelinde büyüme stratejisi nasıl kurulur"},
        )
        assert response.status_code == 200
        assert response.json()["status"] != "rejected", (
            "Pizza franchise business question must NOT be rejected — "
            "business signal 'büyüme' must override the food stem"
        )

    # ── EC11 – Two-word task (below _validate_response threshold) ─────────
    async def test_two_word_respond_message_blocked(self, client: AsyncClient):
        """
        A 2-word /respond message ("biraz arttı") has word_count=2 < 3 →
        _validate_response rejects it, session stays in awaiting_response.
        """
        response = await client.post(
            "/api/v1/respond",
            json={"thread_id": "test-edge-twoword", "message": "biraz arttı"},
        )
        assert response.status_code == 200
        assert response.json()["current_step"] == "awaiting_response", (
            "2-word answer is too short and must be blocked by _validate_response"
        )


# ===========================================================================
# T5xx – Input schema validation
# ===========================================================================

class TestInputValidation:
    """FastAPI / Pydantic schema boundary tests."""

    async def test_agent_execute_missing_task_field(self, client: AsyncClient):
        """Request body without `task` must be rejected with 422."""
        response = await client.post("/api/v1/agent/execute", json={})
        assert response.status_code == 422

    async def test_agent_execute_empty_task_rejected(self, client: AsyncClient):
        """Empty string task must be rejected with 422 (min_length=1)."""
        response = await client.post("/api/v1/agent/execute", json={"task": ""})
        assert response.status_code == 422

    async def test_respond_missing_thread_id(self, client: AsyncClient):
        """RespondRequest without thread_id must return 422."""
        response = await client.post(
            "/api/v1/respond",
            json={"message": "Detaylı bir cevap veriyorum buraya"},
        )
        assert response.status_code == 422

    async def test_history_endpoint_accessible(self, client: AsyncClient):
        """GET /api/v1/history/{user_id} must return 200 with empty list for unknown user."""
        response = await client.get("/api/v1/history/unknown-user-test")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "unknown-user-test"
        assert data["count"] == 0
        assert data["analyses"] == []
