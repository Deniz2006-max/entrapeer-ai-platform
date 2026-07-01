"""
conftest.py – shared pytest fixtures for the ENTRAPEER test suite.

Design principles:
  • All external I/O (OpenAI, MongoDB, Redis) is mocked – tests run fully offline.
  • MOCK_MODE=true is enforced so agent nodes never call the real LLM.
  • app_graph.ainvoke uses a SMART side-effect function that simulates the
    Peer Agent's MOCK_MODE routing so code / rejection / discovery paths all
    behave correctly without running the real LangGraph graph.
"""

import os

# ── Env vars must be set BEFORE any app module is imported ──────────────────
os.environ["MOCK_MODE"] = "true"
os.environ["OPENAI_API_KEY"] = "sk-test-placeholder"
os.environ.setdefault("MONGO_DETAILS", "mongodb://localhost:27017")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

import contextlib
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Shared mock objects
# ---------------------------------------------------------------------------
from app.services.content_agent import ActionPlanReport

MOCK_ACTION_PLAN = ActionPlanReport(
    executive_summary=(
        "Bulut altyapı maliyetlerindeki %50 artış, ölçeksiz kaynak tahsisi ve "
        "optimize edilmemiş depolama yapılandırmasından kaynaklanmaktadır."
    ),
    immediate_actions=[
        "Kullanılmayan EC2/VM örneklerini 48 saat içinde kapatın.",
        "Tüm S3/Blob depolama yaşam döngüsü politikalarını gözden geçirin.",
    ],
    short_term_actions=[
        "FinOps aracı (CloudHealth, Spot.io) ile maliyet görünürlüğü sağlayın.",
        "Sabit yükler için Reserved Instance tekliflerini değerlendirin.",
    ],
    long_term_actions=[
        "Platform ekibine bulut maliyet optimizasyonu sorumluluğu atayın.",
        "Unit Economics çerçevesi oluşturun: servis başına maliyet izleme.",
    ],
    success_metrics=[
        "30 gün içinde bulut maliyetlerinde %20 azalma",
        "Ay sonu maliyet sapması < %5",
        "FinOps dashboard'u 60 gün içinde canlıya alınmış",
    ],
)


# ── Mock LangGraph snapshot (simulates a saved Redis checkpoint) ─────────────
class _MockSnapshot:
    """Mimics the langgraph StateSnapshot returned by aget_state()."""

    def __init__(self, values: dict):
        self.values = values


_DISCOVERY_Q1 = (
    "Bu problemi hangi departman veya süreçte, ne kadar süredir yaşıyorsunuz "
    "ve şu ana kadar uyguladığınız çözüm girişimleri sonuç vermedi mi — yoksa "
    "henüz kök nedeni netleştirmeden doğrudan bir aksiyon planına mı ihtiyaç "
    "duyuyorsunuz? Mümkünse etkilenen metrik ve zaman çerçevesiyle paylaşın."
)

DISCOVERY_STATE: dict = {
    "messages": [
        {"role": "user", "content": "Bulut altyapı maliyetlerimiz %50 arttı"},
        {"role": "assistant", "content": "Anlattığınız problemi dikkatle inceledim. Sizi Discovery Agent'a yönlendiriyorum..."},
        {"role": "assistant", "content": _DISCOVERY_Q1},
    ],
    "current_step": "awaiting_response",
    "structured_problem": {},
    "user_profile": {},
    # Sequential interview fields — Q1 has been asked; waiting for first answer
    "interview_turns": 1,
    "current_question": _DISCOVERY_Q1,
    "interview_history": [],
    "discovery_summary": {},
}

# State used by MOCK_SNAPSHOT (/respond tests):
# interview_turns = 4 means all 4 questions have been asked and answered,
# so the next /respond call routes to Structuring Agent (not Discovery).
_INTERVIEW_DONE_STATE: dict = {
    **DISCOVERY_STATE,
    "interview_turns": 4,
    "current_question": "Bu sorunu birincil olarak hangi departman sahapleniyor?",
    "interview_history": [
        {
            "question": _DISCOVERY_Q1,
            "answer": "Son 3 ayda fark ettik; compute maliyetleri %50, storage %30 arttı.",
        },
        {
            "question": "Bu noktaya kadar hangi önlemleri aldınız?",
            "answer": "Auto-scaling kurallarını güncelledik ama etkisi sınırlı kaldı.",
        },
    ],
}

MOCK_SNAPSHOT = _MockSnapshot(_INTERVIEW_DONE_STATE)


_MOCK_DISCOVERY_SUMMARY: dict = {
    "customer_stated_problem": (
        "Müşteri, bulut altyapı maliyetlerinin son 3 ayda %50 arttığını "
        "ve auto-scaling müdahalesinin sınırlı kaldığını ifade etti."
    ),
    "identified_business_problem": (
        "Merkezi FinOps yönetişimi olmaksızın eklenen yeni takımların bağımsız "
        "kaynak oluşturması, kontrol edilemeyen bir bulut harcaması sarmalı yarattı."
    ),
    "hidden_root_risk": (
        "Maliyet artışının arkasında organizasyonel büyümeyle orantısız ölçeklenen "
        "teknik borç ve merkezi platform sahipliğinin yokluğu yatmakta; bu yapısal "
        "boşluk kısa vadede ek yetenek kaybı ve operasyonel aksaklık riski taşıyor."
    ),
    "customer_chat_summary": (
        "Mülakatta maliyet artışının başlangıç zamanı, etkilenen bileşenler ve "
        "alınan önlemler sorgulandı. Compute maliyetleri %50, storage %30 arttı. "
        "Auto-scaling güncellendi ancak yeterli olmadı. Son 6 ayda 3 yeni takım "
        "eklendi ve merkezi FinOps ekibi mevcut değil."
    ),
}

STRUCTURED_STATE: dict = {
    "messages": [
        *DISCOVERY_STATE["messages"],
        {"role": "user", "content": "Evet, 3 yeni takım eklendi ve auto-scaling kapalıydı."},
        {
            "role": "assistant",
            "content": (
                "Teşekkürler, analiziniz başarıyla tamamlandı. "
                "Problem ağacını sağ panelden inceleyebilirsiniz."
            ),
        },
    ],
    "current_step": "end",
    "structured_problem": {
        "problem_type": "Cost",
        "main_problem": "Kontrol edilemeyen bulut harcamaları organizasyonel büyümeyle orantısız arttı.",
        "industry": "Teknoloji / SaaS",
        "root_causes": [
            {
                "main_cause": "Kaynak tahsisi yönetişim eksikliği",
                "sub_causes": [
                    "Yeni takımlar bağımsız kaynak oluşturabilmekte",
                    "Merkezi FinOps ekibi yok",
                ],
            }
        ],
        "confidence_score": 0.85,
        "discovery_summary": _MOCK_DISCOVERY_SUMMARY,
        "interview_history": [
            {
                "question": _DISCOVERY_Q1,
                "answer": "Son 3 ayda fark ettik; compute maliyetleri %50, storage %30 arttı.",
            },
            {
                "question": "Bu noktaya kadar hangi önlemleri aldınız?",
                "answer": "Auto-scaling kurallarını güncelledik ama etkisi sınırlı kaldı.",
            },
        ],
    },
    "user_profile": {},
    "discovery_summary": _MOCK_DISCOVERY_SUMMARY,
    "interview_history": [
        {
            "question": _DISCOVERY_Q1,
            "answer": "Son 3 ayda fark ettik; compute maliyetleri %50, storage %30 arttı.",
        },
        {
            "question": "Bu noktaya kadar hangi önlemleri aldınız?",
            "answer": "Auto-scaling kurallarını güncelledik ama etkisi sınırlı kaldı.",
        },
    ],
}

# ── Code response state (Peer Agent CODE_REQUEST path) ───────────────────────
CODE_STATE: dict = {
    "messages": [
        {"role": "user", "content": "bana python ile basit bir yazı tura oyunu yap"},
        {
            "role": "assistant",
            "content": (
                "## Yazı Tura Oyunu\n\n"
                "**Dil:** `Python`\n\n"
                "Rastgele yazı/tura seçimi yapan basit bir simülasyon.\n\n"
                "```python\n"
                "import random\n"
                "sonuc = random.choice(['Yazı', 'Tura'])\n"
                "print(f'Sonuç: {sonuc}')\n"
                "```\n\n"
                "**🔧 Sonraki Adımlar:**\n\n"
                "1. Kodu çalıştırın.\n"
                "2. Döngü ekleyerek çoklu atış yapın."
            ),
        },
    ],
    "current_step": "completed",
    "structured_problem": {"response_type": "code"},
    "user_profile": {},
}

# ── Rejection state (Peer Agent OUT_OF_SCOPE path) ────────────────────────────
_REJECTION_CONTENT = (
    "Bu talep **ENTRAPEER** platformunun çalışma kapsamı dışında kalmaktadır.\n\n"
    "ENTRAPEER, yalnızca **iş dünyası sorunları** üzerine uzmanlaşmış bir yapay zeka "
    "analiz platformudur.\n\n---\n\n"
    "### 📈 Büyüme Problemleri\n"
    "> *\"Müşteri edinme maliyetimiz son çeyrekte %40 arttı.\"*\n\n"
    "### 💰 Maliyet Optimizasyonu\n"
    "> *\"Operasyonel giderlerimiz bütçeyi %25 aşıyor.\"*\n\n"
    "### ⚙️ Operasyonel Aksaklıklar\n"
    "> *\"Tedarik zincirimizde ciddi gecikmeler yaşıyoruz.\"*\n\n"
    "### 👥 Organizasyonel Sorunlar\n"
    "> *\"Çalışan devir oranımız sektör ortalamasının 3 katına çıktı.\"*\n\n"
    "Yukarıdaki konulardan biriyle başlamak ister misiniz?"
)

REJECTION_STATE: dict = {
    "messages": [
        {"role": "user", "content": "pizza yicem"},
        {"role": "assistant", "content": _REJECTION_CONTENT},
    ],
    "current_step": "rejected",
    "structured_problem": {},
    "user_profile": {},
}


# ---------------------------------------------------------------------------
# Smart ainvoke factory
#
# Simulates the Peer Agent's MOCK_MODE routing so tests observe the same
# CODE_REQUEST / OUT_OF_SCOPE / BUSINESS_CRISIS split that the real Peer
# Agent applies in production — without actually running the LangGraph graph.
# ---------------------------------------------------------------------------
def _make_smart_ainvoke():
    """Return a coroutine that selects the right mock state based on task content."""
    # Mirror of peer.py's MOCK_MODE classifier tuples
    _code_langs = ("python", "javascript", "js", "typescript", "sql", "bash")
    _code_actions = ("yaz", "yap", "oluştur", "write", "create", "build", "üret")
    _code_concepts = (
        "yazı tura", "tic tac toe", "xox", "hesap makinesi",
        "fibonacci", "dosya okuma", "todo list", "web scraper",
    )
    _oos_stems = (
        "pizza", "dondurma", "kebap", "burger", "tarifi",
        "nasılsın", "naber", "günaydın", "film öner",
        "hava durumu", "futbol", "spor haberi", "netflix",
        "restoran öner", "kafe öner", "ne yesem", "yemek öner",
    )
    _biz_signals = (
        "satış", "maliyet", "müşteri", "pazar", "sektör",
        "kriz", "analiz", "rakip", "büyüme", "strateji",
    )

    async def _ainvoke(state, config=None, **kwargs):
        msgs = state.get("messages", [])
        task = ""
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                task = m.get("content", "")
                break
        k = task.lower()

        # Code detection
        has_lang = any(lang in k for lang in _code_langs)
        has_action = any(action in k for action in _code_actions)
        has_concept = any(concept in k for concept in _code_concepts)
        if (has_lang and has_action) or has_concept:
            # Return a code state carrying the actual user task for assertions
            return {**CODE_STATE, "messages": [
                {"role": "user", "content": task},
                CODE_STATE["messages"][1],
            ]}

        # OOS detection (with business override)
        has_oos = any(s in k for s in _oos_stems)
        has_biz = any(b in k for b in _biz_signals)
        if has_oos and not has_biz:
            return {**REJECTION_STATE, "messages": [
                {"role": "user", "content": task},
                REJECTION_STATE["messages"][1],
            ]}

        # BUSINESS_CRISIS / DIRECT_ANSWER → discovery flow
        return DISCOVERY_STATE

    return _ainvoke


# ---------------------------------------------------------------------------
# Fixture: async HTTP test client with all I/O mocked
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTPX client wired to the FastAPI app.

    Mocked externals:
      • MongoDB  – init/close/save/get are no-ops / return stubs
      • Redis    – compile_with_redis raises RuntimeError → MemorySaver fallback
      • LangGraph – ainvoke uses smart side-effect that mirrors Peer Agent MOCK_MODE
      • ContentAgent – returns MOCK_ACTION_PLAN without calling OpenAI
      • AgentLogger  – log_agent_run is a no-op
    """
    patches = [
        # MongoDB lifecycle
        patch("app.services.mongodb.init_mongo", return_value=None),
        patch("app.services.mongodb.close_mongo", new_callable=AsyncMock),
        # MongoDB operations
        patch(
            "app.services.mongodb.save_analysis",
            new_callable=AsyncMock,
            return_value="mock-doc-id",
        ),
        patch(
            "app.services.mongodb.get_analyses",
            new_callable=AsyncMock,
            return_value=[],
        ),
        # Redis (not available in CI) – triggers MemorySaver fallback in lifespan
        patch(
            "app.services.graph.compile_with_redis",
            side_effect=RuntimeError("Redis not available in test environment"),
        ),
        # Sub-agents – stub to avoid real LLM calls
        patch(
            "app.services.content_agent.generate_action_plan",
            new_callable=AsyncMock,
            return_value=MOCK_ACTION_PLAN,
        ),
        patch(
            "app.services.agent_logger.log_agent_run",
            new_callable=AsyncMock,
            return_value="mock-log-id",
        ),
        # Smart graph ainvoke: routes based on task content (mirrors Peer MOCK_MODE)
        patch(
            "app.services.graph.app_graph.ainvoke",
            new_callable=AsyncMock,
            side_effect=_make_smart_ainvoke(),
        ),
        # Default graph aget_state: returns saved discovery snapshot
        patch(
            "app.services.graph.app_graph.aget_state",
            new_callable=AsyncMock,
            return_value=MOCK_SNAPSHOT,
        ),
        # aupdate_state: called by /respond to persist interview state back to
        # the checkpointer after direct discovery/structuring node calls.
        # In tests this is a no-op (MemorySaver is not thread-safe across fixtures).
        patch(
            "app.services.graph.app_graph.aupdate_state",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ]

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)

        from app.main import app as fastapi_app

        async with AsyncClient(
            transport=ASGITransport(app=fastapi_app),
            base_url="http://test",
        ) as ac:
            yield ac
