import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.models.schemas import ProblemTree, RootCause
from app.models.state import AgentState

MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() == "true"

_MOCK_PROBLEM_TREE = ProblemTree(
    problem_type="Operational",
    main_problem=(
        "Satış ekibi, 6 aydır tutarsız pipeline yönetimi ve yetersiz koçluk desteği "
        "nedeniyle gelir hedeflerinin %40 altında kalmaktadır."
    ),
    industry="B2B Satış / Kurumsal Hizmetler",
    root_causes=[
        RootCause(
            main_cause="Yetersiz pipeline yönetimi ve CRM kullanımı",
            sub_causes=[
                "CRM verileri güncel tutulmadığından gerçek zamanlı tahmin yapılamıyor.",
                "Lead kalitesi ve dönüşüm oranları sistematik olarak izlenmiyor.",
            ],
        ),
        RootCause(
            main_cause="Koçluk ve mentorluk mekanizmalarının yokluğu",
            sub_causes=[
                "Bireysel performans farkları koçluk sistemi olmadığı için kapanmıyor.",
                "Yöneticiler operasyonel işlere gömülü; stratejik rehberlik yapamıyor.",
            ],
        ),
        RootCause(
            main_cause="Hedef belirleme sürecinin pazar gerçekliğinden kopukluğu",
            sub_causes=[
                "Hedefler pazar koşullarındaki değişimlerle senkronize edilmiyor.",
                "Satış ekibi güncel değer önerisiyle donatılmamış.",
            ],
        ),
    ],
    confidence_score=0.95,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM – lazily initialized; ProblemTree bound as structured output
# ---------------------------------------------------------------------------
_llm_structured: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm_structured
    if _llm_structured is None:
        _llm_structured = ChatOpenAI(
            model="gpt-4o", temperature=0.2
        ).with_structured_output(ProblemTree)
    return _llm_structured


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_STRUCTURING_SYSTEM = """Sen ENTRAPEER'in Problem Structuring & Diagnosis Agent'ısın.
Görevin: Discovery aşamasından elde edilen tüm konuşma verisini işleyerek
hiyerarşik ve yapılandırılmış bir Problem Ağacı (Problem Tree) üretmek.

━━━ ALAN TANIMLARI ━━━

problem_type → Aşağıdaki kategorilerden tam olarak birini seç:
  • "Growth"         – Büyüme, pazar payı, gelir artışı sorunları
  • "Cost"           – Maliyet artışı, verimlilik, karlılık sorunları
  • "Operational"    – Süreç, üretim, satış operasyonu, lojistik sorunları
  • "Technology"     – Sistem, altyapı, dijital dönüşüm sorunları
  • "Regulation"     – Uyumluluk, lisans, yasal kısıt sorunları
  • "Organizational" – İnsan kaynakları, yapı, liderlik, kültür sorunları
  • "Hybrid"         – Birden fazla kategori eş zamanlı ve iç içe geçmişse

main_problem → Tüm anlatının özünü TEK, net ve aksiyon alınabilir bir cümleyle ifade et.

industry → Şirketi en iyi tanımlayan sektör etiketi (örn. "SaaS / B2B Teknoloji").

root_causes → ZORUNLU SAYISAL SINIRLAR (Pydantic tarafından da doğrulanır):
  ▸ Ana neden sayısı: EN AZ 3, EN FAZLA 5  (3 ≤ n ≤ 5)
  ▸ Her ana nedenin alt neden sayısı: KESİNLİKLE 2 veya 3  (2 ≤ m ≤ 3)
  ▸ Bu sınırların dışına çıkılırsa şema doğrulama hatası oluşur.

  main_cause : "Neden..." hipotezi formatında tek net cümle
  sub_causes : Bu ana nedeni besleyen TAM OLARAK 2 veya 3 alt neden
               (gözlemlenebilir kanıtlara dayalı, spekülasyon içermez)

  Örnek yapı (4 ana neden, her biri 2 alt nedenle):
    ├── main_cause: "Neden pipeline yönetimi yetersiz?"
    │     sub_causes: ["CRM verisi güncel tutulmuyor", "Tahmin süreci sistemsiz"]
    ├── main_cause: "Neden koçluk mekanizması çalışmıyor?"
    │     sub_causes: ["Yöneticiler operasyona gömülü", "Bireysel fark izlenmiyor"]
    └── ...

confidence_score → 0.0-1.0 arası:
  0.0-0.4 : Yetersiz veri
  0.5-0.7 : Makul veri, bazı varsayımlar mevcut
  0.8-1.0 : Kapsamlı veri, yüksek güven

━━━ KRİTİK KURALLAR ━━━
• Discovery soruları ve kullanıcı cevaplarını dikkate al — bunlar en değerli veridir.
• "Customer Stated Problem" (kullanıcının ilk ifadesi) ile "Identified Business Problem"
  (gerçek kök sorun) arasındaki farkı ortaya çıkar.
• Yeni soru sorma. Elimizdeki veriyi yapılandır.
• Her kök neden farklı bir boyutu temsil etmeli: insan, süreç, teknoloji, pazar, yönetim.
• root_causes sayısı 3'ten az veya 5'ten fazla olmamalıdır — bu kurala uymayan
  çıktılar sistem tarafından reddedilir."""


# ---------------------------------------------------------------------------
# Helper: build full conversation transcript for the LLM
# ---------------------------------------------------------------------------
def _build_transcript(state: AgentState) -> str:
    lines: list[str] = []
    for msg in state["messages"]:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "").strip()
        elif hasattr(msg, "content"):
            role = getattr(msg, "type", "unknown")
            content = msg.content.strip()
        else:
            continue
        label = "KULLANICI" if role in ("user", "human") else "SİSTEM"
        lines.append(f"[{label}]\n{content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: format ProblemTree as Markdown for the chat panel
# ---------------------------------------------------------------------------
def _format_problem_tree(tree: ProblemTree) -> str:
    confidence_pct = int(tree.confidence_score * 100)
    confidence_label = (
        "Yüksek Güven" if tree.confidence_score >= 0.8
        else "Orta Güven" if tree.confidence_score >= 0.5
        else "Düşük Güven – Ek Veri Gerekli"
    )

    causes_md = ""
    for rc in tree.root_causes:
        causes_md += f"- **{rc.main_cause}**\n"
        for sc in rc.sub_causes:
            causes_md += f"  - {sc}\n"

    return (
        "## Problem Ağacı Analiz Raporu\n\n"
        f"**Problem Tipi:** {tree.problem_type} Problemi\n\n"
        f"**Ana Problem:** {tree.main_problem}\n\n"
        f"**Sektör:** {tree.industry}\n\n"
        "### Yapılandırılmış Problem Ağacı\n\n"
        f"{causes_md}\n"
        f"**Analiz Güven Skoru:** %{confidence_pct} — {confidence_label}"
    )


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
async def structuring_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: synthesizes the full conversation into a hierarchical ProblemTree.

    Returns an updated AgentState with:
    - a formatted analysis report appended to `messages`
    - `structured_problem` fully populated with the ProblemTree fields
    - `current_step` set to "end"
    """
    messages: list = list(state["messages"])
    transcript = _build_transcript(state)
    # Pull interview_history early so it is available for both the prompt
    # context and the final structured_problem document written to MongoDB.
    interview_history: list[dict] = list(state.get("interview_history", []))

    if not transcript:
        fallback = (
            "Analiz yapabilmem için yeterli konuşma geçmişi bulunamadı. "
            "Lütfen probleminizi detaylıca aktarın."
        )
        messages.append({"role": "assistant", "content": fallback})
        return {**state, "messages": messages, "current_step": "end"}

    existing = state.get("structured_problem", {})

    # ── Build rich context from DiscoverySummary (4 mandatory hand-off fields) ──
    discovery_summary: dict = state.get("discovery_summary", {}) or {}
    extra_context = ""
    if discovery_summary.get("customer_stated_problem"):
        extra_context += (
            "\n\n━━━ DISCOVERY AGENT SENTEZİ ━━━\n"
            f"Müşterinin Beyan Ettiği Problem:\n"
            f"  {discovery_summary['customer_stated_problem']}\n\n"
            f"Tespit Edilen Gerçek İş Problemi:\n"
            f"  {discovery_summary['identified_business_problem']}\n\n"
            f"Gizli Kök Risk:\n"
            f"  {discovery_summary['hidden_root_risk']}\n\n"
            f"Mülakat Özeti:\n"
            f"  {discovery_summary['customer_chat_summary']}"
        )
    elif existing.get("discovery_questions"):
        # Backwards-compatible fallback for older sessions
        questions_text = "\n".join(
            f"- {q}" for q in existing["discovery_questions"]
        )
        extra_context = (
            f"\n\nDiscovery aşamasında sorulan keşif soruları:\n{questions_text}"
        )

    logger.info("STRUCTURING building ProblemTree from transcript (len=%d)", len(transcript))

    # ------------------------------------------------------------------
    # MOCK MODE – return pre-built ProblemTree without LLM call
    # ------------------------------------------------------------------
    if MOCK_MODE:
        logger.info("[MOCK] STRUCTURING skipping LLM – returning pre-built ProblemTree")
        tree = _MOCK_PROBLEM_TREE
    else:
        tree = await _get_llm().ainvoke(
            [
                SystemMessage(content=_STRUCTURING_SYSTEM),
                HumanMessage(
                    content=(
                        "Aşağıda kullanıcı ile yapılan konuşmanın tam kaydı yer almaktadır."
                        f"{extra_context}\n\n"
                        "--- KONUŞMA KAYDI ---\n\n"
                        f"{transcript}\n\n"
                        "--- KAYIT SONU ---\n\n"
                        "Bu konuşmadan hiyerarşik Problem Ağacını çıkar. "
                        "Her ana nedeni besleyen 2-3 alt neden üret."
                    )
                ),
            ]
        )

    # Serialize root_causes into plain dicts for storage.
    # discovery_summary is preserved via **existing (already merged there by
    # discovery_synthesis_node).  Explicitly list it so the intent is clear.
    updated_structured_problem = {
        **existing,
        "problem_type": tree.problem_type,
        "main_problem": tree.main_problem,
        "industry": tree.industry,
        "root_causes": [rc.model_dump() for rc in tree.root_causes],
        "confidence_score": tree.confidence_score,
        # ── 4 mandatory hand-off fields from Discovery Agent ──────────────
        # Preserved from discovery_summary already merged into `existing`; the
        # explicit key ensures they are never dropped by a partial state update.
        "discovery_summary": discovery_summary or existing.get("discovery_summary", {}),
        # ── Raw interview transcript (ham soru-cevap geçmişi) ─────────────
        # Stored verbatim so business teams can audit every question the
        # Discovery Agent asked and every answer the user provided.
        # Falls back to whatever was already in `existing` for sessions that
        # pre-date this field.
        "interview_history": interview_history or existing.get("interview_history", []),
    }

    report = _format_problem_tree(tree)
    completion_message = (
        "Teşekkürler, analiziniz başarıyla tamamlandı. "
        "Problem ağacını sağ panelden inceleyebilirsiniz.\n\n"
        f"{report}"
    )

    messages.append({"role": "assistant", "content": completion_message})

    return {
        **state,
        "messages": messages,
        "current_step": "end",
        "structured_problem": updated_structured_problem,
    }
