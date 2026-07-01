"""
Discovery Agent — ardışık dinamik mülakat (sequential turn-by-turn interview).

LLM çıktısı her zaman `.with_structured_output(DiscoveryQuestion)` ile
bağlanmıştır → yalnızca TEK bir `question: str` alanı döner.
Çoklu soru üretmek yapısal olarak imkânsızdır.

Tur akışı:
  interview_turns == 0  →  İlk yönelim sorusu   →  turns = 1  →  PAUSE
  interview_turns == 1  →  Konu sorusu 1        →  turns = 2  →  PAUSE
  interview_turns == 2  →  Konu sorusu 2        →  turns = 3  →  PAUSE
  interview_turns == 3  →  Konu sorusu 3        →  turns = 4  →  PAUSE
  interview_turns >= 4  →  Bu düğüm çağrılmaz;
                           /respond router'ı Structuring'e yönlendirir.

MUTLAK KURAL: Çözüm, tavsiye, analiz veya problem ağacı hiçbir zaman önerilmez.
"""
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.models.schemas import DiscoveryQuestion, DiscoverySummary
from app.models.state import AgentState

MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() == "true"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM — `.with_structured_output` garantisi: model daima tek soru döndürür
# ---------------------------------------------------------------------------
_llm_structured: ChatOpenAI | None = None
_llm_synthesis: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    """
    GPT-4o bağlı DiscoveryQuestion çıktısı.
    Tek seferlik yapılandırma; modül ömrü boyunca saklanır.
    """
    global _llm_structured
    if _llm_structured is None:
        _llm_structured = ChatOpenAI(
            model="gpt-4o", temperature=0.4
        ).with_structured_output(DiscoveryQuestion)
    return _llm_structured


def _get_synthesis_llm() -> ChatOpenAI:
    """
    GPT-4o bağlı DiscoverySummary çıktısı.
    Mülakat sonunda 4 zorunlu alanı üretmek için kullanılır.
    """
    global _llm_synthesis
    if _llm_synthesis is None:
        _llm_synthesis = ChatOpenAI(
            model="gpt-4o", temperature=0.2
        ).with_structured_output(DiscoverySummary)
    return _llm_synthesis


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_SYSTEM_FIRST = """\
Sen deneyimli bir kurumsal kriz danışmanısın. Müşterin ilk kriz girdisini
derinlemesine analiz ederek mülakatin ilk turunu yürütüyorsun.

━━━ MUTLAK KURAL ━━━
ASLA birden fazla soru sorma. SADECE 1 adet hibrit, stratejik soru üreteceksin.
Bu kural hiçbir koşulda ihlal edilemez.
━━━━━━━━━━━━━━━━━━━

━━━ SORUNUN HİZMET ETMESİ GEREKEN İKİ AMAÇ ━━━

AMAÇ 1 — Problem Tanımı Netleştirme:
  Sorunun sınırlarını ortaya çıkar: hangi departman/süreç etkileniyor, ne zamandır
  devam ediyor, büyüklük/oran olarak ne kadar hissediliyor?

AMAÇ 2 — Gerçek İhtiyaç ile İstenen Çözüm Ayrımı:
  Kullanıcının zihninde zaten bir "reçete/çözüm" var mı (ve onay mı arıyor), yoksa
  önce problemin kök nedenini ve görünürlük eksikliğini mi gidermek istiyor?
  Bu ayrımı ilk turda anlamak, mülakatın geri kalanını doğru yönlendirmek için
  kritiktir.

━━━ SORU YAZIM KURALLARI ━━━
▸ İki amacı TEK bir akıcı soru içinde birleştir — iki ayrı cümle yazma.
▸ Hem etkilenen alanı/süreyi/ölçeği hem de müşterinin beklentisini (çözüm mü,
  görünürlük mü) sorgulayan hibrit bir yapı kur.
▸ Açık uçlu ol; evet/hayır ile kapatılamayacak şekilde yaz.
▸ Sayısal veri veya zaman çerçevesi içermesini özendiren bir ifade ekle.
▸ Hiçbir çözüm, tavsiye veya yorum ekleme.
▸ Türkçe yaz.

`question` alanına YALNIZCA soruyu yaz — başka hiçbir metin ekleme."""

_SYSTEM_FOLLOWUP = """\
Sen deneyimli bir kurumsal kriz danışmanısın. Bir iş krizi mülakatını
turn-by-turn olarak yürütüyorsun.

━━━ MUTLAK KURAL ━━━
ASLA birden fazla soru sorma. SADECE 1 adet follow-up sorusu üreteceksin.
Bu kural hiçbir koşulda ihlal edilemez.
━━━━━━━━━━━━━━━━━━━

Kullanıcının son verdiği cevabı derinlemesine analiz et ve o cevabı daha da
açacak, netleştirecek TEK bir follow-up sorusu üret.

Soru özellikleri:
▸ Önceki yanıtı doğrudan genişletmeli veya somutlaştırmalı.
▸ Genel/şablon sorular yasak; bu spesifik yanıta özel olmalı.
▸ Hiçbir çözüm, tavsiye veya analiz yorumu içerme.
▸ Açık uçlu ol.
▸ Türkçe yaz.

`question` alanına YALNIZCA soruyu yaz — başka hiçbir metin ekleme."""

# ---------------------------------------------------------------------------
# Synthesis system prompt — 4-field mandatory hand-off
# ---------------------------------------------------------------------------
_SYSTEM_SYNTHESIS = """\
Sen ENTRAPEER'in Discovery Agent'ısın. Şu an tamamlanmış bir iş krizi mülakatını
sentezleyecek ve 4 zorunlu çıktı alanını doldurarak Structuring Agent'a
aktarılacak veriyi üreteceksin.

━━━ ALAN TANIMLARI ━━━

customer_stated_problem:
  Müşterinin kendi ifadesiyle aktardığı problem.
  İlk mesajını ve mülakat boyunca kullandığı dili esas al.
  Yorum veya analiz ekleme — kullanıcının söylediklerini özetle.

identified_business_problem:
  Mülakat verisinden çıkardığın GERÇEK iş problemi.
  "customer_stated_problem"dan farklı olabilir ve çoğunlukla farklıdır.
  Kök neden perspektifinden, tek net cümleyle ifade et.

hidden_root_risk:
  Kullanıcının açıkça söylemediği ancak mülakat cevaplarından çıkarılabilecek
  gizli yapısal risk veya tehdit.
  Bu genellikle sorunun arkasındaki organizasyonel, pazar, liderlik veya
  finansal zaafiyettir.

customer_chat_summary:
  Mülakatın 3-5 cümlelik özeti.
  Hangi soruların sorulduğunu, hangi nicel/nitel verilerin paylaşıldığını
  ve ne tür organizasyonel bağlam ortaya çıktığını içermeli.

━━━ KRİTİK KURALLAR ━━━
• Yalnızca mülakatın sağladığı veriye dayan — spekülasyon yapma.
• Her alan Türkçe olmalıdır.
• `identified_business_problem` `customer_stated_problem`ın kopyası olmamalıdır.
• `hidden_root_risk` konuşmada geçen olgulardan mantıksal çıkarım ile üretilmelidir."""

# ---------------------------------------------------------------------------
# MOCK_MODE — 3 sıralı statik soru (CI / testler için — LLM çağrısı yok)
# ---------------------------------------------------------------------------
_MOCK_SUMMARY = DiscoverySummary(
    customer_stated_problem=(
        "Müşteri, satış ekibinin 6 aydır gelir hedeflerinin altında kaldığını "
        "ve bu durumun devam ettiğini ifade etti."
    ),
    identified_business_problem=(
        "Yetersiz pipeline yönetimi, CRM kullanım eksikliği ve bireysel koçluk "
        "mekanizmasının yokluğu nedeniyle satış ekibi sistematik olarak hedefin "
        "altında kalmaktadır."
    ),
    hidden_root_risk=(
        "Yöneticilerin operasyonel işlere gömülü olması nedeniyle stratejik "
        "satış liderliği fiilen işlevsizleşmiş; bu yapısal boşluk kısa vadede "
        "yetenek kaybı ve müşteri churn riskini artırmaktadır."
    ),
    customer_chat_summary=(
        "Mülakatta satış performans düşüşünün başlangıç zamanı, etkilenen "
        "kanallar ve alınan önlemler sorgulandı. Müşteri CRM verilerinin "
        "güncellenmediğini ve lead dönüşüm oranlarının izlenmediğini belirtti. "
        "Son 6 ayda liderlik değişikliği yaşandığı ve koçluk süreçlerinin "
        "askıya alındığı ortaya çıktı."
    ),
)

_MOCK_QUESTIONS: dict[int, str] = {
    0: (
        "Bu problemi hangi departman veya süreçte, ne kadar süredir yaşıyorsunuz "
        "ve şu ana kadar uyguladığınız çözüm girişimleri sonuç vermedi mi — yoksa "
        "henüz kök nedeni netleştirmeden doğrudan bir aksiyon planına mı ihtiyaç "
        "duyuyorsunuz? Mümkünse etkilenen metrik ve zaman çerçevesiyle paylaşın."
    ),
    1: (
        "Bu noktaya kadar sorunu çözmek için hangi önlemleri aldınız "
        "ve neden yeterince işe yaramadığını düşünüyorsunuz?"
    ),
    2: (
        "Bu sorunu birincil olarak hangi departman veya ekip sahapleniyor? "
        "Son 6 ayda organizasyonunuzda yapısal bir değişiklik (yeniden yapılanma, "
        "liderlik değişikliği, süreç güncellemesi) yaşandı mı?"
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_problem_context(state: AgentState) -> str:
    """
    Kriz bağlamını döndürür.

    Tur 0 (ilk soru): en son kullanıcı mesajını al — bu, yeni krizin tam
    açıklamasıdır.  Önceki arama/doğrudan yanıt mesajları checkpoint
    temizlendiğinden zaten yoktur; bununla birlikte, son mesajı kullanmak
    ilk mesajı almaktan daha güvenlidir ve geçmiş oturum kirliliğine karşı
    ek bir savunma katmanı sağlar.

    Tur >= 1 (takip soruları): ilk kullanıcı mesajı problem tanımı olarak
    daha uygundur (sonraki mesajlar kısa cevaplardır).
    """
    turns = int(state.get("interview_turns", 0))
    messages = state.get("messages", [])

    if turns == 0:
        # First question — use the most recent user message (new crisis text).
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    else:
        # Follow-up questions — first user message is the original crisis.
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    return ""


def _last_user_message(state: AgentState) -> str:
    """Mesaj geçmişindeki en son kullanıcı mesajını döndürür."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _format_history(history: list[dict]) -> str:
    """interview_history'i LLM prompt'u için okunabilir metne dönüştürür."""
    if not history:
        return "(Geçmiş yok)"
    lines: list[str] = []
    for i, qa in enumerate(history, 1):
        lines.append(f"Soru {i}: {qa.get('question', '')}")
        lines.append(f"Cevap {i}: {qa.get('answer', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
async def discovery_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node — ardışık mülakat düğümü.

    Her çağrıda LLM'den `.with_structured_output(DiscoveryQuestion)` aracılığıyla
    YALNIZCA tek bir `question: str` alır; yapısal olarak çoklu soru üretilemez.

    Döndürülen state:
      current_step     = "awaiting_response"
      interview_turns  = eski_değer + 1
      current_question = bu turda sorulan soru
      interview_history = güncellendi (varsa yeni Q-A çifti eklendi)
    """
    messages: list = list(state["messages"])
    turns: int = int(state.get("interview_turns", 0))
    current_q: str = state.get("current_question", "")
    history: list[dict] = list(state.get("interview_history", []))
    problem_context: str = _extract_problem_context(state)

    if not problem_context:
        fallback = (
            "Şirketinizde yaşanan krizi daha net anlayabilmem için lütfen "
            "durumu biraz daha detaylandırır mısınız? Hangi departmanda, "
            "ne zamandır, hangi metrikler etkileniyor?"
        )
        messages.append({"role": "assistant", "content": fallback})
        return {
            **state,
            "messages": messages,
            "current_step": "awaiting_response",
            "interview_turns": 1,
            "current_question": fallback,
            "interview_history": history,
        }

    # ── Tur ≥ 1: bir önceki soruya verilen son cevabı geçmişe kaydet ─────
    if turns > 0:
        last_answer = _last_user_message(state)
        if current_q and last_answer:
            history = list(history)
            history.append({"question": current_q, "answer": last_answer})
            logger.info(
                "DISCOVERY Q-A appended (turns=%d) answer_preview=%r",
                turns,
                last_answer[:60],
            )

    # ── Soru üret ─────────────────────────────────────────────────────────
    if MOCK_MODE:
        # CI/testler: LLM çağrısı yok, sabit şablondan al
        question = _MOCK_QUESTIONS.get(turns, _MOCK_QUESTIONS[2])
        logger.info("[MOCK] DISCOVERY turns=%d → static question", turns)
    else:
        if turns == 0:
            # ── İlk soru: yalnızca problem bağlamı ───────────────────────
            llm_input = [
                SystemMessage(content=_SYSTEM_FIRST),
                HumanMessage(
                    content=(
                        "Aşağıdaki iş krizini analiz et ve kökünü anlamak için "
                        "en kritik stratejik TEK soruyu üret:\n\n"
                        f"{problem_context}"
                    )
                ),
            ]
        else:
            # ── Follow-up: geçmiş Q-A dahil ──────────────────────────────
            llm_input = [
                SystemMessage(content=_SYSTEM_FOLLOWUP),
                HumanMessage(
                    content=(
                        f"Problem bağlamı:\n{problem_context}\n\n"
                        f"Mülakat geçmişi:\n{_format_history(history)}\n\n"
                        "Son yanıtı daha da derinleştiren TEK bir follow-up sorusu üret."
                    )
                ),
            ]

        result: DiscoveryQuestion = await _get_llm().ainvoke(llm_input)
        question = result.question.strip()
        logger.info(
            "DISCOVERY turns=%d structured output preview=%r", turns, question[:80]
        )

    # ── State güncelle ────────────────────────────────────────────────────
    new_turns = turns + 1
    messages.append({"role": "assistant", "content": question})
    logger.info("DISCOVERY asked Q%d (interview_turns → %d)", turns + 1, new_turns)

    return {
        **state,
        "messages": messages,
        "current_step": "awaiting_response",
        "interview_turns": new_turns,
        "current_question": question,
        "interview_history": history,
    }


# ---------------------------------------------------------------------------
# Synthesis node — called ONCE when interview is complete (turns >= 4)
# ---------------------------------------------------------------------------
async def discovery_synthesis_node(state: AgentState) -> AgentState:
    """
    Produces the 4 mandatory hand-off fields after the turn-by-turn interview
    concludes.  Called by the /respond router when interview_turns >= 4,
    BEFORE the Structuring Agent runs.

    Returns an updated AgentState with `discovery_summary` populated:
      - customer_stated_problem
      - identified_business_problem
      - hidden_root_risk
      - customer_chat_summary

    These fields are also merged into `structured_problem` so they are
    included in the MongoDB document saved after structuring completes.
    """
    history: list[dict] = list(state.get("interview_history", []))
    problem_context: str = _extract_problem_context(state)
    history_text: str = _format_history(history)

    if MOCK_MODE:
        logger.info("[MOCK] DISCOVERY_SYNTHESIS returning pre-built DiscoverySummary")
        summary = _MOCK_SUMMARY
    else:
        logger.info(
            "DISCOVERY_SYNTHESIS invoking LLM turns=%d history_len=%d",
            state.get("interview_turns", 0),
            len(history),
        )
        summary: DiscoverySummary = await _get_synthesis_llm().ainvoke(
            [
                SystemMessage(content=_SYSTEM_SYNTHESIS),
                HumanMessage(
                    content=(
                        f"Müşterinin ilk ifadesi:\n{problem_context}\n\n"
                        f"Mülakat geçmişi (soru-cevap):\n{history_text}\n\n"
                        "Yukarıdaki veriye dayanarak 4 zorunlu alanı doldur."
                    )
                ),
            ]
        )
        logger.info(
            "DISCOVERY_SYNTHESIS done customer_stated_preview=%r",
            summary.customer_stated_problem[:60],
        )

    summary_dict = summary.model_dump()

    # Merge into structured_problem so MongoDB persistence captures all fields
    existing_structured = dict(state.get("structured_problem", {}))
    existing_structured["discovery_summary"] = summary_dict

    return {
        **state,
        "discovery_summary": summary_dict,
        "structured_problem": existing_structured,
    }
