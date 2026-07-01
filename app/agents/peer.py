"""
Peer Agent — ENTRAPEER'in giriş düğümü (entry node).

Her gelen talebi semantik ve kurumsal bağlamıyla değerlendirip 4 yoldan
birini seçer.  Router, Guardrail ve CodeAgent tetiklemesi bu düğümde yaşar;
API katmanı (router.py) hiçbir sınıflandırma yapmaz.

Routing kararları:
  BUSINESS_CRISIS  → Discovery Agent'a yönlendir (kriz keşif soruları)
  DIRECT_ANSWER    → DuckDuckGo araması + GPT-4o sentezi → END
  CODE_REQUEST     → CodeAgent çağrısı → END
  OUT_OF_SCOPE     → 4 kurallı Markdown ret mesajı → END (current_step="rejected")
"""
import logging
import os
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.models.state import AgentState
from app.services.search import internet_search
from app.services.code_agent import generate_code_from_task
from app.services.agent_logger import log_agent_run

MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() == "true"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM — lazily initialised (avoids OPENAI_API_KEY requirement at import)
# ---------------------------------------------------------------------------
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    return _llm


# ---------------------------------------------------------------------------
# Rejection message — 4 jüri kuralına uygun Markdown ret
# ---------------------------------------------------------------------------
REJECTION_MSG = """\
Bu talep **ENTRAPEER** platformunun çalışma kapsamı dışında kalmaktadır.

ENTRAPEER, yalnızca **iş dünyası sorunları** üzerine uzmanlaşmış bir yapay zeka \
analiz platformudur. Kişisel sorular, günlük yaşama ait talepler veya genel \
sohbet bu sistemin hizmet alanına girmemektedir.

---

İş perspektifinde aşağıdaki gibi sorular sorabilirsiniz:

### 📈 Büyüme Problemleri
> *"Müşteri edinme maliyetimiz son çeyrekte %40 arttı. Büyüme stratejimizde nerede hata yapıyoruz?"*

### 💰 Maliyet Optimizasyonu
> *"Operasyonel giderlerimiz bütçeyi %25 aşıyor. Kök nedenleri neler olabilir ve nasıl azaltabiliriz?"*

### ⚙️ Operasyonel Aksaklıklar
> *"Tedarik zincirimizde ciddi gecikmeler yaşıyoruz. Süreç verimliliğini nasıl artırabiliriz?"*

### 👥 Organizasyonel Sorunlar
> *"Çalışan devir oranımız sektör ortalamasının 3 katına çıktı. Bu durumu nasıl analiz edebiliriz?"*

Yukarıdaki konulardan biriyle başlamak ister misiniz?"""

_DISCOVERY_TRANSITION = """\
## 🔎 Problem Analizi Gerekiyor

Talebiniz, anlık bir arama yanıtıyla çözülemeyecek kadar **katmanlı ve stratejik** \
bir iş krizidir.

Bu nedenle sizi **Business Sense Discovery & Problem Structuring Agent**'a \
yönlendiriyorum.

---

**Bu ajan ne yapacak?**
- Probleminizin **kök nedenlerini** derinlemesine ortaya çıkarmak için size \
birkaç stratejik soru soracak
- Her soruyu bir önceki cevabınıza göre özelleştirecek (ardışık mülakat)
- Tüm bilgileri topladıktan sonra yapılandırılmış bir **Problem Ağacı** ve \
aksiyon planı üretecek

> Lütfen gelen soruları mümkün olduğunca **sayısal veri ve zaman çerçevesiyle** \
yanıtlayın — bu, analiz kalitesini doğrudan etkiler.

---

İlk soru hazırlanıyor...\
"""

# ---------------------------------------------------------------------------
# Peer Agent system prompt — soyut prensip tabanlı, kelime listesiz
# ---------------------------------------------------------------------------
_ROUTER_SYSTEM = """\
Sen ENTRAPEER'in giriş düğümü olan Peer Agent'sın.

Gelen talebi semantik ve kurumsal bağlamıyla değerlendirip YALNIZCA şu dört
etiketten birini döndür — başka hiçbir sözcük, açıklama veya noktalama ekleme:

  BUSINESS_CRISIS · DIRECT_ANSWER · CODE_REQUEST · OUT_OF_SCOPE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS_CRISIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kullanıcının KENDİ organizasyonunda ya da şirketinde yapısal bir kriz, finansal
sorun veya operasyonel aksaklık var; içgörü üretmek için ek sorular sorulması
gerekiyor.

İşaretler: "bizim", "ekibimiz", "şirketimiz" gibi birinci şahıs aitlik ekleri
+ olumsuz kriz dili (düştü, arttı, kaybettik, tutturamıyoruz, yüksek, kötü).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIRECT_ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dış dünyaya ait kurumsal veya sektörel bir bilgi, pazar araştırması, trend,
rakip analizi veya makroekonomik sorusu. Cevap, mevcut verilere (internet
araması) dayanarak doğrudan üretilebilir; kullanıcıya soru sorulmasına gerek yok.

⚑ ALTIN KURAL — Bağlamsal Baskınlık:
Girdide yiyecek, eğlence veya günlük yaşam sözcükleri geçse dahi, bu sözcükler
bir sektör, pazar, tedarik zinciri, şirket maliyeti veya ticari rekabet
bağlamında kullanılıyorsa → KESİNLİKLE DIRECT_ANSWER.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CODE_REQUEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Herhangi bir programlama dilinde çalıştırılabilir bir kod bloğu, algoritma,
script, teknik sorgu veya yazılım aracı YAZILMASI isteniyor.

Ayırt edici özellik: Kullanıcı bir kod parçasının üretilmesini açıkça talep ediyor.
Bir programlama dilinden söz edilmesi tek başına yeterli değildir; yazma eylemi
açıkça ifade edilmelidir.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUT_OF_SCOPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
İş dünyası, herhangi bir sektör veya teknik kodlamayla hiçbir kurumsal/analitik
bağlantısı olmayan tamamen bireysel, sosyal veya tüketimsel talepler.

(kişisel yemek/sipariş kararları, selamlaşma, saf eğlence, kişisel sağlık,
günlük sohbet — iş ya da sektör boyutu yoksa)

Şüphe durumunda → DIRECT_ANSWER seç.
"""

_SEARCH_SYNTHESIZER_SYSTEM = """\
Sen ENTRAPEER'in pazar istihbarat uzmanısın.

━━━ YANIT FORMATI (kesinlikle uy) ━━━

**🔍 [Sorunun özünü yansıtan 5-8 kelimelik başlık]**

> **Temel Bulgu:** Sorunun en kritik cevabını TEK cümleyle özetle.

**Öne Çıkanlar**
• [Madde 1 — rakip / oyuncu / trend + somut veri veya oran]
• [Madde 2 — rakip / oyuncu / trend + somut veri veya oran]
• [Madde 3 — rakip / oyuncu / trend + somut veri veya oran]
• [Madde 4 — varsa, yoksa 3 madde yeterli]

**Dikkat Edilmesi Gereken Nokta**
[1-2 cümle: risik, fırsat veya gözden kaçan kritik faktör]

---
*Kaynaklar: [Kaynak 1] · [Kaynak 2] · [Kaynak 3]*

━━━ KURALLAR ━━━
• Yanıt Türkçe olmalı.
• Maddeler kısa ve taramalı (scannable) olsun — uzun paragraf yazma.
• Rakamlar, yüzdeler ve yıllar **kalın** yaz: **%40**, **2026**, **3. sıra**.
• Spekülasyon yapma; verilen araştırma verilerine sadık kal.
• Kaynak bulunamazsa "Sektör raporları & pazar araştırması (2025–2026)" yaz."""

# ---------------------------------------------------------------------------
# MOCK_MODE templates — sector-specific search stubs
# ---------------------------------------------------------------------------
_SEARCH_TEMPLATES: list[tuple[set[str], str]] = [
    (
        {"fintech", "banka", "bankacılık", "ödeme", "kripto", "neobank"},
        (
            "**🔍 Fintech & Bankacılık — Rekabet ve Trend Haritası**\n\n"
            "> **Temel Bulgu:** Açık bankacılık ve gömülü finans, **2026** itibarıyla "
            "geleneksel bankaların dijital dönüşümünü zorunlu kılıyor.\n\n"
            "**Öne Çıkanlar**\n"
            "• **Stripe & Revolut** — küresel ödemelerde lider; Revolut **45M+** kullanıcıya ulaştı\n"
            "• **Papara & Midas** — Türkiye'de neobank büyümesini temsil eden öncü oyuncular\n"
            "• **Gömülü Finans** — non-fintech şirketlerin ödeme/kredi sunması; pazar **%32** büyüyor\n"
            "• **Kripto & Stablecoin** — kurumsal benimseme artıyor, düzenleyici baskı sürüyor\n\n"
            "**Dikkat Edilmesi Gereken Nokta**\n"
            "Açık bankacılık API standartlarını geç benimseyen kurumlar müşteri verisi "
            "avantajını hızla kaybediyor.\n\n"
            "---\n"
            "*Kaynaklar: CB Insights Fintech Report 2026 · BDDK Dijital Bankacılık Verileri Q1 2026 · "
            "McKinsey Global Banking Review*"
        ),
    ),
    (
        {"otomotiv", "araç", "araba", "elektrikli", "ev", "tesla", "togg"},
        (
            "**🔍 Otomotiv & EV — Küresel Rekabet Tablosu**\n\n"
            "> **Temel Bulgu:** BYD, **2025**'te küresel EV satışlarında Tesla'yı geçerek "
            "yeni lider konumuna yükseldi.\n\n"
            "**Öne Çıkanlar**\n"
            "• **BYD** — **%22** küresel EV pazar payı; fiyat avantajıyla Avrupa'ya genişliyor\n"
            "• **Tesla** — marka gücü koruyor ancak fiyat baskısı marjları sıkıştırıyor\n"
            "• **Togg** — Türkiye'de **%4** pazar payı; 2026'da ihracat hedefi var\n"
            "• **Yazılım Tanımlı Araç (SDV)** — OTA güncellemeleri ve AI sürücü desteği kritik rekabet ekseni\n\n"
            "**Dikkat Edilmesi Gereken Nokta**\n"
            "Batarya hammadde tedarik zinciri (lityum, kobalt) oyunculara göre "
            "maliyet uçurumu yaratıyor; dikey entegrasyon avantajlı.\n\n"
            "---\n"
            "*Kaynaklar: IEA Global EV Outlook 2026 · McKinsey Center for Future Mobility · BloombergNEF*"
        ),
    ),
    (
        {"e-ticaret", "eticaret", "perakende", "retail", "alışveriş"},
        (
            "**🔍 E-Ticaret & Perakende — Türkiye ve Küresel Tablo**\n\n"
            "> **Temel Bulgu:** Türkiye e-ticaret pazarı **2025**'te **$40 Milyar** eşiğini geçti; "
            "Trendyol liderliğini korurken hızlı teslimat kritik farklılaştırıcı haline geldi.\n\n"
            "**Öne Çıkanlar**\n"
            "• **Trendyol** — **%58** pazar payı; super-app stratejisiyle finans ve yemek teslimatına genişledi\n"
            "• **Hepsiburada** — B2B ve kurumsal segment odağıyla farklılaşıyor\n"
            "• **Amazon TR** — global lojistik ağı avantajıyla rekabeti artırıyor\n"
            "• **Sosyal Ticaret** — Instagram/TikTok shop entegrasyonu **%40** yıllık büyüme kaydediyor\n\n"
            "**Dikkat Edilmesi Gereken Nokta**\n"
            "Son kilometre lojistiği maliyeti, kârlılığın önündeki en büyük engel; "
            "bu alanda yatırım yapan oyuncular uzun vadede ayrışıyor.\n\n"
            "---\n"
            "*Kaynaklar: Statista E-Commerce Turkey 2026 · ETID Yıllık Raporu · eMarketer Global Retail*"
        ),
    ),
    (
        {"yazılım", "saas", "teknoloji", "yapay zeka", "ai", "bulut", "cloud"},
        (
            "**🔍 SaaS & AI Teknoloji — Pazar Dinamikleri**\n\n"
            "> **Temel Bulgu:** Yapay zeka entegrasyonu **2026** itibarıyla SaaS fiyatlandırmasını "
            "yeniden şekillendiriyor; AI'sız ürünler hızla pazar payı kaybediyor.\n\n"
            "**Öne Çıkanlar**\n"
            "• **Microsoft (Copilot)** — kurumsal AI asistan entegrasyonunda dominant; **$10B+** ARR hedefi\n"
            "• **Salesforce & ServiceNow** — CRM ve iş süreci otomasyonunda AI agent yarışı\n"
            "• **Bulut Benimseme** — Kurumların **%78**'i multi-cloud stratejisine geçti\n"
            "• **Türkiye SaaS** — Logo, Foriba ve Türk yazılım ekosistemi uluslararasılaşma aşamasında\n\n"
            "**Dikkat Edilmesi Gereken Nokta**\n"
            "AI özellik yarışı müşteri karar karmışıklığı yaratıyor; "
            "net ROI gösteremeyen ürünler churn ile karşılaşıyor.\n\n"
            "---\n"
            "*Kaynaklar: Gartner AI & Cloud Report 2026 · Bessemer State of the Cloud 2026 · IDC SaaS Forecast*"
        ),
    ),
]

_SEARCH_GENERIC = (
    "**🔍 Sektör Analizi — Genel Rekabet ve Trend Özeti**\n\n"
    "> **Temel Bulgu:** İlgili pazarda dijitalleşme ve otomasyon büyümeyi belirleyen "
    "başlıca faktör olmaya devam ediyor.\n\n"
    "**Öne Çıkanlar**\n"
    "• **Dijital Dönüşüm** — müşteri deneyimi ve süreç otomasyonu yatırımları artıyor\n"
    "• **Veri Analitiği** — gerçek zamanlı karar desteği rekabet avantajı sağlıyor\n"
    "• **AI Entegrasyonu** — operasyonel verimlilik için öncelikli yatırım alanı\n\n"
    "**Dikkat Edilmesi Gereken Nokta**\n"
    "Dijital dönüşümde geç kalan oyuncular müşteri beklentisi açığını kapatmakta zorlanıyor.\n\n"
    "---\n"
    "*Kaynaklar: Gartner Emerging Technologies Report 2026 · McKinsey Global Business Perspective*"
)

_CODE_MOCK_RESPONSE = (
    "## Kod Şablonu\n\n"
    "**Dil:** `Python`\n\n"
    "İstediğiniz kodu oluşturuyorum. Lütfen bekleyin...\n\n"
    "```python\n"
    "# Mock mod: gerçek kod üretimi için MOCK_MODE=false gereklidir\n"
    "print('Merhaba, ENTRAPEER!')\n"
    "```\n\n"
    "**🔧 Sonraki Adımlar:**\n\n"
    "1. Kodu çalıştırın ve çıktıyı inceleyin."
)

# ---------------------------------------------------------------------------
# MOCK_MODE: 4-label keyword classifier (CI / tests — no LLM call)
# ---------------------------------------------------------------------------
_MOCK_CODE_LANGS = ("python", "javascript", "js", "typescript", "sql", "bash", "html")
_MOCK_CODE_ACTIONS = ("yaz", "yap", "oluştur", "geliştir", "write", "create", "build", "üret")
_MOCK_CODE_CONCEPTS = (
    "yazı tura", "tic tac toe", "xox", "hesap makinesi",
    "fibonacci", "dosya okuma", "todo list", "web scraper",
)
_MOCK_OOS_STEMS = (
    "pizza", "dondurma", "kebap", "burger", "tarifi",
    "nasılsın", "naber", "günaydın", "film öner",
    "hava durumu", "futbol", "spor haberi", "netflix",
    "restoran öner", "kafe öner", "lokanta öner",
    "ne yesem", "ne yiyeyim", "yemek öner",
)
_MOCK_BIZ_SIGNALS = (
    "satış", "maliyet", "müşteri", "pazar", "sektör",
    "kriz", "analiz", "rakip", "büyüme", "strateji", "lojistik",
)
_MOCK_CRISIS_SIGNALS = (
    "satışlarım", "satışlarımız", "ekibimiz", "şirketimiz", "maliyetimiz",
    "hedefimiz", "düştü", "arttı", "kaybettik", "tutturamıyoruz",
    "churn", "devir oranı", "aksaklık",
)


def _mock_classify(message: str) -> str:
    """Lightweight 4-label classifier for MOCK_MODE (no LLM, no regex)."""
    k = message.lower()

    # 1. Code detection
    has_lang = any(lang in k for lang in _MOCK_CODE_LANGS)
    has_action = any(action in k for action in _MOCK_CODE_ACTIONS)
    has_concept = any(concept in k for concept in _MOCK_CODE_CONCEPTS)
    if (has_lang and has_action) or has_concept:
        return "CODE_REQUEST"

    # 2. OOS detection (business signal overrides)
    has_oos = any(stem in k for stem in _MOCK_OOS_STEMS)
    has_biz = any(sig in k for sig in _MOCK_BIZ_SIGNALS)
    if has_oos and not has_biz:
        return "OUT_OF_SCOPE"

    # 3. Business crisis vs direct answer
    has_crisis = any(sig in k for sig in _MOCK_CRISIS_SIGNALS)
    has_search = any(sig in k for sig in _MOCK_BIZ_SIGNALS)
    if has_crisis:
        return "BUSINESS_CRISIS"
    if has_search:
        return "DIRECT_ANSWER"

    return "BUSINESS_CRISIS"  # safe default


def _mock_search_response(message: str) -> str:
    """Return a sector-specific mock market analysis based on keywords."""
    lower = message.lower()
    for keywords, response in _SEARCH_TEMPLATES:
        if any(kw in lower for kw in keywords):
            return response
    return _SEARCH_GENERIC


# ---------------------------------------------------------------------------
# Live mode: LLM classify
# ---------------------------------------------------------------------------
async def _classify_live(user_message: str) -> str:
    """
    Call GPT-4o and parse one of:
    BUSINESS_CRISIS | DIRECT_ANSWER | CODE_REQUEST | OUT_OF_SCOPE
    """
    response = await _get_llm().ainvoke(
        [
            SystemMessage(content=_ROUTER_SYSTEM),
            HumanMessage(content=user_message),
        ]
    )
    content = response.content.strip().upper()
    logger.info("PEER raw LLM response: %r", content[:80])

    _VALID = {"BUSINESS_CRISIS", "DIRECT_ANSWER", "CODE_REQUEST", "OUT_OF_SCOPE"}
    for token in content.split():
        clean = token.strip(".,;:()")
        if clean in _VALID:
            return clean

    logger.warning("Unexpected routing label %r — defaulting to DIRECT_ANSWER", content[:40])
    return "DIRECT_ANSWER"


# ---------------------------------------------------------------------------
# Helper: format CodeTemplate as Markdown
# ---------------------------------------------------------------------------
def _format_code_response(tmpl) -> str:
    lang_lower = tmpl.language.lower()
    snippet = tmpl.snippet.strip()
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(tmpl.next_steps, 1))
    parts = [
        f"## {tmpl.title}",
        "",
        f"**Dil:** `{tmpl.language}`",
        "",
        tmpl.description,
        "",
        f"```{lang_lower}",
        snippet,
        "```",
        "",
        "**🔧 Sonraki Adımlar:**",
        "",
        steps,
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helper: extract last human message text from state
# ---------------------------------------------------------------------------
def _last_user_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
async def peer_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node — the single entry point for all routing and guardrail logic.

    Sets `current_step` to one of:
      "discovery"   → forward to DiscoveryAgent (BUSINESS_CRISIS)
      "completed"   → DIRECT_ANSWER or CODE_REQUEST handled, graph ends
      "rejected"    → OUT_OF_SCOPE, rejection message written, graph ends
    """
    user_message = _last_user_message(state)
    messages: list = list(state["messages"])

    if not user_message:
        messages.append({"role": "assistant", "content": REJECTION_MSG})
        return {**state, "messages": messages, "current_step": "rejected"}

    # ── Classify ─────────────────────────────────────────────────────────
    if MOCK_MODE:
        route = _mock_classify(user_message)
        logger.info("[MOCK] PEER %r → %s", user_message[:60], route)
    else:
        route = await _classify_live(user_message)
        logger.info("PEER %r → %s", user_message[:60], route)

    # ── Route 1: OUT_OF_SCOPE — reject immediately ────────────────────────
    if route == "OUT_OF_SCOPE":
        messages.append({"role": "assistant", "content": REJECTION_MSG})
        return {**state, "messages": messages, "current_step": "rejected"}

    # ── Route 2: CODE_REQUEST — CodeAgent → END ───────────────────────────
    if route == "CODE_REQUEST":
        t0 = time.monotonic()
        try:
            thread_id = state.get("thread_id", "unknown")
            if MOCK_MODE:
                code_content = _CODE_MOCK_RESPONSE
                structured_out: dict = {"response_type": "code"}
            else:
                tmpl = await generate_code_from_task(user_message, thread_id=thread_id)
                duration_ms = int((time.monotonic() - t0) * 1000)
                await log_agent_run(
                    agent_name="CodeAgent",
                    thread_id=thread_id,
                    user_id=None,
                    input_data={"task": user_message},
                    output_data=tmpl.model_dump(),
                    duration_ms=duration_ms,
                )
                code_content = _format_code_response(tmpl)
                structured_out = {"response_type": "code", **tmpl.model_dump()}

            messages.append({"role": "assistant", "content": code_content})
            return {
                **state,
                "messages": messages,
                "current_step": "completed",
                "structured_problem": structured_out,
            }
        except Exception as exc:
            logger.warning("PEER CodeAgent failed: %s — falling back to DIRECT_ANSWER", exc)
            # Graceful degradation: treat as direct answer
            route = "DIRECT_ANSWER"

    # ── Route 3: DIRECT_ANSWER — search + synthesise → END ───────────────
    if route == "DIRECT_ANSWER":
        if MOCK_MODE:
            answer = _mock_search_response(user_message)
        else:
            search_results = await internet_search(user_message, max_results=5)
            synthesis_prompt = (
                f"Kullanıcı sorusu:\n{user_message}\n\n"
                f"İnternet araştırma sonuçları:\n{search_results}"
            )
            response = await _get_llm().ainvoke(
                [
                    SystemMessage(content=_SEARCH_SYNTHESIZER_SYSTEM),
                    HumanMessage(content=synthesis_prompt),
                ]
            )
            answer = response.content

        messages.append({"role": "assistant", "content": answer})
        return {**state, "messages": messages, "current_step": "completed"}

    # ── Route 4: BUSINESS_CRISIS — hand off to DiscoveryAgent ────────────
    messages.append({"role": "assistant", "content": _DISCOVERY_TRANSITION})
    return {**state, "messages": messages, "current_step": "discovery"}
