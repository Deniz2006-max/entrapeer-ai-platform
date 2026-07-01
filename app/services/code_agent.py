"""
CodeAgent – Sub-Agent #2

Two entry-points:
  1. generate_code_template(problem_tree)  – activated after Structuring when
     problem_type ∈ {"Technology", "Hybrid"}.  Takes a full ProblemTree dict.
  2. generate_code_from_task(task)         – activated directly from the router
     when the user sends a standalone coding request ("write me a Python …").
     Takes a raw natural-language task string; no ProblemTree needed.
"""

import logging
import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() == "true"

# Problem types that trigger the CodeAgent via the post-structuring path
TECH_PROBLEM_TYPES = {"Technology", "Hybrid"}

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class CodeTemplate(BaseModel):
    language: Literal["Python", "TypeScript", "SQL", "Bash", "YAML", "Other"] = Field(
        ..., description="Programming / scripting language best suited to the problem."
    )
    title: str = Field(..., description="Short title describing what the snippet does.")
    description: str = Field(
        ...,
        description=(
            "2-3 sentence plain-language explanation of what the snippet solves "
            "and how it maps to the identified root cause."
        ),
    )
    snippet: str = Field(
        ...,
        description=(
            "A minimal, production-quality code template (20-60 lines). "
            "Use realistic variable names. Include inline comments."
        ),
    )
    next_steps: list[str] = Field(
        ...,
        description="2-3 concrete next steps to productionise this template.",
    )


# ---------------------------------------------------------------------------
# LLM – lazy init
# ---------------------------------------------------------------------------
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0.2).with_structured_output(
            CodeTemplate
        )
    return _llm


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Used by generate_code_template (post-Structuring / ProblemTree path)
_SYSTEM = """Sen ENTRAPEER'in CodeAgent'ısın — kurumsal teknoloji danışmanı ve kıdemli yazılım mühendisi.
Görevin: sana verilen teknoloji odaklı ProblemTree kök nedenine karşılık
gerçek dünyada uygulanabilir, minimal ama anlamlı bir kod/script şablonu üretmek.

━━━ DİL KURALI (KESİN) ━━━
Tüm çıktılar — title, description, snippet içindeki yorumlar ve next_steps —
TAMAMEN TÜRKÇE olmalıdır. Hiçbir şekilde İngilizce açıklama veya yorum kullanma.

━━━ ÇIKTI KURALLARI ━━━
• language: En uygun dili seç (Python, TypeScript, SQL, Bash, YAML vb.)
• title: "Ne yapan bir şablon" formatında 5-10 kelimelik TÜRKÇE başlık.
• description: Kodun hangi kök nedeni nasıl çözdüğünü 2-3 cümleyle Türkçe açıkla.
• snippet: 20-60 satır, çalışabilir kod. Satır içi Türkçe yorumlar kullan.
  Placeholder'ları büyük harfle belirt: API_ANAHTARI, SUNUCU_ADRESİ vb.
• next_steps: Bu şablonu production'a taşımak için 2-3 somut Türkçe adım.

Örnek: izleme eksikliği → Prometheus metrik toplayıcı şablonu
Örnek: veri silosu → ETL boru hattı şablonu
Örnek: CI/CD yokluğu → GitHub Actions iş akışı YAML şablonu"""

# Used by generate_code_from_task (direct coding request path)
_DIRECT_SYSTEM = """Sen yetenekli ve pratik bir yazılım mühendisisin.
Kullanıcı senden doğrudan bir kod yazmanı istedi; isteği eksiksiz ve çalışır halde uygula.

━━━ DİL KURALI (KESİN) ━━━
Tüm çıktılar — title, description, kod içi yorum satırları ve next_steps —
TAMAMEN TÜRKÇE olmalıdır. "# comment" değil "# yorum" yaz; değişken ve fonksiyon
isimleri de Türkçe veya Türkçe okunuşuna uygun olabilir.
Hiçbir zaman İngilizce açıklama veya satır içi yorum kullanma.

━━━ ÇIKTI KURALLARI ━━━
• language: İstekte belirtilmişse onu kullan, belirtilmemişse en uygun dili seç.
• title: İsteği özetleyen 5-8 kelimelik TÜRKÇE başlık.
• description: Kodun ne yaptığını ve nasıl çalıştığını 2-3 cümleyle Türkçe açıkla.
• snippet: Tam çalışır, temiz, Türkçe yorumlu kod (15-80 satır).
  — Gerekli kütüphaneleri üst kısımda import et.
  — Her önemli adım için Türkçe satır içi yorum ekle.
  — Hata durumlarını try-except ile yönet.
• next_steps: Kodu geliştirmek veya production'a taşımak için 2-3 somut Türkçe adım.

Kural: İşlevsel, doğrudan çalışabilir, gereksiz İngilizce içermeyen kod üret."""


# ---------------------------------------------------------------------------
# MOCK_MODE – Senaryo tabanlı Türkçe şablonlar
# ---------------------------------------------------------------------------

_MOCK_YAZI_TURA = CodeTemplate(
    language="Python",
    title="Yazı Tura Oyunu – Döngülü Simülasyon",
    description=(
        "Kullanıcıdan yazı veya tura tahmini alan, rastgele sonuç üreten ve "
        "kazanma/kaybetme istatistiklerini takip eden interaktif bir Python oyunu. "
        "Döngü, kullanıcı 'q' yazana kadar devam eder."
    ),
    snippet="""\
import random

def yazi_tura_oyunu():
    \"\"\"Yazı tura simülasyonu – istatistik takipli.\"\"\"
    kazanma = 0
    kaybetme = 0

    print("╔══════════════════════════╗")
    print("║   YAZI TURA OYUNU        ║")
    print("╚══════════════════════════╝")
    print("Çıkmak için 'q' yazın.\\n")

    while True:
        tahmin = input("Tahmininiz (yazı / tura): ").strip().lower()

        # Çıkış kontrolü
        if tahmin == "q":
            break

        # Geçerli giriş kontrolü
        if tahmin not in ("yazı", "tura"):
            print("⚠ Geçersiz giriş. Lütfen 'yazı' veya 'tura' yazın.\\n")
            continue

        # Rastgele sonuç üret
        sonuc = random.choice(["yazı", "tura"])
        print(f"🪙 Sonuç: {sonuc.upper()}")

        if tahmin == sonuc:
            kazanma += 1
            print("✅ Kazandınız!\\n")
        else:
            kaybetme += 1
            print("❌ Kaybettiniz!\\n")

    # Oyun sonu özeti
    toplam = kazanma + kaybetme
    oran = (kazanma / toplam * 100) if toplam > 0 else 0
    print(f"\\n📊 Oyun Bitti")
    print(f"   Kazanma : {kazanma} / Kaybetme : {kaybetme}")
    print(f"   Kazanma Oranı: %{oran:.1f}")

if __name__ == "__main__":
    yazi_tura_oyunu()
""",
    next_steps=[
        "Sonuçları JSON veya CSV dosyasına kaydedin (tarih damgalı log).",
        "Birden fazla oyuncu modunu destekleyecek şekilde genişletin.",
        "Tkinter veya Rich kütüphanesiyle görsel arayüz ekleyin.",
    ],
)

_MOCK_TIC_TAC_TOE = CodeTemplate(
    language="Python",
    title="XOX (Tic-Tac-Toe) İki Oyunculu Terminal Oyunu",
    description=(
        "İki oyuncunun sırayla X ve O işaretlediği, 3×3 tahtada kazananı veya "
        "beraberi otomatik tespit eden bir terminal XOX oyunu. "
        "Her hamle sonrası tahta ekrana çizilir; kazanan veya beraberlik durumunda oyun biter."
    ),
    snippet="""\
def tahta_olustur():
    \"\"\"3x3 boş tahta döndürür.\"\"\"
    return [[\" \" for _ in range(3)] for _ in range(3)]

def tahta_goster(tahta):
    \"\"\"Tahtayı terminale çizer.\"\"\"
    print()
    for i, satir in enumerate(tahta):
        print(" | ".join(satir))
        if i < 2:
            print("---------")
    print()

def kazanan_kontrol(tahta, isaretci):
    \"\"\"Verilen işaretçi kazandı mı kontrol eder.\"\"\"
    # Satır ve sütun kontrolü
    for i in range(3):
        if all(tahta[i][j] == isaretci for j in range(3)):
            return True
        if all(tahta[j][i] == isaretci for j in range(3)):
            return True
    # Köşegen kontrolü
    if all(tahta[i][i] == isaretci for i in range(3)):
        return True
    if all(tahta[i][2 - i] == isaretci for i in range(3)):
        return True
    return False

def dolu_mu(tahta):
    \"\"\"Tahta tamamen doldu mu?\"\"\"
    return all(tahta[i][j] != \" \" for i in range(3) for j in range(3))

def oyunu_oyna():
    tahta = tahta_olustur()
    oyuncular = [\"X\", \"O\"]
    tur = 0

    print("=== XOX OYUNU ===")
    print("Konum girerken satır ve sütunu 0-2 arasında yazın.\\n")

    while True:
        mevcut = oyuncular[tur % 2]
        tahta_goster(tahta)
        print(f"Oyuncu {mevcut} hamlesi:")

        try:
            satir = int(input("  Satır (0-2): "))
            sutun = int(input("  Sütun (0-2): "))
        except ValueError:
            print("⚠ Geçersiz giriş, tekrar deneyin.\\n")
            continue

        # Hücre dolu mu?
        if not (0 <= satir <= 2 and 0 <= sutun <= 2):
            print("⚠ 0-2 arası değer girin.\\n")
            continue
        if tahta[satir][sutun] != \" \":
            print("⚠ Bu hücre dolu, başka seçin.\\n")
            continue

        tahta[satir][sutun] = mevcut
        tur += 1

        if kazanan_kontrol(tahta, mevcut):
            tahta_goster(tahta)
            print(f"🏆 Oyuncu {mevcut} kazandı!")
            break
        if dolu_mu(tahta):
            tahta_goster(tahta)
            print("🤝 Beraberlik!")
            break

if __name__ == "__main__":
    oyunu_oyna()
""",
    next_steps=[
        "Bilgisayara karşı oynama için minimax algoritması ekleyin.",
        "Kazanma sayacı ve oyun geçmişi için bir sınıf (OyunYöneticisi) tasarlayın.",
        "Pygame ile görsel arayüze taşıyın.",
    ],
)

_MOCK_DOSYA_IO = CodeTemplate(
    language="Python",
    title="UTF-8 Destekli Güvenli Dosya Okuma ve Yazma",
    description=(
        "Türkçe karakter desteği olan, hata yönetimiyle donatılmış dosya okuma/yazma "
        "yardımcı fonksiyonları. Her işlem try-except bloğuyla sarılmıştır; dosya bulunamadı, "
        "yetki hatası veya kodlama hatası durumlarını ayrı ayrı yakalar ve anlaşılır Türkçe hata mesajları verir."
    ),
    snippet="""\
import os
from pathlib import Path

def dosya_oku(dosya_yolu: str, kodlama: str = "utf-8") -> str | None:
    \"\"\"
    Belirtilen dosyayı okur ve içeriği döndürür.
    Hata durumunda None döner, mesajı terminale yazar.
    \"\"\"
    try:
        yol = Path(dosya_yolu)

        # Dosyanın var olup olmadığını kontrol et
        if not yol.exists():
            print(f"❌ Hata: '{dosya_yolu}' dosyası bulunamadı.")
            return None

        # Dosyayı belirtilen kodlamayla oku
        with yol.open(encoding=kodlama) as dosya:
            icerik = dosya.read()

        print(f"✅ '{yol.name}' başarıyla okundu ({len(icerik)} karakter).")
        return icerik

    except PermissionError:
        print(f"❌ Yetki Hatası: '{dosya_yolu}' dosyasına erişim izniniz yok.")
    except UnicodeDecodeError:
        print(f"❌ Kodlama Hatası: Dosya {kodlama!r} ile okunamadı. 'latin-1' deneyin.")
    except OSError as hata:
        print(f"❌ İşletim Sistemi Hatası: {hata}")
    return None


def dosya_yaz(dosya_yolu: str, icerik: str, kodlama: str = "utf-8",
              ustune_yaz: bool = False) -> bool:
    \"\"\"
    Verilen içeriği dosyaya yazar.
    ustune_yaz=False ise var olan dosyanın sonuna ekler.
    Başarılıysa True, hata durumunda False döner.
    \"\"\"
    try:
        yol = Path(dosya_yolu)

        # Üst klasörü yoksa oluştur
        yol.parent.mkdir(parents=True, exist_ok=True)

        # Yazma modu: üzerine yaz (w) veya sonuna ekle (a)
        mod = "w" if ustune_yaz else "a"
        with yol.open(mod, encoding=kodlama) as dosya:
            dosya.write(icerik)

        boyut = yol.stat().st_size
        print(f"✅ '{yol.name}' başarıyla yazıldı ({boyut} bayt).")
        return True

    except PermissionError:
        print(f"❌ Yetki Hatası: '{dosya_yolu}' dosyasına yazma izniniz yok.")
    except OSError as hata:
        print(f"❌ İşletim Sistemi Hatası: {hata}")
    return False


# ── Kullanım Örneği ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Dosyaya yaz
    dosya_yaz("ornek/veri.txt", "Merhaba, Dünya!\\n", ustune_yaz=True)
    dosya_yaz("ornek/veri.txt", "İkinci satır eklendi.\\n")

    # Dosyayı oku
    icerik = dosya_oku("ornek/veri.txt")
    if icerik:
        print("\\n--- Dosya İçeriği ---")
        print(icerik)
""",
    next_steps=[
        "Büyük dosyalar için `for satir in dosya:` döngüsüyle satır satır okuma ekleyin.",
        "JSON veya CSV formatı için `json` veya `csv` modüllerine geçin.",
        "Loglama için `logging` modülünü entegre edin (print yerine).",
    ],
)

# Genel amaçlı yedek şablon (senaryo eşleşmediğinde kullanılır)
_MOCK_GENEL = CodeTemplate(
    language="Python",
    title="Genel Amaçlı Python Başlangıç Şablonu",
    description=(
        "Hata yönetimi, kullanıcı girdisi ve temel işlem döngüsü içeren "
        "sade bir Python başlangıç iskelet kodu. "
        "İstediğiniz uygulamayı bu şablon üzerine inşa edebilirsiniz."
    ),
    snippet="""\
import sys

def islemi_gerceklestir(girdi: str) -> str:
    \"\"\"
    Ana işlem fonksiyonu.
    Gerekli mantığı buraya ekleyin.
    \"\"\"
    # TODO: gerçek işlem kodunu buraya yazın
    return f"İşlenen veri: {girdi.strip().upper()}"

def ana_dongu():
    \"\"\"Programın ana giriş noktası.\"\"\"
    print("=== Program Başladı ===")
    print("Çıkmak için 'q' veya Ctrl+C tuşlayın.\\n")

    while True:
        try:
            girdi = input("Girdi: ").strip()

            if girdi.lower() == "q":
                print("\\nProgram sonlandırılıyor...")
                break

            if not girdi:
                print("⚠ Boş girdi. Lütfen bir değer yazın.")
                continue

            # İşlemi gerçekleştir ve sonucu göster
            sonuc = islemi_gerceklestir(girdi)
            print(f"✅ Sonuç: {sonuc}\\n")

        except KeyboardInterrupt:
            print("\\n\\nKullanıcı tarafından durduruldu.")
            sys.exit(0)
        except Exception as hata:
            # Beklenmeyen hatalar için güvenli yakalama
            print(f"❌ Beklenmeyen hata: {hata}\\n")

if __name__ == "__main__":
    ana_dongu()
""",
    next_steps=[
        "`islemi_gerceklestir()` fonksiyonunu asıl iş mantığınızla doldurun.",
        "Argüman desteği için `argparse` modülünü ekleyin.",
        "Birim testleri için `tests/test_isleme.py` dosyası oluşturun.",
    ],
)


def _senaryo_sec(task: str) -> CodeTemplate:
    """
    Kullanıcının görev metnine göre en uygun MOCK şablonunu döndürür.
    Gerçek LLM çağrısı yapmadan (MOCK_MODE=true) hızlı ve alakalı yanıt verir.
    """
    k = task.lower()

    # XOX / Tic-Tac-Toe senaryosu
    if any(s in k for s in ("tic", "tac", "toe", "xox", "x-o", "x o", "çarpı sıfır")):
        return _MOCK_TIC_TAC_TOE

    # Dosya okuma / yazma senaryosu
    if any(s in k for s in ("dosya", "file", "okuma", "yazma", "read", "write",
                             "txt", "csv", "json oku", "kaydet")):
        return _MOCK_DOSYA_IO

    # Yazı tura senaryosu (varsayılan oyun şablonu)
    if any(s in k for s in ("yazı tura", "coin", "flip", "tura", "yazı")):
        return _MOCK_YAZI_TURA

    # Genel Python başlangıcı
    return _MOCK_GENEL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_code_template(
    problem_tree: dict,
    thread_id: str = "",
) -> CodeTemplate | None:
    """
    Generate a CodeTemplate only if the problem is technology-related.
    Returns None if not applicable.
    """
    problem_type = problem_tree.get("problem_type", "")
    if problem_type not in TECH_PROBLEM_TYPES:
        logger.info(
            "CodeAgent skipped – problem_type=%s is not technology-related thread_id=%s",
            problem_type,
            thread_id,
        )
        return None

    causes_text = ""
    for i, rc in enumerate(problem_tree.get("root_causes", []), 1):
        if isinstance(rc, dict):
            causes_text += f"\n{i}. {rc.get('main_cause', '')}"
            for s in rc.get("sub_causes", []):
                causes_text += f"\n   - {s}"
        else:
            causes_text += f"\n{i}. {rc}"

    user_prompt = (
        f"**Problem Tipi:** {problem_type}\n"
        f"**Sektör:** {problem_tree.get('industry', 'Belirtilmemiş')}\n"
        f"**Ana Problem:** {problem_tree.get('main_problem', '')}\n\n"
        f"**Teknolojik Kök Nedenler:**{causes_text}\n\n"
        "Bu teknolojik sorunu çözmeye yönelik bir kod/script şablonu üret."
    )

    logger.info(
        "CodeAgent generating code template thread_id=%s problem_type=%s",
        thread_id,
        problem_type,
    )

    result: CodeTemplate = await _get_llm().ainvoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )

    logger.info(
        "CodeAgent completed thread_id=%s language=%s", thread_id, result.language
    )
    return result


async def generate_code_from_task(
    task: str,
    thread_id: str = "",
) -> CodeTemplate:
    """
    Generate a CodeTemplate directly from a natural-language coding request.

    Unlike `generate_code_template` (which requires a ProblemTree), this function
    takes a raw user task string and produces working code immediately.

    Used by the router when the Peer-Agent detects a direct coding intent
    (e.g. "write me a Python coin-flip game") so those requests are served
    without going through the full Discovery → Structuring pipeline.
    """
    if MOCK_MODE:
        secilen = _senaryo_sec(task)
        logger.info(
            "CodeAgent [MOCK] generate_code_from_task thread_id=%s task=%r template=%r",
            thread_id,
            task[:60],
            secilen.title,
        )
        return secilen

    logger.info(
        "CodeAgent generating code from task thread_id=%s task=%r",
        thread_id,
        task[:60],
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2).with_structured_output(
        CodeTemplate
    )

    result: CodeTemplate = await llm.ainvoke(
        [
            SystemMessage(content=_DIRECT_SYSTEM),
            HumanMessage(content=f"Görev: {task}"),
        ]
    )

    logger.info(
        "CodeAgent generate_code_from_task done thread_id=%s language=%s",
        thread_id,
        result.language,
    )
    return result
