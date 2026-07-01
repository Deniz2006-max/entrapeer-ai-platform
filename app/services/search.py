import asyncio
import logging
from typing import Optional

from ddgs import DDGS

logger = logging.getLogger(__name__)

_FALLBACK_MESSAGE = (
    "Arama motoru şu anda yanıt vermiyor, ancak mevcut bilgilerime göre "
    "bu konu hakkında genel bir değerlendirme yapabilirim."
)


def _sync_search(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo text search and return a clean summary string."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return "Arama sonucu bulunamadı."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        href = r.get("href", "").strip()
        lines.append(f"{i}. {title}\n   {body}\n   Kaynak: {href}")

    return "\n\n".join(lines)


async def internet_search(query: str, max_results: int = 5) -> str:
    """
    Perform an async internet search via DuckDuckGo.

    Returns a formatted summary of the top results, or a safe fallback
    message if the search engine is unavailable.
    """
    try:
        result = await asyncio.to_thread(_sync_search, query, max_results)
        return result
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for query %r: %s", query, exc)
        return _FALLBACK_MESSAGE
