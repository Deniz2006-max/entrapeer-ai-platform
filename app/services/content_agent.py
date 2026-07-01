"""
ContentAgent – Sub-Agent #1
Converts a completed ProblemTree into a structured, executive-ready
Action Plan Report (Markdown).
"""

import logging
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ActionPlanReport(BaseModel):
    executive_summary: str = Field(
        ...,
        description=(
            "2-3 sentence C-level summary: what the core problem is, "
            "why it matters, and what the recommended approach is."
        ),
    )
    immediate_actions: List[str] = Field(
        ...,
        description="2-4 concrete actions to take within the next 0-30 days.",
    )
    short_term_actions: List[str] = Field(
        ...,
        description="2-4 strategic initiatives to execute within 1-3 months.",
    )
    long_term_actions: List[str] = Field(
        ...,
        description="2-3 structural / systemic changes to embed within 3-12 months.",
    )
    success_metrics: List[str] = Field(
        ...,
        description=(
            "3-5 measurable KPIs or milestones to track whether the plan is working."
        ),
    )


# ---------------------------------------------------------------------------
# LLM – lazy init
# ---------------------------------------------------------------------------
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0.3).with_structured_output(
            ActionPlanReport
        )
    return _llm


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_SYSTEM = """Sen ENTRAPEER'in ContentAgent'ısın — kurumsal strateji danışmanlığı uzmanı.
Görevin: sana verilen Problem Ağacı analizini (ProblemTree) temel alarak
şirkete özel, uygulanabilir ve önceliklendirilmiş bir Aksiyon Planı Raporu üretmek.

━━━ ÇIKTI KURALLARI ━━━
• executive_summary: 2-3 cümlelik yönetici özeti. Net, abartısız, karar vericilere hitap eden.
• immediate_actions (0-30 gün): Kriz etkisini sınırlandıracak hızlı kazanımlar.
• short_term_actions (1-3 ay): Kök nedenleri hedef alan yapısal adımlar.
• long_term_actions (3-12 ay): Kalıcı çözüm için sistem/kültür değişiklikleri.
• success_metrics: Her eylem setini izleyecek SMART KPI'lar.

Tüm öneriler ProblemTree'deki kök nedenlerle doğrudan ilişkili olmalı.
Genel tavsiyeler yasak — sadece bu şirkete / bu krize özgü öneriler yap."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_action_plan(
    problem_tree: dict,
    thread_id: str = "",
) -> ActionPlanReport:
    """
    Generate an ActionPlanReport from a serialised ProblemTree dict.
    Raises on LLM failure (caller should catch and handle gracefully).
    """
    # Format problem tree as readable context for the LLM
    causes_text = ""
    for i, rc in enumerate(problem_tree.get("root_causes", []), 1):
        if isinstance(rc, dict):
            main = rc.get("main_cause", "")
            subs = rc.get("sub_causes", [])
            causes_text += f"\n{i}. {main}"
            for s in subs:
                causes_text += f"\n   - {s}"
        else:
            causes_text += f"\n{i}. {rc}"

    user_prompt = (
        f"**Problem Tipi:** {problem_tree.get('problem_type', 'Hybrid')}\n"
        f"**Sektör:** {problem_tree.get('industry', 'Belirtilmemiş')}\n"
        f"**Ana Problem:** {problem_tree.get('main_problem', '')}\n\n"
        f"**Kök Nedenler:**{causes_text}\n\n"
        f"**Güven Skoru:** {problem_tree.get('confidence_score', 0):.0%}\n\n"
        "Bu ProblemTree verilerinden şirkete özel bir Aksiyon Planı Raporu üret."
    )

    logger.info(
        "ContentAgent generating action plan thread_id=%s problem_type=%s",
        thread_id,
        problem_tree.get("problem_type"),
    )

    result: ActionPlanReport = await _get_llm().ainvoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )

    logger.info("ContentAgent completed thread_id=%s", thread_id)
    return result
