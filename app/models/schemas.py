from typing import List, Literal

from pydantic import BaseModel, Field


class RootCause(BaseModel):
    main_cause: str = Field(..., description="Primary root cause stated as a 'Why…' hypothesis")
    sub_causes: List[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description=(
            "Exactly 2 or 3 observable, evidence-based sub-causes that feed into "
            "the main cause. Must not be fewer than 2 or more than 3."
        ),
    )


class ProblemTree(BaseModel):
    problem_type: Literal[
        "Growth", "Cost", "Operational", "Technology",
        "Regulation", "Organizational", "Hybrid"
    ] = Field(
        ...,
        description=(
            "High-level category — must be exactly one of the 7 allowed values: "
            "Growth | Cost | Operational | Technology | Regulation | Organizational | Hybrid"
        ),
    )
    main_problem: str = Field(..., description="The core problem identified in one clear sentence")
    industry: str = Field(..., description="Industry or domain of the problem")
    root_causes: List[RootCause] = Field(
        ...,
        min_length=3,
        max_length=5,
        description=(
            "Between 3 and 5 main root causes (inclusive). Each must have exactly "
            "2-3 sub_causes. Fewer than 3 or more than 5 root causes is invalid."
        ),
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence (0–1)")


class DiscoveryQuestions(BaseModel):
    questions: List[str] = Field(..., description="Questions to uncover deeper problem context")
    rationale: str = Field(..., description="Reasoning behind the selected questions")


class DiscoveryQuestion(BaseModel):
    """Single-question structured output for the turn-by-turn interview."""

    question: str = Field(
        ...,
        description=(
            "Exactly ONE strategic open-ended question that deepens the business "
            "crisis interview. Must NOT contain more than one question mark."
        ),
    )


class InterviewEntry(BaseModel):
    """A single question-answer pair from the turn-by-turn interview."""

    question: str = Field(..., description="The question posed by the Discovery Agent")
    answer: str = Field(..., description="The user's verbatim answer")


class DiscoverySummary(BaseModel):
    """
    Structured synthesis produced by the Discovery Agent at the END of the
    turn-by-turn interview (after interview_turns reaches 4).

    These 4 fields are mandatory hand-off data to the Structuring Agent and
    must be persisted to MongoDB (analysis_reports / agent_logs).
    """

    customer_stated_problem: str = Field(
        ...,
        description=(
            "The problem as the customer literally described it across the interview. "
            "Summarise in their own words without interpretation or analysis."
        ),
    )
    identified_business_problem: str = Field(
        ...,
        description=(
            "The REAL business problem identified from the interview data. "
            "May differ significantly from the customer's stated problem. "
            "State as a single clear root-cause sentence."
        ),
    )
    hidden_root_risk: str = Field(
        ...,
        description=(
            "A structural risk or threat NOT explicitly mentioned by the customer "
            "but inferable from their answers — typically an organisational, "
            "market, leadership, or financial fragility underlying the crisis."
        ),
    )
    customer_chat_summary: str = Field(
        ...,
        description=(
            "A 3-5 sentence summary of the full interview: which questions were asked, "
            "what quantitative/qualitative data the customer shared, and what "
            "organisational context emerged."
        ),
    )


class AnalysisReport(BaseModel):
    """
    Full MongoDB document written by the Structuring Agent after the interview
    completes.  This is the single source of truth for a finished analysis.

    Layout mirrors `updated_structured_problem` in structuring.py — both must
    stay in sync.  Business teams read this object from the `analyses` collection.
    """

    # ── ProblemTree fields (LLM-generated) ───────────────────────────────
    problem_type: str = Field(..., description="High-level business problem category")
    main_problem: str = Field(..., description="Core problem in one clear sentence")
    industry: str = Field(..., description="Industry / domain label")
    root_causes: List[dict] = Field(
        ..., description="Hierarchical root causes (main_cause + sub_causes[])"
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0)

    # ── Discovery Agent hand-off (synthesis) ─────────────────────────────
    discovery_summary: DiscoverySummary = Field(
        ...,
        description="4 mandatory fields produced by discovery_synthesis_node",
    )

    # ── Raw interview transcript ──────────────────────────────────────────
    interview_history: List[InterviewEntry] = Field(
        default_factory=list,
        description=(
            "Ordered list of every question-answer pair from the turn-by-turn "
            "interview.  Enables business teams to audit the full conversation "
            "that led to this analysis."
        ),
    )
