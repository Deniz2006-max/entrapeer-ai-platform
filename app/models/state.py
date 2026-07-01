from typing import List, TypedDict


class AgentState(TypedDict):
    messages: List[dict]
    current_step: str
    user_profile: dict
    structured_problem: dict
    # ── Sequential interview fields ──────────────────────────────────────
    # interview_turns  : number of questions already ASKED by the Discovery Agent
    #                    (0 = no question asked yet, 4 = interview complete)
    # current_question : the most recent question the agent posed to the user
    # interview_history: ordered list of {question, answer} pairs collected so far
    interview_turns: int
    current_question: str
    interview_history: List[dict]
    # ── Discovery phase synthesis (populated at end of interview) ────────────
    # Produced by discovery_synthesis_node when interview_turns reaches 4.
    # Contains the 4 mandatory hand-off fields:
    #   customer_stated_problem, identified_business_problem,
    #   hidden_root_risk, customer_chat_summary
    discovery_summary: dict
    # ── Context-switch confirmation (set when user proposes a new crisis mid-interview)
    # Holds the raw text of the proposed new crisis while awaiting user confirmation.
    # Cleared to "" after the user confirms (Evet) or declines (Hayır).
    pending_new_crisis: str
