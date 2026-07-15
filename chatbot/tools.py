"""
chatbot/tools.py

Custom LangChain tools for the Credit Risk AI Assistant.

Responsibilities
----------------
- Maintain the live applicant state supplied by Streamlit.[cite: 3]
- Search the FAISS knowledge base.[cite: 3]
- Provide structured tool outputs for the LangGraph agent.[cite: 3]
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain.tools import tool
from pydantic import BaseModel, Field

from chatbot.vector_store import get_retriever

logger = logging.getLogger(__name__)

# ==============================================================================
# Live Applicant State[cite: 3]
# ==============================================================================

LIVE_APPLICANT_STATE: Dict[str, Any] = {}


def update_live_applicant(prediction_result=None) -> None:
    """
    Updates the in-memory applicant state from the latest Streamlit prediction.[cite: 3]
    """

    global LIVE_APPLICANT_STATE

    if prediction_result is None:
        LIVE_APPLICANT_STATE = {}
        return

    LIVE_APPLICANT_STATE = {
        "credit_score": prediction_result.score,
        "probability_of_default": f"{prediction_result.probability:.2%}",
        "risk_band": prediction_result.risk_band,
        "decision": prediction_result.decision,
    }

    logger.info("Updated live applicant state.[cite: 3]")


# ==============================================================================
# Applicant Tool[cite: 3]
# ==============================================================================

class EmptyInputSchema(BaseModel):
    pass


@tool("get_current_applicant_data", args_schema=EmptyInputSchema)
def get_current_applicant_data() -> str:
    """
    Returns the current applicant's prediction.[cite: 3]

    Use this tool ONLY when the user asks about:[cite: 3]
    - current applicant[cite: 3]
    - applicant score[cite: 3]
    - decision[cite: 3]
    - probability of default[cite: 3]
    - risk band[cite: 3]
    """

    logger.info("Applicant tool invoked.[cite: 3]")

    if not LIVE_APPLICANT_STATE:
        return (
            "STATUS: NO_ACTIVE_APPLICANT\n\n"
            "There is currently no applicant loaded.\n"
            "Do not call this tool again.\n"
            "Inform the user that no prediction has been performed.[cite: 3]"
        )

    return (
        "STATUS: SUCCESS\n\n"
        "Current Applicant\n"
        "-----------------\n"
        f"Credit Score: {LIVE_APPLICANT_STATE['credit_score']}\n"
        f"Probability of Default: {LIVE_APPLICANT_STATE['probability_of_default']}\n"
        f"Risk Band: {LIVE_APPLICANT_STATE['risk_band']}\n"
        f"Decision: {LIVE_APPLICANT_STATE['decision']}\n\n"
        "This information is complete.\n"
        "Answer the user directly.\n"
        "Do not call this tool again.[cite: 3]"
    )


# ==============================================================================
# Documentation Tool[cite: 3]
# ==============================================================================

class DocumentationSearchSchema(BaseModel):
    query: str = Field(
        description="The specific search keyword, credit policy terms, or scorecard methodology query to search for."
    )

@tool("search_model_documentation", args_schema=DocumentationSearchSchema)
def search_model_documentation(query: str) -> str:
    """
    Searches the FAISS knowledge base.[cite: 3]
    """

    logger.info("Documentation search: %s", query)[cite: 3]

    try:
        # --- CRITICAL FIX: Moved get_retriever() inside the try block! ---
        # If Streamlit Cloud fails to build the database, we catch it securely here.
        retriever = get_retriever()[cite: 3]

        if retriever is None:
            logger.error("Retriever unavailable.[cite: 3]")
            return (
                "STATUS: SYSTEM_ERROR\n\n"
                "The documentation database is unavailable.\n"
                "Do not call this tool again.\n"
                "Tell the user the policy database is currently offline.[cite: 3]"
            )

        docs = retriever.invoke(query)[cite: 3]

    except Exception as exc:
        logger.exception("Retriever failure: %s", exc)[cite: 3]
        return (
            "STATUS: SYSTEM_ERROR\n\n"
            f"Retriever failed with error: {exc}\n\n"
            "Do not call this tool again. The server is likely waking up. Tell the user to wait 20 seconds.[cite: 3]"
        )

    if not docs:
        logger.info("No matching documentation found.[cite: 3]")
        return (
            "STATUS: NO_RESULTS\n\n"
            "No relevant policy documentation was found.\n\n"
            "Do not search again.\n"
            "Tell the user you could not find information related to their question.[cite: 3]"
        )

    context = "\n\n".join(
        doc.page_content.strip()
        for doc in docs[:2][cite: 3]
    )

    logger.info("Retrieved %d document chunks.", len(docs))[cite: 3]

    return (
        "STATUS: SUCCESS\n\n"
        "Retrieved Documentation\n"
        "-----------------------\n\n"
        f"{context}\n\n"
        "The above information is sufficient.\n"
        "Answer the user's question now.\n"
        "Do not call this tool again.[cite: 3]"
    )