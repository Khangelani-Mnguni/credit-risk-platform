"""
chatbot/tools.py

Custom LangChain tools for the Credit Risk AI Assistant.

Responsibilities
----------------
- Maintain the live applicant state supplied by Streamlit.
- Search the FAISS knowledge base.
- Provide structured tool outputs for the LangGraph agent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain.tools import tool

from chatbot.vector_store import get_retriever

logger = logging.getLogger(__name__)

# ==============================================================================
# Live Applicant State
# ==============================================================================

LIVE_APPLICANT_STATE: Dict[str, Any] = {}


def update_live_applicant(prediction_result=None) -> None:
    """
    Updates the in-memory applicant state from the latest Streamlit prediction.
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

    logger.info("Updated live applicant state.")


# ==============================================================================
# Applicant Tool
# ==============================================================================

@tool
def get_current_applicant_data(query: str) -> str:
    """
    Returns the current applicant's prediction.

    Use this tool ONLY when the user asks about:

    - current applicant
    - applicant score
    - decision
    - probability of default
    - risk band

    Call this tool only once per question.
    """

    logger.info("Applicant tool invoked.")

    if not LIVE_APPLICANT_STATE:

        return (
            "STATUS: NO_ACTIVE_APPLICANT\n\n"
            "There is currently no applicant loaded.\n"
            "Do not call this tool again.\n"
            "Inform the user that no prediction has been performed."
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
        "Do not call this tool again."
    )


# ==============================================================================
# Documentation Tool
# ==============================================================================

@tool
def search_model_documentation(query: str) -> str:
    """
    Searches the FAISS knowledge base.

    IMPORTANT

    • Call this tool at most ONE time per user question.

    • After receiving the result,
      answer the user immediately.

    • Never retry with different wording.

    • Never call this tool twice for the same question.
    """

    logger.info("Documentation search: %s", query)

    retriever = get_retriever()

    if retriever is None:

        logger.error("Retriever unavailable.")

        return (
            "STATUS: SYSTEM_ERROR\n\n"
            "The documentation database is unavailable.\n"
            "Do not call this tool again.\n"
            "Tell the user the policy database is currently offline."
        )

    try:

        docs = retriever.invoke(query)

    except Exception as exc:

        logger.exception("Retriever failure: %s", exc)

        return (
            "STATUS: SYSTEM_ERROR\n\n"
            f"Retriever failed with error: {exc}\n\n"
            "Do not call this tool again."
        )

    if not docs:

        logger.info("No matching documentation found.")

        return (
            "STATUS: NO_RESULTS\n\n"
            "No relevant policy documentation was found.\n\n"
            "Do not search again.\n"
            "Tell the user you could not find information "
            "related to their question."
        )

    # --------------------------------------------------------------------------
    # Return only the best two chunks.
    # Too much context often causes unnecessary reasoning loops.
    # --------------------------------------------------------------------------

    context = "\n\n".join(
        doc.page_content.strip()
        for doc in docs[:2]
    )

    logger.info("Retrieved %d document chunks.", len(docs))

    return (
        "STATUS: SUCCESS\n\n"
        "Retrieved Documentation\n"
        "-----------------------\n\n"
        f"{context}\n\n"
        "The above information is sufficient.\n"
        "Answer the user's question now.\n"
        "Do not call this tool again."
    )