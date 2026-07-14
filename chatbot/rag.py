"""
chatbot/rag.py

Production-grade Credit Risk AI Agent.

Responsibilities
----------------
- Initialize the Groq LLM
- Configure the LangGraph ReAct agent
- Inject live applicant data
- Maintain conversation history
- Handle GraphRecursionError gracefully
- Provide robust logging and error handling
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from chatbot.prompts import AGENT_SYSTEM_PROMPT
from chatbot.tools import (
    get_current_applicant_data,
    search_model_documentation,
    update_live_applicant,
)

# ==============================================================================
# Logging
# ==============================================================================

logger = logging.getLogger(__name__)

# ==============================================================================
# Agent
# ==============================================================================


class CreditRiskAgent:
    """
    Credit Risk conversational agent backed by:

    • Groq
    • LangGraph
    • FAISS
    """

    DEFAULT_RECURSION_LIMIT = 25

    def __init__(self) -> None:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY was not found."
                " Configure it in your environment or Streamlit secrets."
            )

        logger.info("Initializing Groq model...")

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            api_key=api_key,
        )

        self.tools = [
            get_current_applicant_data,
            search_model_documentation,
        ]

        logger.info("Creating LangGraph ReAct agent...")

        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=AGENT_SYSTEM_PROMPT,
            debug=False,
        )

        logger.info("CreditRiskAgent initialized successfully.")
        
        print("=" * 60)
        print("NEW RAG.PY LOADED")
        print("=" * 60)

    # -------------------------------------------------------------------------

    def ask(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
        current_prediction=None,
    ) -> str:
        """
        Executes one conversational turn.
        """

        if chat_history is None:
            chat_history = []

        # --------------------------------------------------------------
        # Update current Streamlit prediction
        # --------------------------------------------------------------

        update_live_applicant(current_prediction)

        messages = list(chat_history)
        messages.append(HumanMessage(content=user_input))

        logger.info("User question: %s", user_input)

        try:
            print("Using recursion limit =", self.DEFAULT_RECURSION_LIMIT)

            response = self.agent_executor.invoke(
                {"messages": messages},
                config={
                    "recursion_limit": self.DEFAULT_RECURSION_LIMIT,
                    "configurable": {
                        "thread_id": "credit-risk-streamlit"
                    },
                },
            )

            final_message = response["messages"][-1].content

            logger.info("Agent completed successfully.")

            return final_message

        # --------------------------------------------------------------
        # Infinite Tool Loop
        # --------------------------------------------------------------

        except GraphRecursionError:

            logger.exception("LangGraph recursion limit reached.")

            return (
                "I couldn't complete your request because the reasoning "
                "process entered an unexpected loop.\n\n"
                "Please try:\n"
                "• Rephrasing your question.\n"
                "• Asking a more specific question.\n"
                "• Asking about one policy topic at a time."
            )

        # --------------------------------------------------------------
        # LangGraph / Tool Errors
        # --------------------------------------------------------------

        except ValueError as exc:

            logger.exception("Validation error.")

            return (
                "A validation error occurred while processing your request.\n\n"
                f"Details: {exc}"
            )

        # --------------------------------------------------------------
        # Unexpected Errors
        # --------------------------------------------------------------

        except Exception as exc:

            logger.exception("Unexpected agent failure.")

            return (
                "An unexpected error occurred while processing your request.\n\n"
                f"Error: {type(exc).__name__}\n"
                f"{exc}"
            )