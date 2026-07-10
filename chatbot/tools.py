"""
chatbot/tools.py

Custom LangChain tools for the AI Agent.
"""

from langchain.tools import tool
from chatbot.vector_store import get_retriever

# Global state to hold the live Streamlit applicant data
LIVE_APPLICANT_STATE = {}

def update_live_applicant(prediction_result=None):
    """Called by Streamlit to inject the latest prediction into the agent's memory."""
    global LIVE_APPLICANT_STATE
    if prediction_result:
        LIVE_APPLICANT_STATE = {
            "score": prediction_result.score,
            "probability_of_default": f"{prediction_result.probability:.2%}",
            "risk_band": prediction_result.risk_band,
            "decision": prediction_result.decision
        }

@tool
def get_current_applicant_data(query: str) -> str:
    """Use this tool to get the credit score, risk band, and decision for the applicant currently on the screen."""
    if not LIVE_APPLICANT_STATE:
        return "There is no active applicant being evaluated right now."
    
    return str(LIVE_APPLICANT_STATE)

@tool
def search_model_documentation(query: str) -> str:
    """Use this tool to search the model documentation, methodology, or credit policies."""
    if not (retriever := get_retriever()):
        return "No documentation is currently loaded in the database."

    if not (docs := retriever.invoke(query)):
        return "No relevant documentation found."

    return "\n\n".join(doc.page_content for doc in docs)