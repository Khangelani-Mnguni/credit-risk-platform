"""
chatbot/rag.py

Initializes the AI Agent execution environment using modern LangGraph architecture
and the ultra-fast Groq API.
"""

import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from chatbot.prompts import AGENT_SYSTEM_PROMPT
from chatbot.tools import get_current_applicant_data, search_model_documentation, update_live_applicant

class CreditRiskAgent:
    def __init__(self):
        # Initialize the ultra-fast cloud LLM via Groq
        # Llama-3.1 natively supports precise agent tool-calling
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            api_key=os.environ.get("GROQ_API_KEY")
        )
        
        # Bind the tools (Live Data + FAISS Retriever)
        self.tools = [get_current_applicant_data, search_model_documentation]
        
        # Create the execution environment using LangGraph
        self.agent_executor = create_react_agent(
            model=self.llm, 
            tools=self.tools, 
            prompt=AGENT_SYSTEM_PROMPT
        )

    def ask(self, user_input: str, chat_history: list = None, current_prediction=None) -> str:
        """Processes a query and returns the agent's response."""
        if chat_history is None:
            chat_history = []
            
        # Update the tools with the live Streamlit context
        update_live_applicant(current_prediction)
        
        # Combine previous memory with the new user message
        messages = chat_history + [HumanMessage(content=user_input)]
        
        # Invoke the LangGraph agent
        response = self.agent_executor.invoke({"messages": messages})
        
        # LangGraph returns the full conversational state; we grab the final message
        return response["messages"][-1].content