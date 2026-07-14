"""
pages/ai_assistant.py

Dedicated page for the LangChain Credit Risk AI Agent.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Path hack to ensure we can import from the root project folder
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from chatbot.rag import CreditRiskAgent
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Khangs - Risk Analyst", page_icon="🇿🇦🐢", layout="wide")

st.title("🇿🇦🐢 AI Model Assistant & Policy Guide")
st.markdown("Ask questions about South African credit policies, scorecard methodology, or the active applicant.")
st.divider()

# Initialize Agent in session state
if "agent" not in st.session_state:
    st.session_state.agent = CreditRiskAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("e.g., What is our maximum DTI policy for a medium risk applicant?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is searching policies and analyzing..."):
            
            # Fetch the live prediction result from session state (if an applicant was scored)
            live_result = st.session_state.get("current_prediction", None)
            
            # Pass query and history to the LangChain Agent
            answer = st.session_state.agent.ask(
                user_input=prompt,
                chat_history=st.session_state.langchain_history,
                current_prediction=live_result
            )
            st.markdown(answer)
            
    # Save to UI history and LangChain memory format
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.langchain_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=answer)
    ])