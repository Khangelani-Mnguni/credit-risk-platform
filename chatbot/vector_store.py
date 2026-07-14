"""
chatbot/vector_store.py

Handles document ingestion and vector database management using Free Cloud APIs.
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Define paths relative to the project root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DB_DIR = CURRENT_DIR / "vector_db"

def get_api_embeddings():
    """Initializes the lightweight Hugging Face API Embeddings."""
    # Try to get the token from the environment (.env)
    hf_token = os.environ.get("HF_TOKEN")
    
    # If not found, safely try Streamlit secrets (handles terminal execution)
    if not hf_token:
        try:
            hf_token = st.secrets.get("HF_TOKEN")
        except Exception:
            pass 

    # Raise a clear error if the token is completely missing
    if not hf_token:
        raise ValueError("HF_TOKEN is missing. Please add it to your Streamlit secrets or .env file.")
        
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token
    )

def build_vector_store():
    """Reads files from the docs/ folder and builds the FAISS index."""
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print("Created docs/ directory. Please add some PDF or TXT files and run this again.")
        return None

    # Load documents
    text_loader = DirectoryLoader(str(DOCS_DIR), glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(str(DOCS_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    docs = text_loader.load() + pdf_loader.load()
    if not docs:
        print("No documents found in docs/ directory.")
        return None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Create embeddings via API and save to FAISS
    embeddings = get_api_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(DB_DIR))
    print(f"Vector store built successfully with {len(chunks)} chunks.")
    return vector_store

@st.cache_resource(show_spinner=False)
def load_cached_vector_db():
    """Loads the FAISS database utilizing cloud API embeddings."""
    
    # --- CRITICAL FIX: AUTO-BUILD IF MISSING ON CLOUD ---
    if not DB_DIR.exists():
        print("Vector Database not found on server. Auto-building now...")
        build_vector_store()
        
    # If it STILL doesn't exist (e.g., no documents in the folder)
    if not DB_DIR.exists():
        return None
        
    embeddings = get_api_embeddings()
    
    return FAISS.load_local(
        str(DB_DIR), 
        embeddings, 
        allow_dangerous_deserialization=True
    )

def get_retriever():
    """Returns the LangChain retriever interface for the cached database."""
    vector_store = load_cached_vector_db()
    if vector_store:
        return vector_store.as_retriever(search_kwargs={"k": 3})
    return None

if __name__ == "__main__":
    build_vector_store()