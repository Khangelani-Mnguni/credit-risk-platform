"""
chatbot/vector_store.py

Handles document ingestion and vector database management.
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
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths relative to the project root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DB_DIR = CURRENT_DIR / "vector_db"

def build_vector_store():
    """Reads files from the docs/ folder and builds the FAISS index."""
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print("Created docs/ directory. Please add some PDF or TXT files and run this again.")
        return None

    # Load documents (supports TXT, MD, and PDF)
    text_loader = DirectoryLoader(str(DOCS_DIR), glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(str(DOCS_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    docs = text_loader.load() + pdf_loader.load()
    if not docs:
        print("No documents found in docs/ directory.")
        return None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Create embeddings and save to FAISS
    # Using explicit CPU assignment for consistency with the cloud loader
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(DB_DIR))
    print(f"Vector store built successfully with {len(chunks)} chunks.")
    return vector_store

# --- CRITICAL MEMORY FIX ---
# @st.cache_resource ensures the embedding model and database are only 
# loaded into the server's RAM ONCE, drastically reducing memory usage!
@st.cache_resource(show_spinner=False)
def load_cached_vector_db():
    """Loads the FAISS database and CPU-optimized embeddings into a global cache."""
    if not DB_DIR.exists():
        return None
        
    # 1. Force the smallest, most efficient embedding model
    # 2. Force it to run on CPU to prevent CUDA memory bloat
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': False}
    )
    
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

# Run this script directly to ingest new documents
if __name__ == "__main__":
    build_vector_store()