"""
chatbot/vector_store.py

Production-grade FAISS vector store management for the Credit Risk Platform.

Responsibilities
----------------
- Load environment variables
- Create HuggingFace API embeddings
- Build a FAISS vector database from policy documents
- Cache resources for Streamlit
- Auto-build missing vector databases
- Provide a tuned retriever for RAG
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================================================================
# Configuration
# ==============================================================================

load_dotenv()

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DOCS_DIR = PROJECT_ROOT / "docs"
DB_DIR = CURRENT_DIR / "vector_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ==============================================================================
# Embeddings
# ==============================================================================

@st.cache_resource(show_spinner=False)
def get_api_embeddings() -> HuggingFaceEndpointEmbeddings:
    """
    Returns a cached Hugging Face embedding model.

    Priority:
        1. Environment variable
        2. Streamlit Secrets
    """

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        try:
            hf_token = st.secrets["HF_TOKEN"]
        except Exception:
            hf_token = None

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not found. "
            "Please configure it in your .env file or Streamlit secrets."
        )

    logger.info("Loading HuggingFace API embeddings...")

    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=hf_token,
    )


# ==============================================================================
# Document Loading
# ==============================================================================

def load_documents():
    """
    Loads every TXT and PDF document under docs/.
    """

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    loaders = [
        DirectoryLoader(
            str(DOCS_DIR),
            glob="**/*.txt",
            loader_cls=TextLoader,
            silent_errors=True,
        ),
        DirectoryLoader(
            str(DOCS_DIR),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            silent_errors=True,
        ),
    ]

    documents = []

    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as exc:
            logger.exception("Document loading failed: %s", exc)

    return documents


# ==============================================================================
# Vector Store Builder
# ==============================================================================

def build_vector_store() -> Optional[FAISS]:
    """
    Builds a FAISS vector store from documents.
    """

    logger.info("Building vector database...")

    docs = load_documents()

    if not docs:
        logger.warning("No documents found inside docs/.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(docs)

    logger.info(
        "Loaded %d documents split into %d chunks.",
        len(docs),
        len(chunks),
    )

    embeddings = get_api_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    DB_DIR.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(DB_DIR))

    logger.info("FAISS vector database saved to %s", DB_DIR)

    return vector_store


# ==============================================================================
# Cached Loader
# ==============================================================================

@st.cache_resource(show_spinner="Loading policy knowledge base...")
def load_cached_vector_db() -> Optional[FAISS]:
    """
    Loads the cached FAISS database.

    Automatically rebuilds the database if it does not exist.
    """

    if not DB_DIR.exists():

        logger.warning(
            "Vector database not found. Attempting automatic rebuild."
        )

        build_vector_store()

    if not DB_DIR.exists():

        logger.error(
            "Unable to create vector database. "
            "The docs folder may be empty."
        )

        return None

    embeddings = get_api_embeddings()

    logger.info("Loading FAISS vector database...")

    return FAISS.load_local(
        str(DB_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ==============================================================================
# Retriever
# ==============================================================================

def get_retriever():
    """
    Returns a tuned retriever.

    The similarity score threshold greatly reduces irrelevant
    context and prevents unnecessary repeated searches by the LLM.
    """

    vector_store = load_cached_vector_db()

    if vector_store is None:
        return None

    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 2,
            "score_threshold": 0.45,
        },
    )


# ==============================================================================
# Script Entry Point
# ==============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    build_vector_store()