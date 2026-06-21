# =============================================================================
# NemoClaw Virtual Twin Companion — RAG Engine
# =============================================================================
# PURPOSE:
#   Ingests PDFs from data/, builds/loads a FAISS vector store using
#   NVIDIA Nemotron Embed (NV-Embed-QA), and serves similarity queries
#   to the Validator Agent.
#
# DESIGN RATIONALE:
#   The RAG Engine operates in a degraded-mode tolerant fashion: every external
#   dependency (NVIDIA API, PDF parsing, filesystem) has a fallback that returns
#   an empty list rather than raising an exception. This ensures the Validator
#   Agent can always proceed — using only built-in rules if RAG is unavailable.
#
# NVIDIA STACK CONTEXT:
#   Embeddings are generated exclusively via NVIDIAEmbeddings from
#   langchain-nvidia-ai-endpoints, backed by the NV-Embed-QA Nemotron Embed
#   model served through NVIDIA NIM.
# =============================================================================

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.loader import load_rag_config

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG Engine using NVIDIA NIM Nemotron Embed and FAISS vector store.

    Provides engineering context to the Validator Agent by retrieving
    relevant chunks from ingested PDF documents. Operates in degraded mode
    (returning empty results) when PDFs are missing, the NVIDIA API is
    unreachable, or PDF files are unparseable.

    Component: RAG_Engine
    """

    def __init__(self) -> None:
        """
        Initialize RAG Engine with NVIDIA embeddings and FAISS.

        Loads configuration from config/rag.yaml via the config loader,
        initializes NVIDIAEmbeddings with NV-Embed-QA model, and loads
        or builds the FAISS vector store.

        Component: RAG_Engine
        """
        self._config: Dict[str, Any] = self._load_config()
        self._embeddings: NVIDIAEmbeddings = self._init_embeddings()
        self._vector_store = self._load_or_build_store()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load RAG pipeline configuration from config/rag.yaml.

        Returns:
            dict: The rag_pipeline section of the configuration.

        Component: RAG_Engine
        """
        raw_config = load_rag_config()
        return raw_config["rag_pipeline"]

    def _init_embeddings(self) -> NVIDIAEmbeddings:
        """
        Initialize NVIDIAEmbeddings with model and truncation settings.

        Returns:
            NVIDIAEmbeddings: Configured embedding instance.

        Component: RAG_Engine
        """
        embedding_config = self._config["embedding"]
        return NVIDIAEmbeddings(
            model=embedding_config["model"],
            truncate=embedding_config["truncate"],
        )

    def _load_or_build_store(self):
        """
        Load persisted FAISS index or build from PDFs.

        If a persisted vector store exists at the configured persist_directory,
        loads it. Otherwise, builds a new store from PDF files in data/.

        Returns:
            FAISS or None: Vector store instance, or None if no store could
            be created (no PDFs, API error, etc.).

        Component: RAG_Engine
        """
        persist_dir = self._config["vector_store"]["persist_directory"]

        # Check if persisted index exists (FAISS saves index.faiss + index.pkl)
        index_file = Path(persist_dir) / "index.faiss"
        if index_file.exists():
            try:
                logger.info(
                    "Loading existing FAISS vector store from '%s'.", persist_dir
                )
                return FAISS.load_local(
                    persist_dir,
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                logger.exception(
                    "Failed to load persisted FAISS index from '%s'. "
                    "Attempting to rebuild from PDFs.",
                    persist_dir,
                )

        # No persisted store (or load failed) — build from PDFs
        return self._build_from_pdfs()

    def _build_from_pdfs(self):
        """
        Ingest PDFs from data/ directory, chunk, embed, and persist.

        Splits PDF content into 500-character chunks with 50-character overlap
        using "\\n\\n" as the separator. Embeds chunks via NVIDIAEmbeddings and
        persists the resulting FAISS index to data/vectorstore/.

        Skips unparseable PDFs with a warning and continues processing others.
        Returns None if no PDFs are found or all PDFs fail to parse.

        Returns:
            FAISS or None: Newly built vector store, or None if no documents
            could be ingested.

        Component: RAG_Engine
        """
        source_dir = self._config.get("source_directory", "data/")
        source_path = Path(source_dir)

        # Find all PDF files
        if not source_path.exists():
            logger.warning(
                "Source directory '%s' does not exist. "
                "Operating in degraded mode (no RAG context).",
                source_dir,
            )
            return None

        pdf_files = list(source_path.glob("*.pdf"))
        if not pdf_files:
            logger.info(
                "No PDF files found in '%s'. "
                "Operating in degraded mode (no RAG context).",
                source_dir,
            )
            return None

        # Configure text splitter from config
        splitter_config = self._config.get("text_splitter", {})
        chunk_size = splitter_config.get("chunk_size", 500)
        chunk_overlap = splitter_config.get("chunk_overlap", 50)
        separator = splitter_config.get("separator", "\n\n")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[separator],
        )

        # Load and split all PDFs
        all_documents = []
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                pages = loader.load()
                # Add source filename metadata
                for page in pages:
                    page.metadata["source"] = pdf_file.name
                chunks = text_splitter.split_documents(pages)
                all_documents.extend(chunks)
                logger.info(
                    "Ingested '%s': %d pages → %d chunks.",
                    pdf_file.name,
                    len(pages),
                    len(chunks),
                )
            except Exception as e:
                logger.warning(
                    "Failed to parse PDF '%s': %s. Skipping file.",
                    pdf_file.name,
                    e,
                )
                continue

        if not all_documents:
            logger.warning(
                "No documents could be ingested from PDFs in '%s'. "
                "Operating in degraded mode (no RAG context).",
                source_dir,
            )
            return None

        # Build FAISS vector store from documents
        try:
            vector_store = FAISS.from_documents(all_documents, self._embeddings)
        except Exception:
            logger.exception(
                "Failed to build FAISS vector store (NVIDIA API error). "
                "Operating in degraded mode (no RAG context)."
            )
            return None

        # Persist the vector store
        persist_dir = self._config["vector_store"]["persist_directory"]
        try:
            os.makedirs(persist_dir, exist_ok=True)
            vector_store.save_local(persist_dir)
            logger.info("Persisted FAISS vector store to '%s'.", persist_dir)
        except Exception as e:
            logger.warning(
                "Failed to persist FAISS vector store to '%s': %s. "
                "Store is available in memory but will not survive restart.",
                persist_dir,
                e,
            )

        return vector_store

    def query(self, text: str) -> List[Dict[str, str]]:
        """
        Retrieve top-k relevant document chunks for a query.

        Args:
            text (str): Query text to search for in the vector store.

        Returns:
            List[Dict[str, str]]: Top-5 chunks, each with 'text' (chunk content)
                and 'source' (PDF filename) keys. Returns empty list if the
                vector store is unavailable or the query fails.

        Component: RAG_Engine
        """
        if self._vector_store is None:
            logger.info(
                "Vector store is not available. Returning empty context list."
            )
            return []

        top_k = self._config.get("retrieval", {}).get("top_k", 5)

        try:
            results = self._vector_store.similarity_search(text, k=top_k)
            return [
                {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                }
                for doc in results
            ]
        except Exception:
            logger.exception(
                "NVIDIA API error during similarity search. "
                "Returning empty context list."
            )
            return []
