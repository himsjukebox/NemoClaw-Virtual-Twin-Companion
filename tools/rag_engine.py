# =============================================================================
# NemoClaw Virtual Twin Companion — RAG Engine (Multimodal)
# =============================================================================
# PURPOSE:
#   Ingests PDFs from data/, builds/loads a FAISS vector store using
#   NVIDIA Nemotron Embed (NV-Embed-QA), and serves similarity queries
#   to the Validator Agent.
#
# MULTIMODAL ENHANCEMENT:
#   In addition to text extraction, this engine uses PyMuPDF to extract
#   images from PDF pages and sends them to a vision-language model
#   (meta/llama-3.2-90b-vision-instruct via NVIDIA NIM) for captioning.
#   The generated captions are stored as additional text chunks in the
#   same FAISS vector store, enabling the Validator Agent to reason about
#   engineering diagrams, dimension drawings, and technical illustrations.
#
# DESIGN RATIONALE:
#   The RAG Engine operates in a degraded-mode tolerant fashion: every external
#   dependency (NVIDIA API, PDF parsing, filesystem, vision model) has a fallback
#   that returns an empty list rather than raising an exception.
#
# NVIDIA STACK CONTEXT:
#   - NVIDIAEmbeddings (NV-Embed-QA) for text embedding
#   - ChatNVIDIA (meta/llama-3.2-90b-vision-instruct) for image captioning
#   Both via langchain-nvidia-ai-endpoints backed by NVIDIA NIM.
# =============================================================================

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF for image extraction
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.loader import load_rag_config

logger = logging.getLogger(__name__)

# Vision model for image captioning
_VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"

# Minimum image size (pixels) to consider for captioning — skip tiny icons/logos
_MIN_IMAGE_SIZE = 100  # width or height must be >= 100px


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

        # Load and split all PDFs (text extraction)
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
                    "Ingested '%s': %d pages → %d text chunks.",
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

        # --- Multimodal: Extract and caption images from PDFs ---
        vision_config = self._config.get("vision", {})
        if vision_config.get("enabled", True):
            image_docs = self._extract_and_caption_images(pdf_files)
            if image_docs:
                all_documents.extend(image_docs)
                logger.info(
                    "Added %d image caption chunks to vector store.",
                    len(image_docs),
                )
        else:
            logger.info("Image captioning disabled in config. Text-only RAG.")

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

    def _extract_and_caption_images(self, pdf_files: List[Path]) -> List[Document]:
        """
        Extract images from PDF files and generate text captions using a
        vision-language model (meta/llama-3.2-90b-vision-instruct).

        For each PDF, extracts embedded images using PyMuPDF. Images larger
        than _MIN_IMAGE_SIZE are sent to the vision model for captioning.
        The generated captions are returned as LangChain Document objects
        ready for embedding and storage in the vector store.

        This enables multimodal RAG: the Validator Agent can find engineering
        knowledge from diagrams, dimension drawings, and technical illustrations
        that have no associated text in the PDF.

        Args:
            pdf_files (List[Path]): List of PDF file paths to process.

        Returns:
            List[Document]: Caption documents with metadata indicating source
                file and page number. Returns empty list if vision model is
                unavailable or no images are found.

        Component: RAG_Engine
        """
        image_documents = []

        # Initialize vision model
        try:
            vision_llm = ChatNVIDIA(
                model=_VISION_MODEL,
                max_tokens=1024,
                temperature=0.2,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize vision model '%s': %s. "
                "Skipping image captioning (text-only RAG).",
                _VISION_MODEL,
                e,
            )
            return []

        for pdf_file in pdf_files:
            try:
                doc = fitz.open(str(pdf_file))
                image_count = 0

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    images = page.get_images(full=True)

                    for img_idx, img_info in enumerate(images):
                        try:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            img_ext = base_image["ext"]
                            width = base_image.get("width", 0)
                            height = base_image.get("height", 0)

                            # Skip small images (icons, logos, decorations)
                            if width < _MIN_IMAGE_SIZE and height < _MIN_IMAGE_SIZE:
                                continue

                            # Convert to base64 for the vision API
                            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
                            mime_type = f"image/{img_ext}" if img_ext != "jpg" else "image/jpeg"

                            # Send to vision model for captioning
                            caption = self._caption_image(
                                vision_llm, img_b64, mime_type, pdf_file.name, page_num
                            )

                            if caption:
                                image_documents.append(
                                    Document(
                                        page_content=caption,
                                        metadata={
                                            "source": pdf_file.name,
                                            "page": page_num + 1,
                                            "type": "image_caption",
                                            "image_index": img_idx,
                                        },
                                    )
                                )
                                image_count += 1

                        except Exception as e:
                            logger.debug(
                                "Failed to process image %d on page %d of '%s': %s",
                                img_idx, page_num, pdf_file.name, e,
                            )
                            continue

                doc.close()
                if image_count > 0:
                    logger.info(
                        "Captioned %d images from '%s'.", image_count, pdf_file.name
                    )

            except Exception as e:
                logger.warning(
                    "Failed to extract images from '%s': %s. Skipping.",
                    pdf_file.name, e,
                )
                continue

        return image_documents

    def _caption_image(
        self,
        vision_llm: ChatNVIDIA,
        img_b64: str,
        mime_type: str,
        source_file: str,
        page_num: int,
    ) -> str:
        """
        Generate a text caption for an image using the vision-language model.

        Sends the image to meta/llama-3.2-90b-vision-instruct with an
        engineering-focused prompt to extract dimensions, materials,
        structural features, and design constraints.

        Args:
            vision_llm (ChatNVIDIA): The vision model instance.
            img_b64 (str): Base64-encoded image data.
            mime_type (str): MIME type of the image (e.g., "image/png").
            source_file (str): Source PDF filename for logging.
            page_num (int): Page number (0-indexed) for logging.

        Returns:
            str: Generated caption text, or empty string on failure.

        Component: RAG_Engine
        """
        try:
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "You are an expert aerospace and drone engineer. "
                            "Describe this engineering image in detail. Include:\n"
                            "- All dimensions and measurements visible\n"
                            "- Materials mentioned or implied\n"
                            "- Structural features (joints, mounts, cutouts, arms)\n"
                            "- Manufacturing constraints or notes\n"
                            "- Any design parameters or specifications shown\n"
                            "Be precise and technical. This description will be used "
                            "to inform drone chassis design validation."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_b64}",
                        },
                    },
                ]
            )

            response = vision_llm.invoke([message])
            caption = response.content.strip()

            if caption:
                # Prefix with context about where this came from
                caption = (
                    f"[Image from {source_file}, page {page_num + 1}] "
                    f"{caption}"
                )

            return caption

        except Exception as e:
            logger.debug(
                "Vision captioning failed for image on page %d of '%s': %s",
                page_num, source_file, e,
            )
            return ""

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
