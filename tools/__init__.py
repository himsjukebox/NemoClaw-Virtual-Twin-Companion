# =============================================================================
# NemoClaw Virtual Twin Companion — Tools Package
# =============================================================================
# This package contains the tool implementations used by the agent pipeline:
#   1. CAD Tool     — Writes parameters into master_drone_template.py and
#                     executes it inside NemoClaw/OpenShell sandbox
#   2. RAG Engine   — PDF ingestion, FAISS vector store, and similarity
#                     queries using NVIDIA Nemotron Embed (NV-Embed-QA)
#
# Both tools are invoked by the LangGraph Orchestrator as graph nodes and
# communicate exclusively through the shared Whiteboard state dictionary.
# =============================================================================

from tools.cad_tool import CADTool
from tools.rag_engine import RAGEngine

__all__ = ["CADTool", "RAGEngine"]
