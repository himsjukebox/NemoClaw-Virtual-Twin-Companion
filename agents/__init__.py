# =============================================================================
# NemoClaw Virtual Twin Companion — Agents Package
# =============================================================================
# This package contains the two core agents of the multi-agent system:
#   1. Design Agent  — Translates NL goals into parametric CAD values
#   2. Validator Agent — Evaluates designs against RAG + engineering rules
#
# Both agents use ChatNVIDIA from langchain-nvidia-ai-endpoints as their LLM
# backbone, ensuring the entire inference stack runs on NVIDIA's platform.
# =============================================================================

from agents.design_agent import DesignAgent
from agents.validator_agent import ValidatorAgent

__all__ = ["DesignAgent", "ValidatorAgent"]
