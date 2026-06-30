"""
NVIDIA-stack consistency smoke tests.

Static checks ensuring the project uses only NVIDIA inference providers
and that the physics engine has NO inference dependency.

Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5
"""
import ast
import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent

# Files that should use NVIDIA for inference
INFERENCE_FILES = [
    PROJECT_ROOT / "agents" / "design_agent.py",
    PROJECT_ROOT / "agents" / "validator_agent.py",
    PROJECT_ROOT / "main.py",
]

# Files that should NOT import any inference client
NO_INFERENCE_FILES = [
    PROJECT_ROOT / "tools" / "physics_engine.py",
]

# Banned provider imports (non-NVIDIA)
BANNED_IMPORTS = [
    "openai",
    "anthropic",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google",
    "langchain_community.llms",
]

# The ONLY allowed inference import
ALLOWED_INFERENCE_IMPORT = "langchain_nvidia_ai_endpoints"

# Any inference-related module (NVIDIA or otherwise)
ALL_INFERENCE_IMPORTS = BANNED_IMPORTS + [ALLOWED_INFERENCE_IMPORT]


def _get_imports(filepath: Path) -> list:
    """Parse a Python file and return all import module names."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


class TestNvidiaStackConsistency:
    """Smoke tests verifying NVIDIA-only inference stack."""

    def test_no_banned_providers_in_inference_files(self):
        """
        Feature modules do NOT import OpenAI/Anthropic/non-NVIDIA providers (R15.5).

        Validates: Requirements 15.5
        """
        for filepath in INFERENCE_FILES:
            if not filepath.exists():
                continue
            imports = _get_imports(filepath)
            for banned in BANNED_IMPORTS:
                assert banned not in imports, (
                    f"Found banned import '{banned}' in {filepath.name}. "
                    f"Only langchain_nvidia_ai_endpoints is allowed."
                )

    def test_inference_files_use_nvidia(self):
        """
        Design/Validator agents and main.py use langchain_nvidia_ai_endpoints (R15.2, R15.3).

        Validates: Requirements 15.2, 15.3
        """
        for filepath in INFERENCE_FILES:
            assert filepath.exists(), f"{filepath.name} must exist"
            imports = _get_imports(filepath)
            assert ALLOWED_INFERENCE_IMPORT in imports, (
                f"{filepath.name} must import '{ALLOWED_INFERENCE_IMPORT}' "
                f"for NVIDIA-backed inference. Found imports: {imports}"
            )

    def test_physics_engine_no_inference_imports(self):
        """
        tools/physics_engine.py imports NO inference client at all (R15.1).

        Validates: Requirements 15.1
        """
        filepath = PROJECT_ROOT / "tools" / "physics_engine.py"
        assert filepath.exists(), "tools/physics_engine.py must exist"
        imports = _get_imports(filepath)

        # Should not import any LLM/inference modules (NVIDIA or otherwise)
        for module in ALL_INFERENCE_IMPORTS:
            assert module not in imports, (
                f"Physics engine must not import inference module '{module}'. "
                f"All physics math must be local/deterministic (R15.1)."
            )

    def test_physics_engine_no_network_inference_patterns(self):
        """
        tools/physics_engine.py does not reference langchain_core LLM base classes (R15.1).

        Validates: Requirements 15.1
        """
        filepath = PROJECT_ROOT / "tools" / "physics_engine.py"
        assert filepath.exists(), "tools/physics_engine.py must exist"
        imports = _get_imports(filepath)

        # Should not import LLM-related langchain_core modules
        llm_related = [
            "langchain_core.language_models",
            "langchain_core.llms",
            "langchain_core.chat_models",
        ]
        for module in llm_related:
            assert module not in imports, (
                f"Physics engine must not import LLM-related module '{module}'. "
                f"All physics math must be local/deterministic (R15.1)."
            )

    def test_physics_engine_imports_only_allowed_modules(self):
        """
        Physics engine imports only math/dataclasses/typing and config.loader (R15.1).

        Validates: Requirements 15.1
        """
        filepath = PROJECT_ROOT / "tools" / "physics_engine.py"
        assert filepath.exists(), "tools/physics_engine.py must exist"
        imports = _get_imports(filepath)

        # Allowed import prefixes for the physics engine
        allowed_prefixes = (
            "math",
            "dataclasses",
            "typing",
            "config",
        )

        for imp in imports:
            assert any(imp.startswith(prefix) for prefix in allowed_prefixes), (
                f"Physics engine has unexpected import '{imp}'. "
                f"Only standard library (math, dataclasses, typing) and "
                f"config.loader are expected. No inference dependencies allowed."
            )
