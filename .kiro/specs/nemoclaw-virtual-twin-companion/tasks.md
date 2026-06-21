# Implementation Plan: NemoClaw Virtual Twin Companion

## Overview

This plan implements the NemoClaw Virtual Twin Companion — a multi-agent engineering AI system that enables conversational parametric CAD design of quadcopter drone chassis. The implementation follows an incremental approach: core data models and configuration first, then individual agents/tools, then orchestration wiring, and finally the Streamlit UI. All code uses Python with NVIDIA's AI stack exclusively.

## Tasks

- [x] 1. Set up project structure, core interfaces, and configuration loading
  - [x] 1.1 Create project directory structure and package init files
    - Create `tools/` directory with `__init__.py` exporting `CADTool` and `RAGEngine`
    - Create `data/` directory with a `README.md` describing expected PDF file types and naming conventions for RAG source documents
    - Create `tests/` directory with `conftest.py` containing shared fixtures and mocked NVIDIA client stubs
    - Ensure `agents/__init__.py` exports `DesignAgent` and `ValidatorAgent`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.8_

  - [x] 1.2 Implement configuration loader with validation
    - Create `config/loader.py` with functions to load and validate `agents.yaml`, `tools.yaml`, and `rag.yaml`
    - Validate required top-level keys: agents.yaml requires agent name keys with `name`, `model`, `system_prompt`; tools.yaml requires tool name keys with `name`, `script_path` or `embedding_model`; rag.yaml requires `rag_pipeline` with `embedding`, `vector_store`, `retrieval`
    - Raise descriptive errors including file path and specific validation failure if files are missing, have invalid YAML syntax, or are missing required keys
    - Abort startup if any configuration file fails to load or validate
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [ ]* 1.3 Write property test for configuration validation (Property 11)
    - **Property 11: Configuration Validation Error Identification**
    - Test that for any YAML content missing a required top-level key, the validator raises an error whose message includes the name of the missing key and the file path
    - Use `hypothesis` with `st.sampled_from` to randomly remove keys from valid config dicts
    - **Validates: Requirements 8.6**

  - [x] 1.4 Define WhiteboardState TypedDict and shared data models
    - Create `models/state.py` with `WhiteboardState` TypedDict containing: `user_request`, `design_parameters`, `validator_feedback`, `iteration_count`, `cad_output_paths`, `agent_trace`, `validator_verdict`, `validator_score`, `error`
    - Define Design Parameters schema constants (ranges, defaults) in `models/parameters.py`
    - Define Validator Response schema in `models/validation.py`
    - _Requirements: 1.2, 3.2, 3.3, 3.4, 3.5_

- [x] 2. Implement Design Agent with parameter clamping and structural constraints
  - [x] 2.1 Implement DesignAgent class with ChatNVIDIA integration
    - Extend existing `agents/design_agent.py` to import `WhiteboardState` from models
    - Add `clamp_parameters()` method that clamps all four parameters to their defined ranges (arm_length: 80–200, material_thickness: 2–10, arm_width: 8–25, center_cutout_radius: 10–30)
    - Add `enforce_structural_constraint()` method enforcing arm_width >= arm_length × 0.08
    - Ensure `invoke()` increments `iteration_count`, appends to `agent_trace`, and includes `validator_feedback` in the prompt when present
    - Handle empty `user_request` by returning defaults without calling the LLM
    - Handle LLM parse failures by returning safe defaults (120.0, 5.0, 15.0, 20.0)
    - Use ChatNVIDIA with "nvidia/llama-3.1-nemotron-70b-instruct" model loaded from config/agents.yaml
    - _Requirements: 2.1, 2.4, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

  - [ ]* 2.2 Write property test for parameter clamping (Property 1)
    - **Property 1: Parameter Clamping Preserves Bounds**
    - For any dict of four float values, after `clamp_parameters`, every output satisfies its defined range
    - Use `hypothesis` with `st.floats(allow_nan=False, allow_infinity=False)` for each param
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5**

  - [ ]* 2.3 Write property test for structural constraint (Property 2)
    - **Property 2: Structural Constraint Enforcement**
    - For any arm_length in [80, 200] and arm_width in [8, 25], after `enforce_structural_constraint`, output satisfies arm_width >= arm_length × 0.08
    - **Validates: Requirements 3.9**

  - [ ]* 2.4 Write property test for malformed LLM output (Property 3)
    - **Property 3: Malformed LLM Output Returns Defaults**
    - For any string that is not valid JSON with all four parameter keys, `_parse_parameters` returns exactly the default values
    - Use `hypothesis` with `st.text()` filtered to exclude valid JSON with all 4 keys
    - **Validates: Requirements 3.7**

- [x] 3. Implement Validator Agent with rule-based checks and RAG integration
  - [x] 3.1 Implement ValidatorAgent class with engineering rule checks
    - Create `agents/validator_agent.py` with `ValidatorAgent` class
    - Implement structural rule: arm_width >= arm_length × 0.08 (FAIL if violated)
    - Implement manufacturability rule: material_thickness >= 2.0 mm (FAIL if violated)
    - Produce verdict ("PASS" / "FAIL"), numeric score (0.0–1.0), issues list, suggestions list, and reasoning string
    - Write structured evaluation to Whiteboard state (`validator_verdict`, `validator_score`, `validator_feedback`)
    - Append to `agent_trace` with node="validator_agent", action, verdict, and score
    - Use ChatNVIDIA for LLM-based RAG evaluation on top of rule checks
    - Issue PASS only when all rules pass and RAG assessment has no critical issues
    - If RAG is unavailable, proceed with built-in rules only and note in reasoning
    - _Requirements: 2.2, 2.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

  - [ ]* 3.2 Write property test for validator verdict invariants (Property 4)
    - **Property 4: Validator Verdict Output Invariants**
    - For any valid design_parameters dict, output always contains verdict in {"PASS", "FAIL"} and score in [0.0, 1.0]
    - Use `hypothesis` with `st.fixed_dictionaries` and floats in valid ranges, mock LLM
    - **Validates: Requirements 4.1**

  - [ ]* 3.3 Write property test for rule violation causes FAIL (Property 5)
    - **Property 5: Validator Rule Violation Causes FAIL**
    - For any params where arm_width < arm_length × 0.08 OR material_thickness < 2.0, verdict is "FAIL"
    - **Validates: Requirements 4.2, 4.3**

  - [ ]* 3.4 Write property test for PASS correctness (Property 6)
    - **Property 6: Validator PASS When All Rules Satisfied**
    - For any params satisfying all rules with RAG containing no critical issues, verdict is "PASS"
    - Mock RAG to return non-critical context
    - **Validates: Requirements 4.8**

  - [ ]* 3.5 Write property test for FAIL feedback (Property 7)
    - **Property 7: FAIL Verdict Includes Actionable Feedback**
    - For any params causing FAIL, response contains non-empty issues and suggestions lists
    - **Validates: Requirements 4.5, 4.6**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement CAD Tool with NemoClaw/OpenShell sandbox execution
  - [x] 5.1 Implement CADTool class with parameter validation and sandbox execution
    - Create `tools/cad_tool.py` with `CADTool` class
    - Implement `_validate_ranges()` that checks all four parameters against their defined bounds and returns a list of error strings identifying violating parameters
    - Implement `_inject_parameters()` that writes parameter values into `master_drone_template.py`'s parametric variable section by replacing their numeric assignments
    - Include comment block delimited by `# --- NEMOCLAW_OPENSHELL EXECUTION START ---` and `# --- NEMOCLAW_OPENSHELL EXECUTION END ---` for sandbox execution injection
    - Implement `_execute_in_sandbox()` stub with 60-second timeout
    - Reject execution and write error to Whiteboard if any parameter is out of range
    - On successful execution, write output file paths to `cad_output_paths`
    - On failure/timeout, write error to state without crashing the Orchestrator
    - Append to `agent_trace` with node="cad_tool" and action details
    - Include architecture comments explaining parametric BREP vs mesh-based geometry
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 5.2 Write property test for CAD range validation (Property 8)
    - **Property 8: CAD Tool Range Validation Correctness**
    - For any parameter dict, `_validate_ranges` returns non-empty error list iff at least one parameter is out of bounds, and each error identifies the specific parameter
    - Use `hypothesis` with `st.fixed_dictionaries` and unconstrained floats
    - **Validates: Requirements 5.6, 5.7**

  - [ ]* 5.3 Write property test for parameter injection (Property 9)
    - **Property 9: Parameter Injection Preserves Values**
    - For any valid design_parameters, after injection the script text contains the exact numeric literal for each parameter
    - **Validates: Requirements 5.1**

- [x] 6. Implement RAG Engine with NVIDIA Nemotron Embed and FAISS
  - [x] 6.1 Implement RAGEngine class with PDF ingestion and FAISS vector store
    - Create `tools/rag_engine.py` with `RAGEngine` class
    - Load config from `config/rag.yaml` via the config loader
    - Initialize `NVIDIAEmbeddings` with model="NV-Embed-QA" and truncate="END"
    - Implement `_load_or_build_store()`: load persisted FAISS index from `data/vectorstore/` if it exists, otherwise build from PDFs
    - Implement `_build_from_pdfs()`: ingest PDFs from `data/`, split into 500-char chunks with 50-char overlap using "\n\n" separator, embed, and persist to `data/vectorstore/`
    - Implement `query()`: return top-5 chunks with text and source PDF filename
    - Handle no PDFs in data/ → return empty list (degraded mode)
    - Handle NVIDIA API errors → return empty list and log error
    - Handle unparseable PDFs → skip file, log warning, continue with others
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [ ]* 6.2 Write unit tests for RAG Engine degraded modes
    - Test empty data/ directory returns empty context list
    - Test NVIDIA API unreachable returns empty context list
    - Test unparseable PDF is skipped with warning logged
    - _Requirements: 6.6, 6.7, 6.8_

- [x] 7. Implement LangGraph Orchestrator with NeMo Agent Toolkit patterns
  - [x] 7.1 Implement main.py with LangGraph StateGraph and routing logic
    - Create `main.py` at project root defining the LangGraph StateGraph
    - Add nodes: `guardrails`, `design_agent`, `cad_tool`, `validator_agent`
    - Implement `guardrails_node()` applying NeMo Agent Toolkit safety checks on user input
    - Implement `route_after_guardrails()`: route to design_agent if input passes, END if blocked
    - Implement `route_after_validation()`: return "design_agent" on FAIL with iteration_count < 5, return END on PASS or iteration_count >= 5
    - Set entry point to guardrails node, wire edges: design_agent → cad_tool → validator_agent with conditional edges from guardrails and validator
    - Expose `run_graph(user_request: str) -> WhiteboardState` as public entry point
    - Log all state transitions and agent invocations to `agent_trace` (NeMo Agent Toolkit observability)
    - On guardrail rejection, record reason in agent_trace and return error message
    - Load all configs via config loader at startup, abort if validation fails
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10_

  - [ ]* 7.2 Write property test for iteration routing (Property 10)
    - **Property 10: Iteration Routing Correctness**
    - For any state with verdict="FAIL" and iteration_count in [0, 4], returns "design_agent"; for iteration_count >= 5 or verdict="PASS", returns END
    - Use `hypothesis` with `st.integers(min_value=0, max_value=10)` for iteration_count and `st.sampled_from(["PASS", "FAIL"])` for verdict
    - **Validates: Requirements 1.4, 1.9**

  - [ ]* 7.3 Write unit tests for orchestrator paths
    - Test guardrail rejection sets error in state and records in trace
    - Test PASS verdict terminates graph and returns final state
    - Test max iterations reached terminates with last feedback
    - Test empty user_request gets default parameters
    - _Requirements: 1.5, 1.9, 1.10, 3.8_

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement Streamlit UI with chat, trace, and NVIDIA Riva stubs
  - [x] 9.1 Implement app.py with chat interface and agent trace display
    - Create `app.py` at project root as the Streamlit UI entry point
    - Implement chat input where users type natural-language design goals
    - Display chat history within the session preserving all messages
    - Show loading indicator while Orchestrator processes a request
    - Display final Design_Parameters as a formatted table and validator verdict with score
    - Implement expandable "Agent Trace" section showing chronological invocations
    - Display errors (guardrail rejection, timeout, max iterations) with `st.error` visual differentiation
    - Include 3D model viewer placeholder section (PyVista/stpyvista stub)
    - Include NVIDIA Riva ASR stub section with comment explaining Speech-to-CAD workflow
    - Include NVIDIA Riva TTS stub section with comment explaining voice feedback
    - Ensure launchable via `streamlit run app.py`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10_

- [x] 10. Implement NVIDIA NIM integration verification and dependency management
  - [x] 10.1 Verify NVIDIA-exclusive dependency stack and API key handling
    - Ensure `requirements.txt` lists `langchain-nvidia-ai-endpoints`, `langgraph`, `streamlit`, `pyyaml`, `faiss-cpu`, and `cadquery` with minimum version specifiers
    - Verify NO OpenAI, Anthropic, or non-NVIDIA LLM provider dependencies exist
    - Ensure ChatNVIDIA and NVIDIAEmbeddings raise descriptive errors if NVIDIA_API_KEY is not set
    - Add `python-dotenv` integration for local development
    - _Requirements: 2.7, 2.8, 10.7_

- [x] 11. Create README.md and comprehensive documentation
  - [x] 11.1 Create README.md with architecture overview and NVIDIA stack documentation
    - Write system architecture overview section describing multi-agent pipeline and component interactions
    - Write NVIDIA Agentic AI stack summary listing NIM, NemoClaw/OpenShell, NeMo Agent Toolkit, and Riva with one-sentence role descriptions
    - Write whiteboard state pattern section explaining shared state keys and data flow
    - Write startup commands section (install dependencies, set NVIDIA_API_KEY, launch app)
    - Include NVIDIA_API_KEY configuration section with export command and table of NVIDIA NIM models used (model name, purpose, which component uses it)
    - Include NeMo Agent Toolkit Guardrails and observability section describing safety checks and Agent_Trace recording
    - _Requirements: 9.1, 9.5, 9.6_

  - [x] 11.2 Add comprehensive docstrings and inline comments to all modules
    - Ensure every Python function has Google-style docstring with: one-line summary, Args, Returns, and component membership note
    - Add LangGraph node inline comments stating: node name, state keys read, state keys written, downstream node(s)
    - Add BREP vs mesh architecture comments to CAD tool
    - Ensure functions with no args or returning None still have summary and component note
    - _Requirements: 9.2, 9.3, 9.4, 9.7_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using Hypothesis
- Unit tests validate specific examples and edge cases
- All LLM inference uses NVIDIA NIM exclusively via `langchain-nvidia-ai-endpoints`
- The system uses Python 3.10+ with pytest for testing and Hypothesis for property-based tests

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.4"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "2.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "2.4", "3.1"] },
    { "id": 4, "tasks": ["3.2", "3.3", "3.4", "3.5", "5.1", "6.1"] },
    { "id": 5, "tasks": ["5.2", "5.3", "6.2"] },
    { "id": 6, "tasks": ["7.1"] },
    { "id": 7, "tasks": ["7.2", "7.3", "9.1"] },
    { "id": 8, "tasks": ["10.1", "11.1"] },
    { "id": 9, "tasks": ["11.2"] }
  ]
}
```
