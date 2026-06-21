# Requirements Document

## Introduction

The NemoClaw Virtual Twin Companion is a multi-agent engineering AI system for an NVIDIA hackathon. It enables conversational parametric CAD design of quadcopter drone chassis through a LangGraph-orchestrated pipeline. Two specialized agents (Design Agent and Validator Agent) collaborate to translate natural-language design goals into optimized 3D geometry, using NVIDIA's AI platform exclusively for inference and embeddings. The system provides a Streamlit-based UI with chat, agent trace visibility, and 3D model preview.

## Glossary

- **Orchestrator**: The LangGraph-based state machine (built on NeMo Agent Toolkit patterns) that manages routing, state transitions, and iteration flow between all agents and tools.
- **Design_Agent**: The AI agent responsible for translating natural-language design goals into parametric CAD values (arm_length, material_thickness, arm_width, center_cutout_radius).
- **Validator_Agent**: The AI agent responsible for evaluating proposed design parameters against engineering standards and RAG-retrieved knowledge, producing pass/fail verdicts.
- **Whiteboard**: The LangGraph shared state dictionary that all agents read from and write to, enabling inter-agent communication without direct coupling.
- **CAD_Tool**: The tool that invokes `master_drone_template.py` inside the NemoClaw/OpenShell sandboxed environment to generate STEP and STL geometry files from parametric inputs.
- **RAG_Engine**: The Retrieval-Augmented Generation pipeline using NVIDIA NIM Nemotron Embed and FAISS vector store to provide engineering context to the Validator_Agent.
- **NVIDIA_NIM**: NVIDIA Inference Microservices providing fast, optimized inference for both LLM reasoning (Nemotron LLMs) and embedding generation (Nemotron Embed).
- **ChatNVIDIA**: The LLM class from `langchain-nvidia-ai-endpoints` that connects to NVIDIA NIM-hosted Nemotron/Llama models for inference.
- **NVIDIAEmbeddings**: The embedding class from `langchain-nvidia-ai-endpoints` that generates vector representations using the NVIDIA NIM Nemotron Embed model (NV-Embed-QA).
- **Streamlit_UI**: The web-based user interface providing chat input, agent trace log, and 3D model viewer placeholder.
- **NemoClaw_OpenShell**: The secure runtime and sandboxed terminal environment from NVIDIA that ensures agents execute Python/CadQuery geometry scripts safely under strict enterprise policies.
- **NeMo_Agent_Toolkit**: NVIDIA's foundational toolkit providing multi-agent orchestration patterns, state management, observability (Agent Traces), and Guardrails (safety checks).
- **NVIDIA_Riva**: NVIDIA's speech recognition (ASR) and text-to-speech (TTS) engines enabling hands-free, multimodal Speech-to-CAD workflow.
- **Design_Parameters**: A JSON object containing the four controllable CAD variables: arm_length, material_thickness, arm_width, and center_cutout_radius.
- **Agent_Trace**: A chronological log of agent invocations, tool calls, and state transitions visible to the user in the Streamlit_UI, powered by NeMo_Agent_Toolkit observability.

## Requirements

### Requirement 1: LangGraph Orchestrator with NeMo Agent Toolkit Patterns

**User Story:** As a developer, I want a LangGraph-based orchestrator following NeMo Agent Toolkit patterns for state management and routing between agents, so that the multi-agent workflow executes in a deterministic, observable sequence with enterprise-grade safety.

#### Acceptance Criteria

1. THE Orchestrator SHALL define a LangGraph StateGraph with nodes for Design_Agent invocation, CAD_Tool execution, and Validator_Agent evaluation.
2. THE Orchestrator SHALL maintain a Whiteboard state dictionary containing keys for user_request, design_parameters, validator_feedback, iteration_count, cad_output_paths, and agent_trace.
3. WHEN the Orchestrator receives a user design request, THE Orchestrator SHALL route the request to the Design_Agent node as the first processing step.
4. WHEN the Validator_Agent returns a "FAIL" verdict and iteration_count is below the maximum iteration limit of 5, THE Orchestrator SHALL route state back to the Design_Agent node with validator_feedback populated.
5. WHEN the Validator_Agent returns a "PASS" verdict, THE Orchestrator SHALL terminate the graph and return the Whiteboard state containing design_parameters, cad_output_paths, validator_feedback, and iteration_count to the Streamlit_UI.
6. THE Orchestrator SHALL expose its compiled graph via a `main.py` entry point that can be invoked programmatically and by the Streamlit_UI.
7. THE Orchestrator SHALL implement NeMo_Agent_Toolkit observability patterns by logging all state transitions and agent invocations to the Agent_Trace.
8. THE Orchestrator SHALL integrate NeMo_Agent_Toolkit Guardrails as a pre-processing check on user inputs before routing to agents.
9. IF the iteration_count reaches the maximum limit of 5 without a "PASS" verdict, THEN THE Orchestrator SHALL terminate the graph and return the last Whiteboard state with the most recent validator_feedback to the Streamlit_UI.
10. IF the NeMo_Agent_Toolkit Guardrails reject a user input, THEN THE Orchestrator SHALL skip agent routing, record the rejection reason in the Agent_Trace, and return an error message indicating the input was blocked by safety guardrails to the Streamlit_UI.

### Requirement 2: NVIDIA NIM Integration via langchain-nvidia-ai-endpoints

**User Story:** As a developer, I want all LLM inference and embedding generation to use NVIDIA NIM microservices via the langchain-nvidia-ai-endpoints library exclusively, so that the project demonstrates full commitment to the NVIDIA AI stack.

#### Acceptance Criteria

1. THE Design_Agent SHALL use ChatNVIDIA from langchain-nvidia-ai-endpoints backed by NVIDIA_NIM for all LLM inference.
2. THE Validator_Agent SHALL use ChatNVIDIA from langchain-nvidia-ai-endpoints backed by NVIDIA_NIM for all LLM inference.
3. THE RAG_Engine SHALL use NVIDIAEmbeddings from langchain-nvidia-ai-endpoints backed by NVIDIA_NIM Nemotron Embed for all vector embedding generation.
4. THE Design_Agent SHALL target the "nvidia/llama-3.1-nemotron-70b-instruct" model as configured in config/agents.yaml.
5. THE Validator_Agent SHALL target the "nvidia/llama-3.1-nemotron-70b-instruct" model as configured in config/agents.yaml.
6. THE RAG_Engine SHALL target the "NV-Embed-QA" Nemotron Embed model for embeddings as configured in config/rag.yaml.
7. THE project SHALL NOT include any OpenAI, Anthropic, or non-NVIDIA LLM provider dependencies in requirements.txt.
8. IF the NVIDIA_API_KEY environment variable is not set, THEN ChatNVIDIA and NVIDIAEmbeddings SHALL raise a descriptive error at initialization indicating the missing key.

### Requirement 3: Design Agent Parameter Generation

**User Story:** As a user, I want the Design Agent to convert my natural-language design goals into concrete parametric values, so that I can design drone chassis through conversation rather than manual CAD manipulation.

#### Acceptance Criteria

1. WHEN the Design_Agent receives a user_request from the Whiteboard, THE Design_Agent SHALL produce a Design_Parameters JSON containing arm_length, material_thickness, arm_width, and center_cutout_radius within 30 seconds of invocation.
2. THE Design_Agent SHALL constrain arm_length to the range 80.0–200.0 mm by clamping any LLM-generated value that falls outside this range to the nearest bound.
3. THE Design_Agent SHALL constrain material_thickness to the range 2.0–10.0 mm by clamping any LLM-generated value that falls outside this range to the nearest bound.
4. THE Design_Agent SHALL constrain arm_width to the range 8.0–25.0 mm by clamping any LLM-generated value that falls outside this range to the nearest bound.
5. THE Design_Agent SHALL constrain center_cutout_radius to the range 10.0–30.0 mm by clamping any LLM-generated value that falls outside this range to the nearest bound.
6. WHEN validator_feedback is present in the Whiteboard, THE Design_Agent SHALL include the validator_feedback text in the prompt sent to the LLM and produce a revised Design_Parameters JSON where at least one parameter value differs from the previous iteration's output.
7. IF the Design_Agent fails to parse valid JSON from the LLM response, THEN THE Design_Agent SHALL return safe default values (arm_length=120.0, material_thickness=5.0, arm_width=15.0, center_cutout_radius=20.0).
8. IF the user_request field in the Whiteboard is empty or missing, THEN THE Design_Agent SHALL return safe default values (arm_length=120.0, material_thickness=5.0, arm_width=15.0, center_cutout_radius=20.0) without invoking the LLM.
9. THE Design_Agent SHALL enforce the structural constraint arm_width >= arm_length × 0.08 by increasing arm_width to meet this minimum when the LLM-generated values violate it.

### Requirement 4: Validator Agent Engineering Evaluation

**User Story:** As a user, I want the Validator Agent to evaluate proposed designs against engineering rules and RAG knowledge, so that generated designs are structurally sound and manufacturable.

#### Acceptance Criteria

1. WHEN the Validator_Agent receives Design_Parameters from the Whiteboard, THE Validator_Agent SHALL produce a verdict of "PASS" or "FAIL" with a numeric score between 0.0 and 1.0.
2. THE Validator_Agent SHALL evaluate structural integrity using the rule: arm_width >= arm_length * 0.08. If this rule is violated, the verdict SHALL be "FAIL".
3. THE Validator_Agent SHALL evaluate manufacturability by verifying material_thickness >= 2.0 mm for FDM processes. If this rule is violated, the verdict SHALL be "FAIL".
4. THE Validator_Agent SHALL query the RAG_Engine to retrieve relevant engineering context before producing its verdict.
5. IF the Validator_Agent issues a "FAIL" verdict, THEN THE Validator_Agent SHALL include a list of specific issues (each referencing the violated rule) and actionable suggestions (each referencing a specific parameter with a recommended value) in its response.
6. THE Validator_Agent SHALL write its structured evaluation (verdict, score, issues, suggestions, reasoning) to the Whiteboard state.
7. IF the RAG_Engine is unavailable or returns an empty context list, THEN THE Validator_Agent SHALL proceed with its evaluation using only the built-in engineering rules and note in its reasoning that RAG context was unavailable.
8. THE Validator_Agent SHALL issue a "PASS" verdict only when all built-in engineering rules pass and the RAG-informed assessment (if available) does not identify critical structural or manufacturability issues.

### Requirement 5: CAD Tool with NemoClaw/OpenShell Sandboxed Execution

**User Story:** As a developer, I want the CAD tool to invoke master_drone_template.py inside the NemoClaw/OpenShell secure runtime, so that LLM-generated parameters execute safely under enterprise policies without compromising the local system.

#### Acceptance Criteria

1. WHEN the Orchestrator invokes the CAD_Tool node, THE CAD_Tool SHALL write the Design_Parameters values into master_drone_template.py's parametric variable section (arm_length, material_thickness, arm_width, center_cutout_radius) by replacing their numeric assignments.
2. THE CAD_Tool SHALL contain a comment block delimited by `# --- NEMOCLAW_OPENSHELL EXECUTION START ---` and `# --- NEMOCLAW_OPENSHELL EXECUTION END ---` where the NemoClaw_OpenShell sandbox execution command is injected.
3. THE CAD_Tool SHALL NOT execute master_drone_template.py directly on the local filesystem.
4. WHEN execution completes successfully inside NemoClaw_OpenShell, THE CAD_Tool SHALL write the output file paths (optimized_drone_chassis.step, optimized_drone_chassis.stl) to the Whiteboard cad_output_paths key.
5. IF the sandboxed execution fails or times out (exceeding 60 seconds), THEN THE CAD_Tool SHALL write an error message to the Whiteboard indicating the failure reason (timeout or execution error) and preserve any previously existing Whiteboard state without crashing the Orchestrator.
6. THE CAD_Tool SHALL validate that all Design_Parameters fall within their defined min/max ranges (arm_length: 80.0–200.0 mm, material_thickness: 2.0–10.0 mm, arm_width: 8.0–25.0 mm, center_cutout_radius: 10.0–30.0 mm) before invoking execution.
7. IF any Design_Parameter value falls outside its defined min/max range, THEN THE CAD_Tool SHALL reject the execution request and write an error message to the Whiteboard identifying which parameter(s) violated their bounds.

### Requirement 6: RAG Engine with NVIDIA NIM Nemotron Embed

**User Story:** As a developer, I want a RAG pipeline using NVIDIA NIM Nemotron Embed (NV-Embed-QA) and FAISS, so that the Validator Agent can ground its assessments in published drone and aerospace engineering standards.

#### Acceptance Criteria

1. THE RAG_Engine SHALL ingest PDF documents from the data/ directory and split them into chunks of 500 characters with 50-character overlap using "\n\n" as the separator.
2. THE RAG_Engine SHALL generate embeddings using NVIDIAEmbeddings backed by NVIDIA_NIM Nemotron Embed with the "NV-Embed-QA" model with truncation set to "END".
3. IF a persisted vector store exists at data/vectorstore/, THEN THE RAG_Engine SHALL load the existing FAISS index instead of rebuilding from source PDFs.
4. WHEN the RAG_Engine builds or rebuilds the vector store from source PDFs, THE RAG_Engine SHALL persist the FAISS index to data/vectorstore/.
5. WHEN the Validator_Agent requests context with a text query, THE RAG_Engine SHALL return the top 5 document chunks ranked by similarity score, each including the chunk text and source PDF filename.
6. IF the data/ directory contains no PDF files, THEN THE RAG_Engine SHALL operate in a degraded mode returning an empty context list without raising an exception.
7. IF the NVIDIA_NIM embedding API is unreachable or returns an error during a query, THEN THE RAG_Engine SHALL return an empty context list and log an error message indicating the failure reason without raising an exception to the caller.
8. IF a PDF file in the data/ directory cannot be parsed, THEN THE RAG_Engine SHALL skip that file, log a warning identifying the filename, and continue processing the remaining PDF files.

### Requirement 7: Streamlit UI with Chat, Agent Trace, and NVIDIA Riva Integration

**User Story:** As a user, I want a Streamlit web interface with chat, agent trace visibility, a 3D viewer placeholder, and NVIDIA Riva speech input, so that I can interact with the system conversationally (including hands-free voice) and observe its reasoning.

#### Acceptance Criteria

1. THE Streamlit_UI SHALL provide a text-based chat input where users can type natural-language design goals.
2. THE Streamlit_UI SHALL display an expandable "Agent Trace" section showing chronological agent invocations, tool calls, and state transitions as logged by NeMo_Agent_Toolkit observability.
3. THE Streamlit_UI SHALL include a placeholder section for a 3D model viewer (PyVista/stpyvista integration stub).
4. THE Streamlit_UI SHALL include a stub section for NVIDIA_Riva ASR (Speech-to-Text) input with a comment explaining the Speech-to-CAD workflow integration.
5. THE Streamlit_UI SHALL include a stub section for NVIDIA_Riva TTS (Text-to-Speech) output with a comment explaining voice feedback for design results.
6. WHEN the Orchestrator completes a design cycle, THE Streamlit_UI SHALL display the final Design_Parameters as a formatted table and the Validator_Agent verdict with score prominently.
7. THE Streamlit_UI SHALL be launched via `streamlit run app.py` from the project root.
8. IF the Orchestrator returns an error (guardrail rejection, timeout, or max iterations reached), THEN THE Streamlit_UI SHALL display the error message in the chat area with visual differentiation from normal responses.
9. THE Streamlit_UI SHALL display a loading indicator while the Orchestrator is processing a design request.
10. THE Streamlit_UI SHALL display the full chat history within the current session, preserving all user messages and system responses.

### Requirement 8: YAML-Based Configuration

**User Story:** As a developer, I want all agent configs, system prompts, and tool endpoints defined in YAML files, so that behavior is externally configurable without code changes.

#### Acceptance Criteria

1. THE Orchestrator SHALL load agent configurations exclusively from config/agents.yaml.
2. THE Orchestrator SHALL load tool configurations exclusively from config/tools.yaml.
3. THE RAG_Engine SHALL load its pipeline configuration exclusively from config/rag.yaml.
4. THE config/ directory SHALL contain no JSON files; all configuration SHALL use YAML format.
5. IF a configuration file is missing, THEN THE Orchestrator SHALL raise an error at startup that includes the file path of the missing file and the expected YAML structure.
6. IF a configuration file contains invalid YAML syntax or is missing required top-level keys (agents.yaml: agent name keys with name, model, system_prompt; tools.yaml: tool name keys with name, script_path or embedding_model; rag.yaml: rag_pipeline with embedding, vector_store, retrieval), THEN THE Orchestrator SHALL raise an error at startup that identifies the file and the specific validation failure.
7. IF any configuration file fails to load or validate, THEN THE Orchestrator SHALL abort startup and SHALL NOT begin processing requests.

### Requirement 9: Documentation and Architecture Comments

**User Story:** As a hackathon judge, I want comprehensive docstrings, inline comments, and a README explaining the architecture and full NVIDIA Agentic AI stack integration, so that I can quickly understand the system design and evaluate NVIDIA platform usage.

#### Acceptance Criteria

1. THE project SHALL include a README.md with dedicated sections for: (a) a system architecture overview describing the multi-agent pipeline and component interactions, (b) an NVIDIA Agentic AI stack summary listing NIM, NemoClaw/OpenShell, NeMo Agent Toolkit, and Riva with a one-sentence role description for each, (c) the whiteboard state pattern explaining shared state keys and data flow between agents, and (d) startup commands required to install dependencies and launch the application.
2. Every Python function SHALL include a docstring in Google-style format containing: a one-line summary of purpose, an Args section listing each parameter with type and description, a Returns section specifying the return type and meaning, and a one-sentence note identifying which glossary component (Orchestrator, Design_Agent, Validator_Agent, CAD_Tool, RAG_Engine, or Streamlit_UI) the function belongs to.
3. Every LangGraph node definition SHALL include an inline comment stating the node's name, the Whiteboard state keys it reads, the Whiteboard state keys it writes, and the downstream node(s) it routes to.
4. THE CAD_Tool definition SHALL include architecture comments explaining the difference between Dassault CATIA/3DEXPERIENCE parametric BREP generation and mesh-based geometry approaches.
5. THE README.md SHALL include a section on how to configure the NVIDIA_API_KEY environment variable listing the required export command, and a table of NVIDIA NIM models used with columns for model name, purpose, and which agent or component uses it.
6. THE README.md SHALL include a section describing the NeMo_Agent_Toolkit Guardrails integration and observability features, specifying what safety checks are applied to user input and how Agent_Trace entries are recorded.
7. IF a Python function has no arguments or returns None, THEN the docstring SHALL still include the one-line summary and the component membership note.

### Requirement 10: Project Structure and Entry Points

**User Story:** As a developer, I want a well-organized project structure with clear entry points, so that the codebase is navigable and runnable with standard commands.

#### Acceptance Criteria

1. THE project SHALL organize agent implementations in an agents/ directory containing an __init__.py file and one or more Python module files.
2. THE project SHALL organize tool implementations in a tools/ directory containing an __init__.py file and one or more Python module files.
3. THE project SHALL organize configuration files in a config/ directory using YAML format with a .yaml file extension.
4. THE project SHALL include a data/ directory with a README file that describes the expected PDF file types and naming conventions for RAG source documents.
5. THE project SHALL provide main.py at the project root as the LangGraph entry point that exposes a callable function to invoke the agent graph and is executable via `python main.py`.
6. THE project SHALL provide app.py at the project root as the Streamlit UI entry point that is launchable via `streamlit run app.py`.
7. THE project SHALL include a requirements.txt at the project root listing langchain-nvidia-ai-endpoints, langgraph, streamlit, pyyaml, faiss-cpu, and cadquery as dependencies, each with a minimum version specifier.
8. WHEN a developer runs `python -c "import agents; import tools"` from the project root, THE project SHALL resolve both imports without raising ImportError.
