# Development Plan: Phase 2 & 3

This document outlines the development plan for the next two major phases of GAAPF, focusing on adding code generation/repair capabilities and implementing a content synchronization engine for Retrieval-Augmented Generation (RAG).

---

## PHASE 2 – “CODE GEN / CODE FIX MVP”
**(Estimated Time: 2 weeks)**

### A. Core Objectives
1.  **Code Generation**: Implement an agent capable of generating Python code based on user specifications.
2.  **Code Repair**: Implement an agent capable of analyzing and fixing bugs in existing code files.
3.  **Sandbox Execution**: Ensure all generated/repaired code is tested in a secure, isolated sandbox environment.
4.  **CLI Integration**: Expose these new capabilities through simple CLI commands (`/codegen`, `/fix`).
5.  **Profile Integration**: Track coding-related metrics in the user's learning profile.

### B. Detailed Task Breakdown

#### 1. ⌨️ `CodeGenerationAgent`
-   **File**: `src/GAAPF/core/agents/code_generator.py`
-   **Primary Method**: `generate_code(requirements, context)` which returns a dictionary containing `code`, `tests`, and `documentation`.
-   **Validation Method**: `validate(code)` which will use the `SandboxRunner` to execute `pytest`, `ruff`, and `bandit`.
-   **Workflow**: The agent will follow a "Plan → Generate → Validate" workflow, with an optional self-correction loop (max 3 retries).
-   **Output**: The agent will return the generated code along with a `quality_score` and any `lint_errors` for analysis.

#### 2. 🛠️ `CodeRepairAgent`
-   **File**: `src/GAAPF/core/agents/code_repair.py`
-   **Primary Method**: `repair_file(path, context)` which returns the patched code and test results.
-   **Workflow**: Load the file, generate a git diff, ask the LLM to suggest a patch, apply the patch using the `unidiff` library, and run validation tests.
-   **Output**: The agent will return a `patch_summary`, `test_result`, and any `security_issues` found.

#### 3. 🔒 `SandboxRunner`
-   **File**: `src/GAAPF/tools/sandbox_runner.py`
-   **Functionality**: A wrapper for running shell commands in an isolated environment with resource limits.
-   **Method**: `run_in_sandbox(command, timeout=10)` which returns `stdout`, `stderr`, and an `exit_code`.
-   **Platform Handling**:
    -   **Unix/macOS**: Use `subprocess` with `resource.setrlimit` and a timeout.
    -   **Windows**: Use `subprocess` with Job Objects or a simpler time-based timeout.

#### 4. ⚙️ CLI Integration
-   **File**: `src/GAAPF/interfaces/cli/simple_cli.py`
-   **New Commands**:
    -   `/codegen "Create a function to calculate Fibonacci"`
    -   `/fix src/utils/parser.py`
-   **Agent Registration**: Add `code_generator` and `code_repair` to the `self.agents` dictionary.
-   **UI**: Display the results from the agents, including the generated code, test status, and quality score.

#### 5. 📄 Profile & Analytics
-   **File**: `src/GAAPF/core/memory/profile_manager.py`
-   **New Method**: `update_code_metrics(...)` to track `code_snippets_generated`, `tests_passed`, `bugs_fixed`, and `average_quality`.
-   **UI**: Display these new metrics under a "Coding Practice" section in the `/profile` command output.

### C. Definition of Done
-   `/codegen "spec"` successfully creates a file, and its auto-generated tests pass with a quality score >= 70/100.
-   `/fix path/to/file.py` applies a patch that makes the tests pass, or clearly reports the remaining errors.
-   The sandbox successfully terminates code that runs into an infinite loop.
-   The user's profile correctly stores and displays the number of snippets generated and bugs fixed.
-   All new code passes `pytest`, `ruff`, and `bandit` checks in the CI pipeline.
-   A `docs/CODE_GEN_FIX.md` file is created to document the new features.

---

## PHASE 3 – “CONTENT SYNC + RAG”
**(Estimated Time: 1.5 - 2 weeks, excluding RLHF)**

### A. Core Objectives
1.  **Content Synchronization**: Build an engine to automatically sync documentation, blog posts, and recent commits from framework repositories.
2.  **Vector Store**: Store the processed content as embeddings in a local Vector Store (ChromaDB) for Retrieval-Augmented Generation (RAG).
3.  **Live-Docs Retrieval**: Enable agents (like `InstructorAgent`) to query this Vector Store to provide the most up-to-date, context-aware answers.
4.  **CLI Integration**: Add a `/update examples` command to trigger the sync manually and report on new content.
5.  **Safety & Control**: Ensure the sync process is secure, rate-limited, and respects content licenses.

### B. Detailed Task Breakdown

#### 1. 🔌 `ContentSyncEngine`
-   **Directory**: `src/GAAPF/core/sync/`
-   **File**: `sync_engine.py`
-   **Methods**:
    -   `detect_changes()`: Check for new commits (GitHub API) or articles (RSS feeds).
    -   `load_content()`: Use LangChain loaders (`GitHubRepositoryLoader`, `SitemapLoader`) to fetch content.
    -   `process_content()`: Clean HTML, convert Markdown to plain text, chunk documents.
    -   `license_filter()`: Check for SPDX license identifiers and skip incompatible content.
    -   `generate_embeddings()`: Convert text chunks to vectors.
    -   `upsert_to_vectorstore()`: Add or update vectors in ChromaDB.
    -   `write_sync_log()`: Log the results of the sync to `monitoring_data/`.

#### 2. 📚 `VectorStore` Setup
-   **File**: `src/GAAPF/core/tools/vector_store.py`
-   **Functionality**: A wrapper for ChromaDB to manage the persistent directory (`data/framework_cache/`) and metadata schema.

#### 3. 🔍 `Retriever` Utility
-   **File**: `src/GAAPF/core/utils/retriever.py`
-   **Method**: `retrieve_docs(query, k=3)` to perform a semantic search and return the top `k` documents.
-   **Integration**: `InstructorAgent` and `CodeAssistantAgent` will call this utility. If relevant docs are found, they will be injected into the LLM prompt as context.

#### 4. 🖥️ CLI Integration
-   **File**: `src/GAAPF/interfaces/cli/simple_cli.py`
-   **New Command**: `/update examples` which triggers `sync_engine.run_once()` and reports the number of new and updated documents.
-   **UI**: Use a Rich progress bar to show the status of the sync process.

#### 5. 🌐 Cron / Script Mode
-   **File**: `scripts/maintenance/update_examples.py`
-   **Functionality**: A standalone script that can be run manually (`python scripts/maintenance/update_examples.py`) or scheduled as a cron job / Windows Task.

#### 6. 📝 Profile & Analytics
-   **File**: `src/GAAPF/core/memory/profile_manager.py`
-   **New Method**: `update_sync_stats(profile, num_new_docs)` to track `total_docs_available` and `last_sync_time`.
-   **UI**: The `/profile` command will display the number of documents in the DB and the last sync timestamp.

### C. Definition of Done
-   Running `python scripts/maintenance/update_examples.py` successfully populates the Vector Store with documents and creates a log file.
-   The `/update examples` command in the CLI works and reports the number of new documents.
-   When a user asks a question about a concept present in the synced documents, the `InstructorAgent`'s response includes a reference like *"According to the latest documentation..."*.
-   All new code passes CI checks.
-   A `docs/SYNC_ENGINE.md` file is created to document the architecture and setup.

