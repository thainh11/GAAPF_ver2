# Simple Agent Benchmark

This directory contains a straightforward script to evaluate the performance of individual agents from the GAAPF project.

## Phase 1: Individual Agent Evaluation

This evaluation focuses on testing the core capabilities of the `ResearchAssistant` and `CodeAssistant` agents.

### Prerequisites

1.  **Install Dependencies**: Make sure you have all the project dependencies installed.
    ```bash
    pip install -r requirements.txt
    ```

2.  **API Keys & Environment**: Ensure you have a `.env` file in the root of the project directory.

    ### For Google Generative AI (Default)
    Your `.env` file must contain your Google API key.
    ```
    # .env file
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

    ### For Google Vertex AI
    To use Vertex AI, you need to set the `LLM_PROVIDER` and provide your Google Cloud project details.
    ```
    # .env file
    LLM_PROVIDER="vertex-ai"
    GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
    
    # You also need to be authenticated with gcloud.
    # Run `gcloud auth application-default login` in your terminal.
    ```

### How to Run

**This is the final, correct way to run the benchmark.**

1.  **Start the Agent Adapter**: In your first terminal, make sure you are in the project's root directory (`vinagent-main`) and run the agent adapter. It must be running before you start the benchmark.
    ```bash
    python run_benchmark/gaapf_agent_adapter.py
    ```
    Leave this terminal running.

2.  **Execute the Benchmark Script**: In a second terminal, navigate to the project's root directory (`vinagent-main`) and run the new PowerShell script.
    ```powershell
    .\run_benchmark\run_agentbench.ps1
    ```
    
    If you get a security error about scripts being disabled on your system, you may need to change the execution policy for this one time. You can do so with this command:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    ```
    Then, try running `.\run_benchmark\run_agentbench.ps1` again.

### What to Expect

The `run_agentbench.ps1` script will:
1.  Automatically set the required `PYTHONPATH`.
2.  Activate the correct `conda` environment.
3.  Run the benchmark using the proper module-based execution.
4.  Connect to your agent adapter and run the "Knowledge Graph" evaluation.
5.  Print the results and score upon completion.

This provides a single, reliable command to run the entire benchmark.

This simple setup allows for quick, targeted testing and makes it easy to debug individual agent performance without the overhead of a complex benchmarking framework. 