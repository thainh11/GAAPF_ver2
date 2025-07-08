# This script is the definitive way to run the AgentBench benchmark.
# It is location-aware and will work regardless of where you execute it from.

# --- Step 1: Find the correct directory ---
# Get the directory where this script itself is located.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Construct the absolute path to the 'agentbench' directory.
$AGENTBENCH_DIR_ABSOLUTE = (Resolve-Path (Join-Path $ScriptDir "agentbench")).Path

# Construct the absolute path to the config file needed by the benchmark.
$CONFIG_FILE_ABSOLUTE = Join-Path $AGENTBENCH_DIR_ABSOLUTE "configs/agents/gaapf-research-assistant.yaml"

Write-Host "--- Starting AgentBench Evaluation (Final Attempt) ---"
Write-Host "Target Directory: $AGENTBENCH_DIR_ABSOLUTE"

# --- Step 2: Run the benchmark from the correct location ---
# The --cwd flag tells 'conda run' to change to this directory before executing Python.
# This is the key to solving the 'ModuleNotFoundError: No module named src' error.
# It forces Python to see the 'src' directory inside 'agentbench'.
Write-Host "▶️ Running the benchmark..."
conda run -n agent-bench --cwd $AGENTBENCH_DIR_ABSOLUTE python -m src.assigner --agent-config $CONFIG_FILE_ABSOLUTE --task-name kg --max-turns 10

Write-Host "--- Benchmark Complete ---" 