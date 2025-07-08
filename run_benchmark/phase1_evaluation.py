import os
import sys
import asyncio
from dotenv import load_dotenv

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.GAAPF.core.core.session_manager import SessionManager
from src.GAAPF.core.agents.research_assistant import ResearchAssistant
from src.GAAPF.core.agents.code_assistant import CodeAssistant

# Load environment variables from .env file
load_dotenv()

# --- Evaluation Dataset ---
BENCHMARK_DATASET = [
    {
        "agent": "ResearchAssistant",
        "question": "What is the primary purpose of the 'requests' library in Python?"
    },
    {
        "agent": "CodeAssistant",
        "question": "Write a Python code snippet to make a GET request to 'https://api.github.com' and print the status code."
    },
    {
        "agent": "ResearchAssistant",
        "question": "Explain the difference between a GET and a POST request in HTTP."
    },
    {
        "agent": "CodeAssistant",
        "question": "Show me how to send a POST request with a JSON payload using the 'requests' library."
    },
    {
        "agent": "ResearchAssistant",
        "question": "What does the 'timeout' parameter do in a 'requests' call?"
    },
    {
        "agent": "CodeAssistant",
        "question": "Provide a Python example of how to handle a 'requests.exceptions.RequestException' using a try-except block."
    }
]

async def run_evaluation():
    """
    Runs a simple benchmark evaluation for specified agents.
    """
    print("--- Starting Agent Benchmark Evaluation ---")

    # Initialize agents
    # In a real scenario, these would be loaded from your agent management system.
    # For simplicity, we instantiate them directly.
    try:
        research_agent = ResearchAssistant()
        code_agent = CodeAssistant()
        print("✅ Agents initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing agents: {e}")
        return

    agent_map = {
        "ResearchAssistant": research_agent,
        "CodeAssistant": code_agent
    }

    results = []

    for i, item in enumerate(BENCHMARK_DATASET):
        agent_name = item["agent"]
        question = item["question"]
        agent_instance = agent_map.get(agent_name)

        print(f"\n--- Test Case {i+1}/{len(BENCHMARK_DATASET)} ---")
        print(f"Agent: {agent_name}")
        print(f"Question: {question}")
        print("---------------------------------")

        if not agent_instance:
            print(f"❌ Agent '{agent_name}' not found. Skipping.")
            continue

        try:
            # Using the 'run' method which is standard for many agent implementations
            response = await agent_instance.run(question)
            print("✅ Agent Response:")
            print(response)
            results.append({"test_case": i+1, "status": "Success", "response": response})
        except Exception as e:
            error_message = f"❌ An error occurred: {e}"
            print(error_message)
            results.append({"test_case": i+1, "status": "Error", "response": error_message})
        
        print("---------------------------------")

    print("\n--- Evaluation Summary ---")
    successful_tests = sum(1 for r in results if r['status'] == 'Success')
    failed_tests = len(results) - successful_tests
    print(f"Total Tests: {len(results)}")
    print(f"✅ Successful: {successful_tests}")
    print(f"❌ Failed/Errors: {failed_tests}")
    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    # Ensure you have a .env file in the root with your API keys (e.g., GOOGLE_API_KEY)
    if not os.path.exists(os.path.join(os.path.dirname(__file__), '..', '.env')):
        print("❌ Error: .env file not found in the project root.")
        print("Please create one with your LLM API keys (e.g., GOOGLE_API_KEY).")
    else:
        asyncio.run(run_evaluation()) 