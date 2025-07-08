import os
import sys
import asyncio
from flask import Flask, request, jsonify

# Ensure the project's src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables (ensure .env is in the project root)
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))

# Import your agent and LLM wrappers
from src.GAAPF.core.agents.research_assistant import ResearchAssistantAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI

# --- Flask App for Agent Integration ---
app = Flask(__name__)

# Global variable to hold the agent instance
agent = None

def initialize_agent():
    """Initializes the ResearchAssistant with a specified LLM provider."""
    global agent
    if agent is None:
        try:
            llm = None
            # Read the desired provider from environment variables
            provider = os.getenv("LLM_PROVIDER", "google-genai").lower()

            print(f"Attempting to initialize agent with provider: {provider}")

            if provider == "vertex-ai":
                # Ensure the project ID is set for Vertex AI
                if not os.getenv("GOOGLE_CLOUD_PROJECT"):
                    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set for Vertex AI")
                print("Using Vertex AI LLM...")
                llm = ChatVertexAI(model_name="gemini-pro", temperature=0.7)
            else: # Default to google-genai
                print("Using Google Generative AI LLM...")
                llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

            print("Initializing ResearchAssistant...")
            # The agent expects the llm instance directly
            agent = ResearchAssistantAgent(llm=llm)
            print("✅ ResearchAssistant initialized successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize ResearchAssistant: {e}")
            # Exit if the agent cannot be created
            sys.exit("Agent initialization failed. Check API keys and dependencies.")
    return agent

@app.route('/ask', methods=['POST'])
def ask_agent():
    """
    Endpoint to receive a question and return the agent's response.
    AgentBench will call this endpoint.
    """
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data['question']
    print(f"Received question: {question}")
    
    agent_instance = initialize_agent()
    
    try:
        # Agents are invoked with a dictionary input
        response = agent_instance.invoke({"input": question})
        
        # The actual response is typically in an 'output' key
        output = response.get("output", "No output found.")

        print(f"Sending response: {output}")
        return jsonify({"response": output})

    except Exception as e:
        error_message = f"An error occurred while processing the question: {e}"
        print(f"❌ {error_message}")
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    # Initialize the agent on startup
    initialize_agent()
    # Start the Flask server
    # AgentBench controller will connect to this port
    app.run(host='0.0.0.0', port=8000) 