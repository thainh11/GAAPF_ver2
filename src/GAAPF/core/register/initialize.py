from langchain_together import ChatTogether
from dotenv import load_dotenv
import os

load_dotenv()

# Get API key from environment
api_key = os.environ.get("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY environment variable is not set")

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    api_key=api_key
)
