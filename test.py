import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains.llm_math.base import LLMMathChain
from langchain_core.prompts import PromptTemplate

def create_and_run_agent(question: str):
    """
    Creates and runs a simple Langchain agent with search and math capabilities.

    Args:
        question (str): The question for the agent to answer.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Ensure API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in a .env file or directly in your environment.")
        return
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found in environment variables.")
        print("Please set it in a .env file or directly in your environment.")
        return

    print("Initializing Langchain agent...")

    # 1. Initialize the Language Model (LLM)
    # We use gpt-4o for better reasoning capabilities, but you can use gpt-3.5-turbo as well.
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 2. Define the tools the agent can use
    # Tool 1: Tavily Search for internet browsing
    tavily_tool = TavilySearchResults(max_results=3) # Limit results for brevity

    # Tool 2: LLM Math Chain for calculations
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        Tool(
            name="Search",
            func=tavily_tool.run,
            description="useful for when you need to answer questions about current events or facts. Input should be a search query."
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math. Input should be a mathematical expression."
        )
    ]

    # 3. Define the prompt template for the agent
    # This prompt guides the agent's reasoning process.
    # It tells the agent what its goal is, what tools it has, and how to format its thoughts.
    prompt = PromptTemplate.from_template("""
    You are a helpful AI assistant. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """)

    # 4. Create the agent
    # We use create_react_agent which implements the ReAct framework.
    agent = create_react_agent(llm, tools, prompt)

    # 5. Create the AgentExecutor
    # The AgentExecutor is responsible for running the agent, managing its steps,
    # and handling tool execution.
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    print(f"\nRunning agent with question: '{question}'")
    try:
        response = agent_executor.invoke({"input": question})
        print("\n--- Agent's Final Answer ---")
        print(response["output"])
    except Exception as e:
        print(f"\nAn error occurred while running the agent: {e}")
        print("Please check your API keys and internet connection.")

if __name__ == "__main__":
    # Example usage:
    # This question requires both search (for current year) and calculation.
    create_and_run_agent("What is the current year minus 1999, and what is the capital of France?")

    # Another example: A simple math question
    # create_and_run_agent("What is 12345 * 6789?")

    # Another example: A simple search question
    # create_and_run_agent("Who won the last Super Bowl?")