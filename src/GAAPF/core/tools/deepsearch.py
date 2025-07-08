import os
import re
from typing import TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_together import ChatTogether
from tavily import TavilyClient

load_dotenv()

# Move model initialization to a function to avoid import-time errors
def get_model():
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable is not set")
    return ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        api_key=api_key
    )

def get_tavily_client():
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    return TavilyClient(api_key=api_key)

# Initialize lazily to avoid import-time errors
model = None
tavily = None


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    adjustment: str
    sections: list[str]
    chapters: list[str]
    revision_number: int
    max_revisions: int
    max_chapters: int = 5
    max_paragraphs_per_chapter: int = 5
    max_critical_queries: int = 5
    number_of_chapters: int
    current_chapter_order: int


class TheadModel(BaseModel):
    class Configurable(BaseModel):
        thread_id: str

    configurable: Configurable


class DeepSearch:
    """DeepSearch class implements deep search feature with external search calling"""

    builder: StateGraph = StateGraph(AgentState)

    PLAN_PROMPT: str = """You are an expert writer tasked with writing a high level outline of an analytical essay on the topic. \
    Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
    or instructions for the chapters. Not more than {max_chapters} chapters. The output should be in the following format:
    1. Chapter 1
    2. Chapter 2
    ...
    """

    WRITER_PROMPT: str = """You are an researcher assistant tasked with writing excellent {max_paragraphs_per_chapter} paragraph research article.\
    Generate the best research possible for the chapter based on user's collected information. \
    If the user provides critique and suggested adjustment, respond with a revised version of your previous content. \
    The article should include comparisions, statistics data, and references to make clear the arguments. \
    Directly generate without any explanation. \
    Having a various conclusion expression. \
    Utilize all the information below as needed: \

    ------
    - Previous content:
    {content}
    - Critique:
    {critique}
    - Suggested Adjustment:
    {adjustment}
    """

    REFLECTION_PROMPT: str = """You are a teacher grading an research submission. \
    Generate critique and recommendations for the user's submission. \
    Provide detailed recommendations, including requests for coherence & cohension, lexical resource, task achievement, comparison, statistics data. \
    Only generate critique and recommendations less than 200 words."""

    RESEARCH_CRITIQUE_PROMPT: str = """
    You are a researcher charged with critiquing information as outlined below. \
    Generate a list of search queries that will gather any relevant information. Only generate maximum {max_critical_queries} queries.
    """

    def __init__(self):
        self.builder = StateGraph(AgentState)
        self.builder.add_node("planner", self.plan_node)
        self.builder.add_node("generate", self.generation_node)
        self.builder.add_node("reflect", self.reflection_node)
        self.builder.add_node("research_critique", self.research_critique_node)
        self.builder.set_entry_point("planner")
        self.builder.add_conditional_edges(
            "generate", self.should_continue, {END: END, "reflect": "reflect"}
        )
        self.builder.add_edge("planner", "generate")
        self.builder.add_edge("reflect", "research_critique")
        self.builder.add_edge("research_critique", "generate")
        memory = MemorySaver()
        self.graph = self.builder.compile(checkpointer=memory)

    def plan_node(self, state: AgentState):
        print("----------------------------------")
        max_chapters = state.get("max_chapters", 5)
        messages = [
            SystemMessage(content=self.PLAN_PROMPT.format(max_chapters=max_chapters)),
            HumanMessage(content=state["task"]),
        ]
        response = get_model().invoke(messages)

        def find_section(text: str) -> bool:
            is_match = re.match("^\d+. ", text)
            return is_match is not None

        list_tasks = [
            task
            for task in response.content.split("\n\n")
            if task != "" and find_section(task)
        ]
        return {
            "plan": list_tasks,
            "current_chapter_order": 0,
            "number_of_chapters": len(list_tasks),
        }

    def generation_node(self, state: AgentState):
        current_chapter_order = state["current_chapter_order"]
        chapter_outline = state["plan"][current_chapter_order]
        queries = [query.strip() for query in chapter_outline.split("\n")[1:]]
        chapter_title = chapter_outline.split("\n")[0].strip()
        sections = state.get("sections", [])
        print("----------------------------------")
        print(chapter_title)
        if chapter_title not in sections:
            sections.append(chapter_title)
        content = []

        for q in queries:
            response = get_tavily_client().search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
                if q not in sections:
                    sections.append(q)

        adjustment = state["adjustment"] if "adjustment" in state else []
        critique = state["critique"] if "critique" in state else []
        max_paragraphs_per_chapter = state.get("max_paragraphs_per_chapter", 5)
        user_message = HumanMessage(
            content=f"Chapter outline: {chapter_outline}\n\nHere is the collected information for this chaper:\n\n{' '.join(content)}"
        )
        messages = [
            SystemMessage(
                content=self.WRITER_PROMPT.format(
                    max_paragraphs_per_chapter=max_paragraphs_per_chapter,
                    content=content,
                    critique=critique,
                    adjustment=adjustment,
                )
            ),
            user_message,
        ]
        response = get_model().invoke(messages)
        chapters = state["chapters"] if "chapters" in state else []
        chapters.append(f"{chapter_title} \n {response.content}")
        print("revision_number: ", state.get("revision_number", 1))
        if (
            state.get("revision_number", 1) >= state["max_revisions"]
        ):  # exceed revision number per chapter
            current_chapter_order = state.get("current_chapter_order", 0) + 1
            revision_number = 1
        else:
            revision_number = state["revision_number"] + 1

        return {
            "chapters": chapters,
            "draft": response.content,
            "revision_number": revision_number,
            "current_chapter_order": current_chapter_order,
            "sections": sections,
        }

    def reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=self.REFLECTION_PROMPT),
            HumanMessage(content=state["draft"]),
        ]
        response = get_model().invoke(messages)
        return {"critique": response.content}

    def should_continue(self, state: AgentState):
        if state["current_chapter_order"] == state["number_of_chapters"]:
            return END
        return "reflect"

    def research_critique_node(self, state: AgentState):
        critique = get_model().invoke(
            [
                SystemMessage(
                    content=self.RESEARCH_CRITIQUE_PROMPT.format(
                        max_critical_queries=state.get("max_critical_queries", 5)
                    )
                ),
                HumanMessage(content=f"Overall critique: \n{state['critique']}"),
            ]
        )

        def find_query(text: str) -> bool:
            is_match = re.match("^\d+. ", text)
            return is_match is not None

        queries = [query for query in critique.content.split("\n") if find_query(query)]
        content = []
        for q in queries:
            match = re.search(r'"([^"]+)"', q)
            if match:
                q = match.group(1)

            response = get_tavily_client().search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
        return {"adjustment": content}

    def streaming_response(
        self,
        query: str,
        thread: TheadModel = {"configurable": {"thread_id": "1"}},
        max_chapters: int = 5,
        max_paragraphs_per_chapter: int = 5,
        max_critical_queries: int = 5,
        max_revisions: int = 1,
    ):
        for s in self.graph.stream(
            {
                "task": query,
                "max_chapters": max_chapters,
                "max_paragraphs_per_chapter": max_paragraphs_per_chapter,
                "max_critical_queries": max_critical_queries,
                "max_revisions": max_revisions,
                "revision_number": 1,
            },
            thread,
        ):
            print(f"Agent name: {list(s.keys())[0]} : {list(s.values())[0]}")

        plans = "\n".join(self.graph.get_state(thread).values["sections"])
        chapters = "## " + "\n\n## ".join(
            self.graph.get_state(thread).values["chapters"]
        )
        content = f"# I. Planning\n{plans}\n\n# II. Results\n{chapters}"
        return content


def deepsearch_tool(
    query: str,
    max_chapters: int = 4,
    max_paragraphs_per_chapter: int = 5,
    max_critical_queries: int = 3,
    max_revisions: int = 1,
):
    """Invoke deepsearch to deeply analyze the query and generate a more detailed response.
    Args:
        query (str): The query to analyze.
        max_chapters (int, optional): The maximum number of chapters to generate.
        max_paragraphs_per_chapter (int, optional): The maximum number of paragraphs per chapter.
        max_critical_queries (int, optional): The maximum number of critical queries to generate.
        max_revisions (int, optional): The maximum number of revisions to generate.
    Returns:
        str: The detailed response generated by deepsearch.
    """
    deepsearch = DeepSearch()
    content = deepsearch.streaming_response(
        query=query,
        max_chapters=max_chapters,
        max_paragraphs_per_chapter=max_paragraphs_per_chapter,
        max_critical_queries=max_critical_queries,
        max_revisions=max_revisions,
    )
    return content


# content = deepsearch_tool(
#     query="Analyzing tesla stock price",
#     max_chapters=4,
#     max_paragraphs_per_chapter=5,
#     max_critical_queries=3,
#     max_revisions=2
# )
