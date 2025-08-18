import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st

# Ensure project src path is available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../vinagent-main
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Windows asyncio policy to avoid event loop issues
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Import GAAPF modules (reuse existing implementations)
from GAAPF.core.utils.credentials_helper import get_vertex_ai_config
from GAAPF.core.core.simple_hub import SimpleLearningHub  # Core orchestration
from GAAPF.core.agents.instructor import InstructorAgent
from GAAPF.core.agents.code_assistant import CodeAssistantAgent
from GAAPF.core.agents.practice_facilitator import PracticeFacilitatorAgent
from GAAPF.core.agents.socratic_instructor import SocraticInstructorAgent
from GAAPF.core.tools.framework_collector import FrameworkCollector

# Vertex AI LLM
from langchain_google_vertexai import ChatVertexAI
# Vertex AI LLM import will be done lazily inside init_llm to allow fallback if not installed

# -----------------------------
# Helpers
# -----------------------------
FRAMEWORKS = ["langchain", "langgraph", "crewai", "autogen"]


def init_llm():
    """Initialize ChatVertexAI using the shared credential helper.
    Falls back to a simple FakeListLLM for local/demo when Vertex AI isn't configured.
    """
    try:
        cfg = get_vertex_ai_config()
        return ChatVertexAI(
            model_name=cfg["model_name"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            project=cfg["project"],
            location=cfg["location"],
        )
    except Exception:
        # Fallback mock LLM for demo/testing (no external provider)
        try:
            from langchain_core.language_models.fake import FakeListLLM
            return FakeListLLM(responses=[
                "Hello! I'm your AI learning assistant. How can I help you today?",
                "I understand you're learning. Let me help you with that.",
                "That's a great question! Let me explain...",
                "I'm here to help you learn step by step."
            ])
        except Exception:
            class MockLLM:
                def invoke(self, prompt):
                    return "Hello! This is a mock response. Please configure Vertex AI."
                def __call__(self, prompt):
                    return self.invoke(prompt)
            return MockLLM()


def build_agents(llm, framework: str):
    """Create per-framework agents with shared memory file path."""
    mem_file = Path(f"templates/memory_{framework}.json")
    return {
        "instructor": InstructorAgent(llm, is_logging=True, memory_path=mem_file),
        "code_assistant": CodeAssistantAgent(llm, is_logging=True, memory_path=mem_file),
        "practice": PracticeFacilitatorAgent(llm, is_logging=True, memory_path=mem_file),
        "socratic_instructor": SocraticInstructorAgent(llm, is_logging=True, memory_path=mem_file),
    }


def ensure_framework_ingested(framework: str):
    try:
        collector = FrameworkCollector(is_logging=False)
        collector.ensure_ingested(framework, persistent_dir="data/frameworks/vectordb")
    except Exception as e:
        st.sidebar.caption(f"(RAG init failed: {e})")


async def async_process(hub: SimpleLearningHub, query: str, context: Dict[str, Any]):
    return await hub.process_query(query, context)


def process_query_sync(hub: SimpleLearningHub, query: str, context: Dict[str, Any]):
    """Run the hub's async process_query in a synchronous Streamlit callback."""
    try:
        return asyncio.run(async_process(hub, query, context))
    except RuntimeError:
        # If an event loop is already running (rare in Streamlit), create a new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_process(hub, query, context))
        finally:
            loop.close()


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="GAAPF - Learning Assistant", page_icon="🤖", layout="wide")

# Sidebar controls
with st.sidebar:
    st.title("GAAPF")
    st.caption("Guidance AI Agent for Python Frameworks")

    # User ID
    user_id = st.text_input("User ID", value=st.session_state.get("user_id", "default"))
    st.session_state["user_id"] = user_id or "default"

    # Framework selection
    current_framework = st.selectbox(
        "Framework",
        FRAMEWORKS,
        index=FRAMEWORKS.index(st.session_state.get("framework", "langchain"))
        if st.session_state.get("framework", "langchain") in FRAMEWORKS else 0,
    )

    # Study mode toggle
    study_mode = st.toggle("Enable Study Mode (Socratic)", value=st.session_state.get("study_mode", True))

    # Clear chat button
    if st.button("Clear chat"):
        st.session_state["messages"] = []

    st.markdown("---")
    st.caption("Vertex AI is used as the LLM provider. Configure via environment variables.")

# Initialize LLM and Hub once
if "llm" not in st.session_state:
    try:
        st.session_state["llm"] = init_llm()
        # Inform about which LLM is being used
        if isinstance(st.session_state["llm"], ChatVertexAI):
            st.toast("Using Google Vertex AI LLM", icon="🤖")
        else:
            st.toast("Using mock LLM (configure Vertex AI for real responses)", icon="🧪")
    except Exception as e:
        st.error(f"LLM initialization failed: {e}")
        st.stop()

# Initialize hub and agents if not present or framework changed
framework_changed = st.session_state.get("framework") != current_framework
if ("hub" not in st.session_state) or framework_changed:
    ensure_framework_ingested(current_framework)
    agents = build_agents(st.session_state["llm"], current_framework)
    hub = SimpleLearningHub(st.session_state["llm"], agents, is_logging=True)
    hub.set_framework(current_framework)
    st.session_state["hub"] = hub
    st.session_state["agents"] = agents
    st.session_state["framework"] = current_framework

# Study mode setting
st.session_state["study_mode"] = study_mode
if study_mode:
    st.session_state["hub"].enable_study_mode()
else:
    st.session_state["hub"].disable_study_mode()

# Messages history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("GAAPF - Learning Assistant")
st.caption(f"Ask anything about {st.session_state['framework'].title()}.")

# Display existing messages
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
prompt = st.chat_input("Type your question...")

if prompt:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare context
    context = {
        "framework": st.session_state["framework"],
        "study_mode": st.session_state["study_mode"],
        "interaction_count": sum(1 for m in st.session_state["messages"] if m["role"] == "user"),
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = process_query_sync(st.session_state["hub"], prompt, context)
                content = resp.get("content", "")
                st.markdown(content)

                # Optional suggestions
                suggestions: List[str] = resp.get("suggestions", []) or []
                if suggestions:
                    with st.expander("Suggestions"):
                        for s in suggestions[:3]:
                            st.markdown(f"- {s}")

                # Save assistant message
                st.session_state["messages"].append({"role": "assistant", "content": content})
            except Exception as e:
                st.error(f"Error: {e}")