"""
Simplified Learning Hub - Core orchestration for GAAPF
Replaces the complex 1326-line learning_hub.py with essential functionality only.

This module provides the core learning orchestration with:
1. Simple agent coordination (3 agents only)
2. Basic session management
3. LLM-orchestrated routing with a light LangGraph study graph
4. Study mode support
"""

import logging
import json
from typing import Dict, List, Any, TypedDict
from langchain_core.language_models.base import BaseLanguageModel
from ..graph.function_graph import FunctionStateGraph, NodeWrapper
from ..graph.constants import START, END

# Setup logging (avoid duplicate basicConfig)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudyState(TypedDict, total=False):
    """Minimal state for the Study Graph routing.
    - query/context are inputs
    - content/agent_used are outputs from stage nodes
    - next is an internal hint, not required
    """
    query: str
    context: dict
    content: str
    agent_used: str
    next: str

class SimpleLearningHub:
    """
    Simplified learning hub focusing on core functionality:
    1. Agent coordination (3 agents only)
    2. Basic session management
    3. Simple routing logic
    4. Study mode support (prepared for Phase 2)
    
    This replaces the complex 1326-line learning_hub.py with essential
    functionality only, reducing complexity by 85%.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        agents: Dict[str, Any],
        is_logging: bool = False,
        shared_memory_path: str = "templates/memory_session.json"
    ):
        """
        Initialize the simplified learning hub.
        
        Args:
            llm: Language model for AI interactions
            agents: Dictionary of 3 core agents (instructor, code_assistant, practice)
            is_logging: Enable detailed logging
            shared_memory_path: Path for shared memory across all agents
        """
        self.llm = llm
        self.agents = agents  # Only 3 agents: instructor, code_assistant, practice
        self.is_logging = is_logging
        self.study_mode = False  # Study mode flag (for Phase 2)
        self.lt_memory = None  # Long-term memory per framework
        self._study_graph_compiled = None  # Compiled study graph (LangGraph)
        
        # Initialize shared session memory for all agents (framework-independent)
        try:
            from pathlib import Path
            from ..memory.memory import Memory
            # Ensure consistent session memory path regardless of framework
            session_memory_path = Path("templates/memory_session.json")
            self.shared_memory = Memory(
                memory_path=session_memory_path,
                is_reset_memory=False
            )
            if self.is_logging:
                logger.info(f"🧠 Shared session memory initialized at {session_memory_path}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize shared session memory: {e}")
            self.shared_memory = None
        
        # Simple session tracking
        self.current_session = {
            "user_id": "default",
            "framework": "langchain",
            "interaction_count": 0,
            "start_time": None
        }
        
        # Update all agents to use shared memory
        self._update_agents_memory()
        
        if self.is_logging:
            logger.info(f"SimpleLearningHub initialized with {len(self.agents)} agents")
            logger.info(f"Available agents: {list(self.agents.keys())}")
    
    def _update_agents_memory(self):
        """Update all agents to use the shared memory instance."""
        if self.shared_memory is None:
            return
            
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'memory'):
                agent.memory = self.shared_memory
                if self.is_logging:
                    logger.info(f"🔗 Updated {agent_name} to use shared memory")
        
    async def process_query(
        self,
        query: str,
        context: Dict[str, Any] = None,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process user query with simplified routing logic.
        
        Routing Rules:
        - Code-related keywords → CodeAssistantAgent
        - Practice/exercise keywords → PracticeFacilitatorAgent  
        - Everything else → InstructorAgent (with optional Study Mode)
        
        Args:
            query: User's question or request
            context: Additional context information
            user_id: User identifier
            
        Returns:
            Response dictionary with content and metadata
        """
        if context is None:
            context = {}
            
        # Update session info
        self.current_session["user_id"] = user_id
        self.current_session["interaction_count"] += 1
        
        # Manage simple onboarding stage progression when in study mode
        try:
            if self.study_mode:
                # Initialize stage if missing
                if "onboarding_stage" not in self.current_session:
                    self.current_session["onboarding_stage"] = "concept_basics"
                else:
                    # Progress with interaction_count: 1 -> concept_basics, 2 -> simple_tool, 3 -> build_agent
                    ic = int(self.current_session.get("interaction_count", 0) or 0)
                    st = self.current_session.get("onboarding_stage")
                    if st == "concept_basics" and ic >= 2:
                        self.current_session["onboarding_stage"] = "simple_tool"
                    elif st == "simple_tool" and ic >= 3:
                        self.current_session["onboarding_stage"] = "build_agent"
            else:
                # Clear onboarding stage if not in study mode
                if "onboarding_stage" in self.current_session:
                    del self.current_session["onboarding_stage"]
        except Exception:
            pass
        
        # Enhance context with study mode info and per-framework curriculum/config
        enhanced_context = {
            **context,
            "study_mode": self.study_mode,
            "session": self.current_session,
            "framework": context.get("framework", self.current_session["framework"]),
        }
        # Inject framework config and curriculum if available
        try:
            fw = enhanced_context.get("framework", "langchain").lower()
            # Load lightweight cached framework config if present
            from pathlib import Path as _Path
            cfg_path_new = _Path(f"data/frameworks/cache/{fw}.json")
            cfg_path_old = _Path(f"data/framework_cache/framework_init/{fw}.json")
            framework_config = None
            for p in [cfg_path_new, cfg_path_old]:
                if p.exists():
                    try:
                        import json as _json
                        framework_config = _json.loads(p.read_text(encoding="utf-8"))
                        break
                    except Exception:
                        pass
            if framework_config:
                enhanced_context["framework_config"] = framework_config
            # Load dynamic curriculum if available
            cur_path = _Path("data/curriculums/dynamic_curriculum_langchain.json") if fw == "langchain" else None
            if cur_path and cur_path.exists():
                try:
                    import json as _json
                    enhanced_context["curriculum"] = _json.loads(cur_path.read_text(encoding="utf-8"))
                    # Bridge curriculum (list-based) to framework_config.modules (dict-based)
                    try:
                        curriculum = enhanced_context.get("curriculum") or {}
                        fc = enhanced_context.get("framework_config") or {}
                        modules_map = fc.get("modules", {}) if isinstance(fc.get("modules", {}), dict) else {}
                        cur_modules = curriculum.get("modules") if isinstance(curriculum, dict) else None
                        if isinstance(cur_modules, list):
                            # Build modules map from curriculum list
                            for idx, mod in enumerate(cur_modules, start=1):
                                title = str(mod.get("title", "")).strip()
                                key = (
                                    title.lower()
                                    .replace(":", "")
                                    .replace("-", " ")
                                    .replace("/", " ")
                                    .replace(" ", "_")
                                ) or f"module_{idx}"
                                # Extract concepts from topics[].resources like "Concept: X"
                                concepts: List[str] = []
                                for t in mod.get("topics", []) or []:
                                    for res in t.get("resources", []) or []:
                                        if isinstance(res, str) and res.lower().startswith("concept:"):
                                            concepts.append(res.split(":", 1)[1].strip())
                                modules_map[key] = {
                                    "title": title or key,
                                    "description": mod.get("description", ""),
                                    "concepts": concepts,
                                }
                            # Ensure framework name present
                            if not fc.get("name") and curriculum.get("framework"):
                                fc["name"] = str(curriculum.get("framework")).title()
                            fc["modules"] = modules_map
                            enhanced_context["framework_config"] = fc
                            # Default current module if not set
                            if not enhanced_context.get("current_module"):
                                try:
                                    enhanced_context["current_module"] = next(iter(modules_map.keys()))
                                except StopIteration:
                                    pass
                            # Propagate user level if provided
                            if not enhanced_context.get("user_level") and curriculum.get("user_level"):
                                enhanced_context["user_level"] = str(curriculum.get("user_level")).lower()
                            # Provide a compact curriculum summary for LLM consumption
                            try:
                                lines: List[str] = []
                                for idx, (k2, v2) in enumerate(modules_map.items()):
                                    if idx >= 3:
                                        break
                                    title2 = (v2.get("title") or k2).strip()
                                    concepts2 = ", ".join((v2.get("concepts") or [])[:3])
                                    lines.append(f"- {title2}: {concepts2}" if concepts2 else f"- {title2}")
                                if lines:
                                    enhanced_context["curriculum_summary"] = "\n".join(lines)
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass
        
        try:
            selected_agent = None
            # If study mode is enabled, run through the light Study Graph (LangGraph)
            if self.study_mode:
                self._ensure_study_graph()
                initial_state: StudyState = {"query": query, "context": enhanced_context}
                result_state: Dict[str, Any]
                if hasattr(self._study_graph_compiled, "ainvoke"):
                    result_state = await self._study_graph_compiled.ainvoke(initial_state)
                else:
                    result_state = self._study_graph_compiled.invoke(initial_state)

                response_dict = {
                    "content": result_state.get("content", ""),
                    "agent_used": result_state.get("agent_used", "socratic_instructor"),
                    "study_mode": True,
                    "interaction_count": self.current_session["interaction_count"],
                    # Preserve structured fields from agent response/state when available
                    "suggestions": result_state.get("suggestions", []),
                    "type": result_state.get("type"),
                    "mode": result_state.get("mode"),
                    "progress": result_state.get("progress"),
                }
                # Ensure selected_agent is set for logging and session tracking
                selected_agent = response_dict["agent_used"]
            else:
                # Non-study path: simple routing logic based on keywords
                selected_agent = self._route_to_agent(query)
                if self.is_logging:
                    logger.info(f"Query: '{query[:50]}...' routed to {selected_agent}")

                agent = self.agents[selected_agent]
                response = await agent.ainvoke(query, is_save_memory=True, user_id=user_id, learning_context=enhanced_context)

                # Normalize response to dict
                content_value = response
                try:
                    from langchain_core.messages import AIMessage
                    if isinstance(response, AIMessage):
                        content_value = response.content
                except Exception:
                    pass
                if isinstance(content_value, str):
                    response_dict = {"content": content_value}
                elif isinstance(response, dict):
                    response_dict = dict(response)
                else:
                    response_dict = {"content": str(content_value)}

                # Add metadata
                response_dict.update({
                    "agent_used": selected_agent,
                    "study_mode": False,
                    "interaction_count": self.current_session["interaction_count"],
                })

            # Track last exchange for quick recall in UI
            try:
                self.current_session["last_exchange"] = {
                    "user": query,
                    "assistant": response_dict.get("content", ""),
                    "agent": selected_agent,
                }
            except Exception:
                pass
            
            if self.is_logging:
                logger.info(f"Response generated by {selected_agent}")

            # Periodic summarization to long-term memory every 10 interactions
            try:
                if self.current_session["interaction_count"] % 10 == 0:
                    await self._summarize_and_store(user_id, enhanced_context)
            except Exception as _e:
                if self.is_logging:
                    logger.warning(f"LT summary failed: {_e}")

            return response_dict
            
        except Exception as e:
            error_msg = f"Error processing query with {selected_agent}: {str(e)}"
            logger.error(error_msg)
            
            return {
                "content": "I apologize, but I encountered an error processing your request. Please try rephrasing your question.",
                "error": error_msg,
                "agent_used": "unknown",
                "study_mode": self.study_mode
            }

    def _ensure_study_graph(self):
        if self._study_graph_compiled:
            return
        graph = FunctionStateGraph(state_schema=StudyState, config_schema=None)

        route_node = NodeWrapper(self._route_stage, branching=self._route_branch, name="route")
        onboarding = NodeWrapper(self._node_onboarding, name="onboarding")
        scoping = NodeWrapper(self._node_scoping, name="scoping")
        build = NodeWrapper(self._node_build, name="build")
        direct = NodeWrapper(self._node_direct, name="direct")

        flow = [
            START >> route_node,
            # Route to one of the stages, then finish
            route_node >> {"onboarding": onboarding, "scoping": scoping, "build": build, "direct": direct},
            onboarding >> END,
            scoping >> END,
            build >> END,
            direct >> END,
        ]
        self._study_graph_compiled = graph.compile(flow=flow)

    def _route_stage(self, state: StudyState) -> Dict[str, Any]:
        query = state.get("query", "")
        ctx = state.get("context", {})
        # Heuristic: if user mentions build/code/agent, bias to build stage
        ql = (query or "").lower()
        # Respect onboarding stage guidance for Study Mode
        try:
            session = ctx.get("session", {}) or {}
            onboarding_stage = session.get("onboarding_stage")
            if ctx.get("study_mode") and onboarding_stage:
                if onboarding_stage in {"concept_basics", "simple_tool"}:
                    ctx["_orchestrated_agent"] = "socratic_instructor"
                    state["context"] = ctx
                    state["next"] = "onboarding"
                    return state
                if onboarding_stage == "build_agent":
                    ctx["_orchestrated_agent"] = "code_assistant"
                    state["context"] = ctx
                    state["next"] = "build"
                    return state
        except Exception:
            pass
        if any(k in ql for k in ["build", "agent", "example", "code", "implement", "create"]):
            ctx["_orchestrated_agent"] = "code_assistant"
            state["context"] = ctx
            state["next"] = "build"
            return state
        try:
            prompt = (
                "Route the user request to one agent. Agents: [\"instructor\",\"code_assistant\",\"practice\",\"socratic_instructor\"].\n"
                "Stages: [\"onboarding\",\"scoping\",\"build\",\"direct\"].\n"
                f"Context: {{\"framework\": \"{ctx.get('framework','')}\", \"interaction_count\": {int(ctx.get('interaction_count', 0) or 0)} }}\n"
                f"User: \"{query}\"\n"
                "Return ONLY JSON: {\"agent\":\"...\",\"stage\":\"onboarding|scoping|build|direct\"}"
            )
            resp = self.llm.invoke(prompt)
            data = json.loads(getattr(resp, "content", str(resp)).strip())
            agent = data.get("agent")
            stage = data.get("stage")
            if agent in {"instructor", "code_assistant", "practice", "socratic_instructor"} and stage in {"onboarding", "scoping", "build", "direct"}:
                # keep agent hint in context for stage nodes if needed
                ctx["_orchestrated_agent"] = agent
                state["context"] = ctx
                state["next"] = stage
                return state
        except Exception:
            pass
        # Fallback heuristic
        stage = "onboarding" if self.current_session["interaction_count"] <= 2 else "scoping"
        state["next"] = stage
        return state

    def _route_branch(self, state: StudyState) -> str:
        # Read the desired next stage written by _route_stage
        return state.get("next", "onboarding")

    def _node_onboarding(self, state: StudyState) -> StudyState:
        q, ctx = state.get("query", ""), state.get("context", {})
        ctx["stage"] = "onboarding"
        agent = self.agents.get("socratic_instructor")
        res = agent.invoke(q, is_save_memory=True, user_id=ctx.get("session", {}).get("user_id", "default"), learning_context=ctx)
        # Preserve structured fields if the agent returns a dict
        if isinstance(res, dict):
            state["content"] = res.get("content", "")
            state["agent_used"] = "socratic_instructor"
            if res.get("suggestions") is not None:
                state["suggestions"] = res.get("suggestions")
            if res.get("type") is not None:
                state["type"] = res.get("type")
            if res.get("mode") is not None:
                state["mode"] = res.get("mode")
            if res.get("progress") is not None:
                state["progress"] = res.get("progress")
        else:
            content = getattr(res, "content", res if isinstance(res, str) else str(res))
            state["content"] = content
            state["agent_used"] = "socratic_instructor"
        return state

    def _node_scoping(self, state: StudyState) -> StudyState:
        q, ctx = state.get("query", ""), state.get("context", {})
        ctx["stage"] = "scoping"
        agent = self.agents.get("socratic_instructor")
        res = agent.invoke(q, is_save_memory=True, user_id=ctx.get("session", {}).get("user_id", "default"), learning_context=ctx)
        if isinstance(res, dict):
            state["content"] = res.get("content", "")
            state["agent_used"] = "socratic_instructor"
            if res.get("suggestions") is not None:
                state["suggestions"] = res.get("suggestions")
            if res.get("type") is not None:
                state["type"] = res.get("type")
            if res.get("mode") is not None:
                state["mode"] = res.get("mode")
            if res.get("progress") is not None:
                state["progress"] = res.get("progress")
        else:
            content = getattr(res, "content", res if isinstance(res, str) else str(res))
            state["content"] = content
            state["agent_used"] = "socratic_instructor"
        return state

    def _node_build(self, state: StudyState) -> StudyState:
        q, ctx = state.get("query", ""), state.get("context", {})
        ctx["stage"] = "build"
        agent = self.agents.get("code_assistant")
        res = agent.invoke(q, is_save_memory=True, user_id=ctx.get("session", {}).get("user_id", "default"), learning_context=ctx)
        content = getattr(res, "content", res if isinstance(res, str) else str(res))
        state["content"] = content
        state["agent_used"] = "code_assistant"
        return state

    def _node_direct(self, state: StudyState) -> StudyState:
        q, ctx = state.get("query", ""), state.get("context", {})
        ctx["stage"] = "direct"
        agent = self.agents.get("instructor")
        res = agent.invoke(q, is_save_memory=True, user_id=ctx.get("session", {}).get("user_id", "default"), learning_context=ctx)
        content = getattr(res, "content", res if isinstance(res, str) else str(res))
        state["content"] = content
        state["agent_used"] = "instructor"
        return state
    
    def _route_to_agent(self, query: str) -> str:
        """
        Simple routing logic to select appropriate agent.
        
        Args:
            query: User's query
            
        Returns:
            Agent name to handle the query
        """
        query_lower = query.lower()
        
        # Code-related keywords
        code_keywords = [
            "code", "implement", "debug", "error", "function", 
            "class", "method", "variable", "syntax", "bug",
            "write code", "create function", "fix", "program"
        ]
        
        # Practice-related keywords  
        practice_keywords = [
            "practice", "exercise", "quiz", "test", "challenge",
            "try", "attempt", "hands-on", "example", "demo"
        ]
        
        # Check for code-related queries
        if any(keyword in query_lower for keyword in code_keywords):
            return "code_assistant"
            
        # Check for practice-related queries
        elif any(keyword in query_lower for keyword in practice_keywords):
            return "practice"
            
        # Default to instructor for explanations, concepts, etc.
        else:
            return "instructor"
    
    def enable_study_mode(self):
        """
        Enable Socratic Study Mode (Phase 2 feature).
        In Study Mode, agents guide learning through questions instead of direct answers.
        """
        self.study_mode = True
        if self.is_logging:
            logger.info("🤔 Study Mode activated - Learning through questions!")
    
    def disable_study_mode(self):
        """
        Disable Study Mode - return to direct answers.
        """
        self.study_mode = False
        if self.is_logging:
            logger.info("📚 Direct Mode activated - Learning through answers!")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        return self.current_session.copy()
    
    def reset_session(self):
        """Reset current session."""
        self.current_session = {
            "user_id": "default",
            "framework": "langchain", 
            "interaction_count": 0,
            "start_time": None
        }
        if self.is_logging:
            logger.info("Session reset")
    
    def set_framework(self, framework: str):
        """Set the current learning framework and initialize framework-specific long-term memory."""
        self.current_session["framework"] = framework
        if self.is_logging:
            logger.info(f"Framework set to: {framework}")
        # Initialize per-framework Long-Term Memory (separate from session memory)
        try:
            from pathlib import Path
            from ..memory.long_term_memory import LongTermMemory
            # Use standardized path structure for long-term memory
            lt_mem_file = Path(f"memory/lt_{framework}.json")
            lt_chroma_path = Path(f"memory/chroma_db/{framework}")
            # Ensure memory directory exists
            lt_mem_file.parent.mkdir(parents=True, exist_ok=True)
            lt_chroma_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.lt_memory = LongTermMemory(
                memory_path=lt_mem_file,
                chroma_path=lt_chroma_path,
                collection_name=f"lt_{framework}",
                is_logging=self.is_logging,
            )
            if self.is_logging:
                logger.info(f"📚 Long-term memory initialized for {framework} at {lt_mem_file}")
        except Exception as e:
            if self.is_logging:
                logger.warning(f"Failed to initialize LT memory for {framework}: {e}")

    async def _summarize_and_store(self, user_id: str, context: Dict[str, Any]):
        """Summarize recent conversation and store into long-term memory for this framework."""
        if not getattr(self, "lt_memory", None):
            return
        framework = context.get("framework", self.current_session.get("framework", ""))

        # pick an agent with memory
        pick = None
        for name in ["socratic_instructor", "instructor", "code_assistant", "practice"]:
            if name in self.agents and getattr(self.agents[name], "memory", None):
                pick = self.agents[name]
                break
        if not pick or not pick.memory:
            return
        try:
            msgs = pick.memory.get_messages(user_id=user_id) or []
            tail = msgs[-20:]
            text = "\n".join([getattr(m, "content", str(m)) for m in tail])[:4000]
            prompt = (
                f"Summarize the recent learning conversation (framework: {framework}). "
                "Return JSON: {\"summary\":\"<150 words>\", \"key_concepts\":[\"k1\",\"k2\",\"k3\"]}.\n"
                f"Conversation:\n{text}\n"
            )
            if hasattr(self.llm, "ainvoke"):
                resp = await self.llm.ainvoke(prompt)
            else:
                resp = self.llm.invoke(prompt)
            body = getattr(resp, "content", str(resp))
            try:
                import json
                data = json.loads(body)
                summary = data.get("summary") or body
            except Exception:
                summary = body
            # store into LT memory (it will vectorize via graph transformer)
            self.lt_memory.save_short_term_memory(self.llm, summary, user_id=user_id, agent_type="summarizer")
        except Exception as e:
            if self.is_logging:
                logger.warning(f"Summarize/store failed: {e}")
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return list(self.agents.keys())
    
    def is_agent_available(self, agent_name: str) -> bool:
        """Check if an agent is available."""
        return agent_name in self.agents

    def has_history(self, user_id: str) -> bool:
        """Return True if any agent has stored memory for this user."""
        try:
            for name in ["socratic_instructor", "instructor", "code_assistant", "practice"]:
                agent = self.agents.get(name)
                if agent and getattr(agent, "memory", None):
                    try:
                        data = agent.memory.load_memory(load_type='list', user_id=user_id)
                        if isinstance(data, list) and len(data) > 0:
                            return True
                    except Exception:
                        continue
        except Exception:
            pass
        return False

    def get_last_exchange(self, user_id: str) -> Dict[str, str]:
        """Get the last user/assistant exchange for quick display in the UI."""
        # Prefer session-tracked last exchange (most reliable for current run)
        last_exchange = self.current_session.get("last_exchange")
        if isinstance(last_exchange, dict) and last_exchange.get("assistant"):
            return last_exchange
        # Fallback: attempt to reconstruct from message history, if available
        try:
            for name in ["socratic_instructor", "instructor", "code_assistant", "practice"]:
                agent = self.agents.get(name)
                if agent and getattr(agent, "memory", None):
                    msgs = agent.memory.get_messages(user_id=user_id) or []
                    if len(msgs) >= 2:
                        # Look backwards for last HumanMessage and AIMessage contents
                        user_text = None
                        ai_text = None
                        for m in reversed(msgs):
                            content = getattr(m, "content", None)
                            typ = m.__class__.__name__.lower()
                            if ai_text is None and ("ai" in typ or "assistant" in typ):
                                ai_text = content
                            elif user_text is None and ("human" in typ or "user" in typ):
                                user_text = content
                            if ai_text is not None and user_text is not None:
                                break
                        if ai_text or user_text:
                            return {
                                "user": user_text or "",
                                "assistant": ai_text or "",
                                "agent": name,
                            }
        except Exception:
            pass
        return {}

    def generate_history_aware_opener(self, user_id: str, framework: str, study_mode: bool) -> str:
        """Generate a short guided opener when history exists to lead the user forward."""
        last = self.get_last_exchange(user_id)
        # Pick a concise hint based on mode
        mode_hint = (
            "I’ll start with a couple of questions to guide you." if study_mode else "I can give a quick recap or show the next concept."
        )
        if last and (last.get("assistant") or last.get("user")):
            last_topic = (last.get("assistant") or last.get("user") or "").strip()
            last_topic = (last_topic[:120] + "…") if len(last_topic) > 120 else last_topic
            return (
                f"We have previous progress in {framework.title()}. Last time we discussed: \n• {last_topic}\n"
                f"What would you like next — continue from there or try a small hands-on step? {mode_hint}"
            )
        return (
            f"Welcome back to {framework.title()}! Would you like a quick recap, continue where you left off, or do a short practice? {mode_hint}"
        )