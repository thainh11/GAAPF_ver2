import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from langchain_core.messages import HumanMessage, SystemMessage
from .instructor import InstructorAgent
from ..tools.vector_store import VectorStore
from pathlib import Path

# Setup logging (avoid duplicate handlers)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocraticInstructorAgent(InstructorAgent):
    """
    Instructor agent with Socratic methodology support.
    
    Implements the 5 core principles from OpenAI Study Mode:
    1. Encouraging Active Participation
    2. Managing Cognitive Load  
    3. Developing Metacognition and Self-Reflection
    4. Fostering Curiosity
    5. Providing Actionable Feedback and Support
    """
    
    # Socratic question templates organized by learning objectives
    SOCRATIC_QUESTION_TEMPLATES = {
        "concept_exploration": [
            "What do you think {concept} means in the context of {framework}?",
            "How would you explain {concept} to someone new to programming?", 
            "What problems do you think {concept} is designed to solve?",
            "Can you think of a real-world analogy for {concept}?"
        ],
        "problem_solving": [
            "What's the first step you'd take to approach this problem?",
            "What information do we need before we can solve this?",
            "How does this relate to concepts you already know?",
            "What would happen if we broke this into smaller parts?"
        ],
        "code_understanding": [
            "What do you think this code is trying to accomplish?",
            "Which part of this code seems most important to you?",
            "What would happen if we changed this part?",
            "How does this code connect to the overall goal?"
        ],
        "deeper_thinking": [
            "Why do you think the framework designers made this choice?",
            "What are the trade-offs of this approach?",
            "How might this be different from other frameworks you know?",
            "What questions does this raise for you?"
        ],
        "application": [
            "How would you use this in a real project?",
            "What challenges might you face when implementing this?",
            "Can you think of a situation where this would be particularly useful?",
            "What would you need to learn next to master this concept?"
        ]
    }
    
    # Hint templates for when students get stuck
    HINT_TEMPLATES = [
        "💡 Think about what you already know about {topic}...",
        "💡 Consider breaking this down into smaller pieces...",
        "💡 What's the main goal we're trying to achieve here?",
        "💡 How does this connect to concepts you've learned before?",
        "💡 What would happen if you started with the simplest case?"
    ]
    
    def __init__(self, *args, **kwargs):
        """Initialize Socratic Instructor Agent"""
        super().__init__(*args, **kwargs)
        # Ensure correct agent identity and avoid inherited instructor flow
        self.agent_type = "socratic_instructor"
        self._agent_type = "socratic_instructor"
        # Remove compiled graph from InstructorAgent to prevent long explanatory flow
        if hasattr(self, "compiled_graph"):
            del self.compiled_graph
        # Clear flow so parent logic won't attempt to (re)compile
        self.flow = []
        self.socratic_mode = True
        self.conversation_history = []
        self.user_progress = {
            "concepts_explored": [],
            "questions_asked": 0,
            "discoveries_made": 0,
            "help_requests": 0,
            # Limit onboarding to a single round per session to avoid loops
            "onboarding_rounds": 0,
        }
        # Lightweight per-session caches
        self._intent_cache: Dict[str, Dict[str, Any]] = {}
        self._onboarding_cache: Dict[str, List[str]] = {}
        self._recent_q_cache: deque[str] = deque(maxlen=8)
        # Track flow to trigger Learning Path after onboarding quick check
        self._awaiting_learning_path: bool = False
        # Lazy cache VectorStore per framework
        self._framework_vectorstores: Dict[str, VectorStore] = {}
        
        if self.is_logging:
            logger.info("SocraticInstructorAgent initialized with Study Mode enabled")
    
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query with optional Socratic approach based on study_mode flag
        """
        study_mode = context.get("study_mode", False)
        
        if study_mode:
            return await self.socratic_response(query, context)
        else:
            # Fall back to direct instruction mode
            return await super().process(query, context)

    async def ainvoke(
        self,
        query: str,
        is_save_memory: bool = False,
        user_id: str = "unknown_user",
        learning_context: Dict = None,
        **kwargs,
    ) -> Any:
        """
        Override to enforce Socratic path in Study Mode and bypass instructor graph.
        """
        # Ensure correct agent identity for logs/memory
        self.agent_type = "socratic_instructor"
        self._agent_type = "socratic_instructor"

        # If any compiled graph exists from base class, remove it to avoid verbose flow
        if hasattr(self, "compiled_graph"):
            del self.compiled_graph

        # Ensure current user id preference mirrors base Agent behavior
        try:
            if learning_context and "user_profile" in learning_context:
                profile = learning_context.get("user_profile") or {}
                self._user_id = profile.get("user_id", user_id or "unknown_user")
            else:
                self._user_id = user_id or "unknown_user"
        except Exception:
            self._user_id = user_id or "unknown_user"

        study_mode = bool(learning_context and learning_context.get("study_mode"))
        if study_mode:
            # Produce concise Socratic response but still persist short-term memory like base class
            context = learning_context or {}
            # Save the user query
            try:
                if self.memory:
                    framework_id = context.get("framework")
                    uid = user_id or self._user_id or "unknown_user"
                    self.memory.append_chat_message(uid, role="user", content=query, framework=framework_id)
                    self.save_memory(query, user_id=uid)
            except Exception:
                pass

            result = await self.socratic_response(query, context)

            # Save the assistant response content
            try:
                if self.memory:
                    framework_id = context.get("framework")
                    content = result.get("content", "") if isinstance(result, dict) else str(result)
                    if content:
                        uid = user_id or self._user_id or "unknown_user"
                        self.memory.append_chat_message(uid, role="assistant", content=content, framework=framework_id)
                        self.save_memory(content, user_id=uid)
            except Exception:
                pass

            return result

        # Fallback to normal behavior in Direct Mode
        return await super().ainvoke(query, is_save_memory, user_id, learning_context, **kwargs)
    
    async def socratic_response(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Socratic response - questions instead of direct answers
        
        This is the core of Study Mode implementation following OpenAI's methodology
        """
        framework = context.get("framework", "programming")
        user_level = context.get("user_level", "beginner")
        
        # Track the interaction
        self.user_progress["questions_asked"] += 1
        self.conversation_history.append({"type": "user_query", "content": query})

        # --- Dynamic onboarding path (no hardcoded content) ---
        session = context.get("session", {}) or {}
        interaction_count = int(context.get("interaction_count", session.get("interaction_count", 0)) or 0)

        # 1) If we were waiting for user's reply after onboarding, synthesize Learning Path first
        if self._awaiting_learning_path:
            self._awaiting_learning_path = False
            return await self._generate_learning_path_via_rag(query=query, context=context)

        # 2) Classify intent and decide if we should run onboarding (limit to 1 round)
        intent = await self.classify_user_intent(query, context)
        stage = (context or {}).get("stage")
        self.user_progress.setdefault("onboarding_rounds", 0)
        allow_onboarding = (
            (stage != "build")
            and not self._should_skip_onboarding(query, context)
            and not self._awaiting_learning_path
            and (interaction_count <= 1 or (intent.get("type") == "onboarding" and interaction_count <= 2))
            and (self.user_progress.get("onboarding_rounds", 0) < 1)
        )

        if allow_onboarding:
            # Use staged onboarding aligned with conversation.json
            try:
                session = context.get("session", {}) or {}
                onboarding_stage = session.get("onboarding_stage", "concept_basics")
            except Exception:
                onboarding_stage = "concept_basics"

            questions = await self.generate_onboarding_questions(query, {**context, "onboarding_stage": onboarding_stage})
            # Format and return immediately
            response_content = self.format_socratic_response(
                questions=questions,
                original_query=query,
                analysis={"type": "onboarding", "requires_scaffolding": False, "onboarding_stage": onboarding_stage},
                context=context,
            )
            self.conversation_history.append({"type": "socratic_response", "content": response_content})
            # Next user reply should trigger Learning Path synthesis or stage advancement
            self._awaiting_learning_path = True
            self.user_progress["onboarding_rounds"] = self.user_progress.get("onboarding_rounds", 0) + 1
            return {
                "content": response_content,
                "type": "socratic_question",
                "mode": "study_mode",
                "suggestions": [
                    "Share your background",
                    "Tell your goal",
                    "Pick one use case",
                ],
                "progress": self.user_progress.copy(),
            }

        # 3) Otherwise proceed with contextual Socratic questions
        # Analyze query type and learning intent
        query_analysis = self.analyze_query_for_socratic_response(query, framework)
        
        # Generate contextual Socratic questions
        socratic_questions = await self.generate_contextual_questions(
            query, query_analysis, framework, user_level, context=context
        )

        # De-duplicate against recent question cache
        filtered: List[str] = []
        seen = {q.strip().lower() for q in self._recent_q_cache}
        for q in socratic_questions:
            qn = q.strip().lower()
            if qn and qn not in seen:
                filtered.append(q)
                self._recent_q_cache.append(q)
                seen.add(qn)
        if filtered:
            socratic_questions = filtered
        
        # Build next action and transition per template
        try:
            next_action, transition = self._build_next_action_and_transition(query_analysis, framework)
        except Exception:
            next_action, transition = None, None

        # Create structured Socratic response
        response_content = self.format_socratic_response(
            socratic_questions, query, query_analysis, context,
            concise_answer=None, next_action=next_action, transition=transition
        )
        
        # Track progress
        self.conversation_history.append({"type": "socratic_response", "content": response_content})
        
        return {
            "content": response_content,
            "type": "socratic_question",
            "mode": "study_mode",
            "suggestions": self.generate_follow_up_suggestions(query_analysis),
            "progress": self.user_progress.copy()
        }

    def _build_next_action_and_transition(self, analysis: Dict[str, Any], framework: str) -> Tuple[str, str]:
        t = analysis.get("type")
        if t == "problem_solving":
            return (
                f"Describe the very first step to build with {framework} in one sentence, then do it.",
                "We’ll add the second step or a simple tool right after."
            )
        if t == "code_understanding":
            return (
                "Point to the single most important line and explain why in one sentence.",
                "Then we’ll map each line to the overall goal."
            )
        if t == "concept_exploration":
            return (
                "Summarize the concept in your own words (1–2 sentences).",
                "Next, we can compare it to a related concept or try a tiny example."
            )
        return (
            "Pick one small goal (chatbot, RAG, or a single tool) to focus on first.",
            "Then we’ll choose tools vs memory accordingly."
        )

    async def _generate_learning_path_via_rag(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        framework = self._get_framework_from_context(context)
        level, goal = self._parse_quick_check_signals(query)
        rag_query = self._build_rag_query(framework=framework, level=level, goal=goal)

        # Retrieve reference snippets from local vector store
        results = await self._retrieve_docs_async(framework=framework, query=rag_query, top_k=6)
        if not results:
            # LLM-first fallback: let the LLM synthesize a plan from curriculum summary and user context, with history in system
            curriculum_summary = (context or {}).get("curriculum_summary", "")
            level = (context or {}).get("user_level") or "beginner"
            goal_text = (self._parse_quick_check_signals(query)[1] or query)
            history_context = self._build_history_context(context)
            history_block = ("\n\nContext (Recent history):\n" + history_context) if history_context else ""
            system = (
                "You are a concise learning coach. Create a short, actionable Learning Path (3-5 steps) "
                "for the given framework and learner goal. Use ONLY the provided curriculum summary and context. "
                "Avoid hallucinations. Keep steps crisp and practical. Do not include sources.\n\n"
                f"Framework: {framework}\n"
                f"Learner level: {level}\n"
                f"Goal: {goal_text}\n\n"
                f"Curriculum summary (bullet list):\n{curriculum_summary or '- (no summary)'}\n"
                f"{history_block}\n"
                "Format:\n**Recommended Learning Path**\n- Step 1: ...\n- Step 2: ...\n- Step 3: ...\n(+ optional Step 4-5)"
            )
            try:
                if hasattr(self.llm, "ainvoke"):
                    resp = await self.llm.ainvoke([SystemMessage(content=system), HumanMessage(content=query)])
                else:
                    resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
                content = getattr(resp, "content", str(resp))
            except Exception:
                content = "**Recommended Learning Path**\n- Step 1: Core concepts\n- Step 2: Minimal example\n- Step 3: Tools\n- Step 4: Memory"

            # sanitize possible Sources blocks
            try:
                content = "\n\n".join([p for p in content.split("\n\n") if not p.strip().lower().startswith("sources:")]).strip()
            except Exception:
                pass

            return {
                "content": content,
                "type": "learning_path",
                "mode": "study_mode",
                "suggestions": [
                    "Start with Step 1",
                    "Show a minimal example",
                    "Adjust the plan for my goal",
                ],
                "progress": self.user_progress.copy(),
            }

        citations_block, context_block = self._format_retrieved_context(results)
        history_context = self._build_history_context(context)
        history_block = ("\n\nContext (Recent history):\n" + history_context) if history_context else ""
        curriculum_summary = (context or {}).get("curriculum_summary", "")
        system = (
            "You are a concise learning coach. Create a short, actionable Learning Path (3-5 steps) for the given framework and goal. "
            "Constrain yourself to ONLY use the provided reference snippets and curriculum summary. Do not include sources.\n\n"
            f"Framework: {framework}\n"
            f"Learner level: {level or 'beginner'}\n"
            f"Goal: {goal or query}\n\n"
            f"Curriculum summary:\n{curriculum_summary or '- (no summary)'}\n"
            f"{history_block}\n\n"
            f"References:\n{context_block}\n\n"
            "Format strictly:\n"
            "**Recommended Learning Path**\n\n"
            "- Step 1: ...\n"
            "- Step 2: ...\n"
            "- Step 3: ...\n"
            "(+ optional Step 4-5 if needed)"
        )

        try:
            # Use chat-style inputs to encourage structured output
            if hasattr(self.llm, "ainvoke"):
                resp = await self.llm.ainvoke([SystemMessage(content=system), HumanMessage(content=query)])
            else:
                resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
            content = getattr(resp, "content", str(resp))
            # sanitize possible Sources blocks
            try:
                content = "\n\n".join([p for p in content.split("\n\n") if not p.strip().lower().startswith("sources:")]).strip()
            except Exception:
                pass
        except Exception:
            content = (
                "Here is a concise Learning Path to get you started. If you want, I can also fetch official docs for citations.\n\n"
                "- Step 1: Core concepts\n- Step 2: Minimal agent/graph\n- Step 3: Add tools\n- Step 4: Memory & customization"
            )

        return {
            "content": content,
            "type": "learning_path",
            "mode": "study_mode",
            "suggestions": [
                "Start with Step 1",
                "Show a minimal example",
                "Adjust the plan for my goal"
            ],
            "progress": self.user_progress.copy(),
        }

    def _get_framework_from_context(self, context: Dict[str, Any]) -> str:
        fw = (context or {}).get("framework") or "langchain"
        return str(fw).strip().lower()

    def _parse_quick_check_signals(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        t = (text or "").lower()
        level = None
        if any(k in t for k in ["beginner", "basic", "new"]):
            level = "beginner"
        elif any(k in t for k in ["intermediate", "some", "familiar"]):
            level = "intermediate"
        elif any(k in t for k in ["advanced", "expert"]):
            level = "advanced"
        # crude goal detection: if user mentions intent/target, keep whole answer as goal context
        goal = None
        for key in ["goal", "i want", "i wanna", "i would like", "build", "agent", "rag", "tool", "memory"]:
            if key in t:
                goal = text.strip()
                break
        return level, goal

    def _build_rag_query(self, framework: str, level: Optional[str], goal: Optional[str]) -> str:
        base = f"{framework} learning path for {level or 'beginner'}"
        if goal:
            base += f" focused on {goal}"
        return base

    async def _retrieve_docs_async(self, framework: str, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        try:
            vs = self._framework_vectorstores.get(framework)
            if vs is None:
                # Per-framework collection keeps retrieval on-topic (match ingestion path)
                base = Path("data/frameworks/vectordb")
                vs_dir_path = base / framework
                vs = VectorStore(persistent_dir=str(vs_dir_path), collection_name=f"framework_docs_{framework}")
                self._framework_vectorstores[framework] = vs
            results = vs.similarity_search(query=query, k=top_k)
            return results or []
        except Exception:
            return []

    def _format_retrieved_context(self, results: List[Dict[str, Any]]) -> Tuple[str, str]:
        citations: List[str] = []
        blocks: List[str] = []
        for idx, r in enumerate(results, start=1):
            meta = r.get("metadata", {}) or {}
            url = meta.get("url") or meta.get("source") or meta.get("link") or ""
            title = meta.get("title") or meta.get("section") or f"Ref {idx}"
            citations.append(f"[{idx}] {title} - {url}")
            text = (r.get("text") or "").strip()
            if text:
                blocks.append(f"[Ref {idx}]\n{text}")
        citations_block = "\n".join(citations)
        context_block = "\n\n".join(blocks)
        return citations_block, context_block
    
    def analyze_query_for_socratic_response(self, query: str, framework: str) -> Dict[str, Any]:
        """
        Analyze user query to determine the best Socratic approach
        """
        query_lower = query.lower()

        # LLM-driven path will already detect onboarding; also keep a simple lexical fallback
        onboarding_phrases = [
            "i want to learn", "bắt đầu", "new to", "beginner", "from scratch",
            "how to start", "get started", "học từ đầu",
        ]
        if any(p in query_lower for p in onboarding_phrases):
            return {
                "type": "onboarding",
                "complexity": "basic",
                "key_concepts": [],
                "framework": framework,
                "requires_scaffolding": False,
            }
        
        # Determine query type
        if any(word in query_lower for word in ["what is", "define", "explain", "tell me about"]):
            query_type = "concept_exploration"
            complexity = "basic"
        elif any(word in query_lower for word in ["how to", "implement", "create", "build", "make"]):
            query_type = "problem_solving" 
            complexity = "intermediate"
        elif any(word in query_lower for word in ["code", "function", "class", "method", "example"]):
            query_type = "code_understanding"
            complexity = "intermediate"
        elif any(word in query_lower for word in ["why", "difference", "compare", "better", "choose"]):
            query_type = "deeper_thinking"
            complexity = "advanced"
        else:
            query_type = "application"
            complexity = "intermediate"
        
        # Extract key concepts
        key_concepts = self.extract_key_concepts(query, framework)
        
        return {
            "type": query_type,
            "complexity": complexity,
            "key_concepts": key_concepts,
            "framework": framework,
            "requires_scaffolding": complexity in ["intermediate", "advanced"]
        }
    
    def extract_key_concepts(self, query: str, framework: str) -> List[str]:
        """Extract key technical concepts from the query"""
        # Framework-specific concept keywords
        framework_concepts = {
            "langchain": ["chain", "agent", "tool", "memory", "prompt", "llm", "retriever", "vectorstore"],
            "langgraph": ["graph", "node", "edge", "state", "workflow", "checkpoint", "stream"],
            "crewai": ["crew", "agent", "task", "role", "goal", "backstory", "tool"],
            "autogen": ["conversation", "agent", "chat", "group", "assistant", "user"]
        }
        
        concepts = []
        query_lower = query.lower()
        
        # Get framework-specific concepts
        if framework in framework_concepts:
            for concept in framework_concepts[framework]:
                if concept in query_lower:
                    concepts.append(concept)
        
        # Add general programming concepts
        general_concepts = ["function", "class", "method", "variable", "api", "database", "file"]
        for concept in general_concepts:
            if concept in query_lower:
                concepts.append(concept)
        
        return concepts[:3]  # Limit to top 3 concepts
    
    async def generate_contextual_questions(
        self,
        query: str,
        analysis: Dict[str, Any],
        framework: str,
        user_level: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate contextual Socratic questions using both templates and LLM
        """
        # Short-circuit onboarding to avoid template/LLM mixing here
        if analysis.get("type") == "onboarding":
            # Should not normally reach here because onboarding handled earlier,
            # but keep a safe fallback that generates onboarding questions.
            return await self.generate_onboarding_questions(query, {"framework": framework, "user_level": user_level})
        query_type = analysis["type"]
        # complexity is not used below; avoid unused var warning
        key_concepts = analysis["key_concepts"]
        
        # Get base templates
        templates = self.SOCRATIC_QUESTION_TEMPLATES.get(query_type, 
                    self.SOCRATIC_QUESTION_TEMPLATES["concept_exploration"])
        
        # Use LLM to generate contextual questions
        try:
            history_context = self._build_history_context(context)
            llm_questions = await self.generate_llm_questions(query, analysis, framework, user_level, history_context=history_context)
            # Combine template and LLM questions
            all_questions = llm_questions + [
                template.format(concept=concept, framework=framework) 
                for template in templates[:2] 
                for concept in (key_concepts[:1] if key_concepts else [query.split()[0]])
            ]
        except Exception as e:
            if self.is_logging:
                logger.warning(f"LLM question generation failed: {e}, using templates")
            # Fallback to templates only
            all_questions = [
                template.format(concept=concept, framework=framework)
                for template in templates[:3]
                for concept in (key_concepts[:1] if key_concepts else [query.split()[0]])
            ]
        
        # Select best questions (2-3 questions max)
        return self.select_best_questions(all_questions, analysis, user_level)
    
    async def generate_llm_questions(
        self,
        query: str,
        analysis: Dict[str, Any],
        framework: str,
        user_level: str,
        history_context: Optional[str] = None,
    ) -> List[str]:
        """
        Use LLM to generate contextual Socratic questions
        """
        history_block = ("\n\nContext (Recent history):\n" + history_context) if history_context else ""
        system_prompt = f"""
You are a Socratic tutor for {framework}, adapting to a {user_level} learner.

GOALS
- Encourage active participation through strategic questions.
- Build from prior knowledge toward application in small steps.

STYLE
- Friendly, concise, supportive; use emojis sparingly (🔹, ✅).
- Default to English unless the user specifies otherwise.
- Do not reveal chain-of-thought; no explanations, no answers.

TASK
Ask 2–3 single-sentence follow-up questions that build directly on the student's message and the recent context.

CONSTRAINTS
- No greetings or prefaces.
- Output only questions as a numbered list (1., 2., 3.).
- Avoid generic onboarding if we've already started; be specific to the query.
- Match level: {user_level}.
- Focus type: {analysis.get('type', 'concept_exploration')}.
{history_block}
"""

        user_prompt = query
        
        try:
            # Try async invoke first, fallback to sync
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            else:
                response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            
            # Handle both string responses and objects with .content attribute
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            questions = self.extract_questions_from_llm_response(response_text)
            return questions
        except Exception as e:
            if self.is_logging:
                logger.error(f"LLM question generation error: {e}")
            return []

    def _build_history_context(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Build a compact recent history context from transcript if available."""
        try:
            if not context:
                return None
            framework = context.get("framework")
            session = context.get("session", {}) or {}
            user_id = session.get("user_id", "default")
            if not getattr(self, "memory", None):
                return None
            recent = self.memory.get_recent_chat(user_id=user_id, framework=framework, k=6) or []
            if not recent:
                # Fallback to last_exchange if transcript empty
                try:
                    last = (session or {}).get("last_exchange") or {}
                    if last and (last.get("user") or last.get("assistant")):
                        user = (last.get("user") or "").strip()[:150]
                        assistant = (last.get("assistant") or "").strip()[:150]
                        lines = []
                        if user:
                            lines.append(f"- User: {user}")
                        if assistant:
                            lines.append(f"- AI: {assistant}")
                        return "\n".join(lines) if lines else None
                except Exception:
                    pass
                return None
            lines: List[str] = []
            for item in recent[-6:]:
                role = item.get("role", "?")
                content = (item.get("content", "") or "").strip()[:150]
                prefix = "User" if role == "user" else "AI"
                lines.append(f"- {prefix}: {content}")
            return "\n".join(lines)
        except Exception:
            return None

    def _should_skip_onboarding(self, query: str, context: Dict[str, Any]) -> bool:
        """Return True if we should bypass onboarding (we already have context and the query is vague/continuation)."""
        try:
            framework = context.get("framework")
            session = context.get("session", {}) or {}
            user_id = session.get("user_id", "default")
            has_history = False
            if getattr(self, "memory", None):
                recent = self.memory.get_recent_chat(user_id=user_id, framework=framework, k=4) or []
                has_history = len(recent) >= 2
            q = (query or "").lower()
            vague_markers = [
                "what do you know", "what we learn", "right now", "continue", "next",
                "tiếp tục", "tiếp theo", "bây giờ",
            ]
            is_vague = any(m in q for m in vague_markers)
            return has_history and is_vague
        except Exception:
            return False

    async def classify_user_intent(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify the user's intent dynamically via LLM with small token budget.
        Caches by last user message to avoid repeated calls per turn.
        """
        cache_key = (context.get("session", {}).get("user_id", "default"), query.strip())
        if cache_key in self._intent_cache:
            return self._intent_cache[cache_key]

        framework = context.get("framework", "programming")
        prompt = (
            "Classify the user's message for a learning assistant.\n"
            f"Framework: {framework}.\n"
            "Types: onboarding, concept, problem, code, deep.\n"
            "Return ONLY compact JSON: {\"type\": \"<type>\", \"confidence\": 0.0-1.0}.\n"
            f"Message: \"{query}\""
        )

        try:
            if hasattr(self.llm, "ainvoke"):
                resp = await self.llm.ainvoke(prompt)
            else:
                resp = self.llm.invoke(prompt)
            text = getattr(resp, "content", str(resp))
            result = {"type": "onboarding", "confidence": 0.5}
            # Try parse JSON
            try:
                parsed = json.loads(text.strip())
                if isinstance(parsed, dict) and "type" in parsed:
                    result = {"type": str(parsed.get("type", "onboarding")), "confidence": float(parsed.get("confidence", 0.5))}
            except Exception:
                # heuristic fallback based on interaction_count
                session = context.get("session", {}) or {}
                ic = int(context.get("interaction_count", session.get("interaction_count", 0)) or 0)
                if ic <= 2:
                    result = {"type": "onboarding", "confidence": 0.6}
            self._intent_cache[cache_key] = result
            return result
        except Exception:
            # conservative fallback
            return {"type": "onboarding", "confidence": 0.5}

    async def generate_onboarding_questions(self, query: str, context: Dict[str, Any]) -> List[str]:
        """
        LLM-based onboarding questions with optional mix from framework config.
        Keeps total to 2–3 questions and caches by last query.
        """
        cache_key = (context.get("session", {}).get("user_id", "default"), query.strip())
        if cache_key in self._onboarding_cache:
            return self._onboarding_cache[cache_key]

        framework = context.get("framework", "programming")
        user_level = context.get("user_level", "beginner")
        onboarding_stage = context.get("onboarding_stage", "concept_basics")

        # Try to extract a seed question from framework config if available
        seed_questions: List[str] = []
        framework_config = context.get("framework_config") or {}
        try:
            modules = framework_config.get("modules", {})
            # pick first module/objective as directional question
            if modules:
                first_key = next(iter(modules))
                module = modules.get(first_key, {})
                objectives = module.get("learning_objectives") or module.get("objectives") or []
                if objectives:
                    seed_questions.append(f"Which objective would you like to focus on first in module '{first_key}'? (e.g., {objectives[0]})")
        except Exception:
            pass

        # English-only prompting and stage-specific guidance
        if onboarding_stage == "concept_basics":
            stage_directive = "Focus on gauging background, prior experience, and target outcomes."
        elif onboarding_stage == "simple_tool":
            stage_directive = "Guide the user toward trying one simple tool or function relevant to the framework."
        else:  # build_agent
            stage_directive = "Confirm readiness to build a small agent and clarify minimal requirements."

        # Build history context into system prompt (user prompt stays raw)
        try:
            history_context = self._build_history_context({"framework": framework, **(context or {})})
            history_block = ("\n\nContext (Recent history):\n" + history_context) if history_context else ""
        except Exception:
            history_block = ""

        system_prompt = f"""
You are a Socratic tutor. Framework: {framework}. User level: {user_level}.

STYLE
- Friendly, concise, supportive; use emojis sparingly (🔹, ✅).
- Default to English unless the user specifies otherwise.
- Do not reveal chain-of-thought; no explanations, no answers.

TASK
Ask 2–3 short onboarding questions for the stage: {onboarding_stage}.
{stage_directive}

CONSTRAINTS
- Questions only; single sentence each; numbered list (1., 2., 3.).
- Be concrete and relevant to the user's message.
- Mention the framework by name at most once if helpful.
{history_block}
"""

        user_prompt = query

        try:
            resp = await self.llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]) if hasattr(self.llm, "ainvoke") else self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            text = getattr(resp, "content", str(resp))
            questions = self.extract_questions_from_llm_response(text)
            # Mix 1 seed question from config if present
            if seed_questions:
                mixed: List[str] = [seed_questions[0]] + questions
                questions = mixed[:3]
            # Ensure 2–3 questions
            questions = [q for q in questions if q][:3]
            if len(questions) < 2:
                # add a generic calibration if LLM returned too little
                questions.append(f"What do you want to use {framework.title()} for? (chatbot, RAG, agent, tools)")
            self._onboarding_cache[cache_key] = questions
            return questions
        except Exception:
            # Fallback minimal dynamic question set
            qs = [
                f"Have you used {framework.title()} or similar libraries before?",
                f"What do you want to use {framework.title()} for?",
            ]
            self._onboarding_cache[cache_key] = qs
            return qs
    
    def extract_questions_from_llm_response(self, response: str) -> List[str]:
        """Extract clean questions from LLM response"""
        lines = response.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            if line and '?' in line:
                line = line.lstrip('1234567890.-•* ')
                if len(line) > 10 and line.count('?') == 1:  # Valid question
                    questions.append(line)
        
        return questions[:3]  # Max 3 questions
    
    def select_best_questions(
        self, 
        questions: List[str], 
        analysis: Dict[str, Any], 
        user_level: str
    ) -> List[str]:
        """
        Select the best 2-3 questions based on learning objectives
        """
        # Remove duplicates and filter
        unique_questions = []
        seen = set()
        
        for q in questions:
            q_clean = q.lower().strip()
            if q_clean not in seen and len(q) > 15:
                unique_questions.append(q)
                seen.add(q_clean)
        
        # Prioritize questions based on complexity and user level
        if user_level == "beginner":
            # Prefer simpler, foundational questions
            return unique_questions[:2]
        else:
            # Can handle more complex questions
            return unique_questions[:3]
    
    def format_socratic_response(
        self,
        questions: List[str],
        original_query: str,
        analysis: Dict[str, Any],
        context: Dict[str, Any],
        concise_answer: Optional[str] = None,
        next_action: Optional[str] = None,
        transition: Optional[str] = None,
    ) -> str:
        """
        Concise Socratic response: Positive opening → 2–3 questions → optional 1-line hint →
        Actionable next step → Transition bridge.
        """
        output_lines: List[str] = ["🎉 Great question! Let's think together:"]

        # Optional brief direct answer (≤3 sentences) before questions
        if concise_answer:
            brief = concise_answer.strip()
            if brief:
                output_lines.append(f"Brief answer: {brief}")

        # Limit to top 3 concise questions
        top_qs = [q for q in questions[:3] if q]
        if top_qs:
            output_lines.append("\n".join([f"{i+1}. {q}" for i, q in enumerate(top_qs)]))

        # One-line hint only if scaffolding is needed
        if analysis.get("requires_scaffolding", False):
            if analysis.get("type") == "problem_solving":
                output_lines.append("💡 Hint: Try to isolate the very first step and attempt it immediately.")
            elif analysis.get("type") == "concept_exploration":
                output_lines.append("💡 Hint: Start from what you already know about the concept.")
            else:
                output_lines.append("💡 Hint: Consider pros and cons of each direction.")

        # Actionable next step (immediate application)
        if next_action:
            output_lines.append(f"🧪 Action: {next_action}")

        # Transition bridge to next concept
        if transition:
            output_lines.append(f"🔄 Next: {transition}")

        # If judge addback guidance is present in context, lightly reinforce in 1 line
        try:
            addback = context.get("soc_addback") or []
            if addback:
                tip = str(addback[-1])[:120]
                if tip:
                    output_lines.append(f"Coach tip: {tip}")
        except Exception:
            pass

        # Hard cap to keep responses short
        content = "\n\n".join([output_lines[0], "\n\n".join(output_lines[1:])]).strip()
        return content[:600]
    
    def generate_follow_up_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate contextual follow-up suggestions"""
        suggestions = [
            "Think step by step",
            "Consider what you already know",
            "Ask for a hint if you get stuck"
        ]
        
        # Add specific suggestions based on query type
        if analysis["type"] == "concept_exploration":
            suggestions.append("Try explaining it in your own words")
        elif analysis["type"] == "problem_solving":
            suggestions.append("Break the problem into smaller parts")
        elif analysis["type"] == "code_understanding":
            suggestions.append("Trace through the code line by line")
        
        return suggestions
    
    async def provide_hint(self, context: Dict[str, Any], specific_topic: str = None) -> str:
        """
        Provide a helpful hint when student is stuck (not a full answer)
        """
        self.user_progress["help_requests"] += 1
        
        if specific_topic:
            hint = f"💡 **Hint about {specific_topic}**: "
        else:
            hint = "💡 **Hint**: "
        
        # Select appropriate hint template
        import random
        hint_template = random.choice(self.HINT_TEMPLATES)
        hint += hint_template.format(topic=specific_topic or "this concept")
        
        hint += "\n\n🎯 Try answering one of the questions above, even if you're not completely sure. "
        hint += "Learning happens through exploration!"
        
        return hint
    
    async def celebrate_discovery(self, user_response: str, context: Dict[str, Any]) -> str:
        """
        Celebrate when student makes a discovery or shows understanding
        """
        self.user_progress["discoveries_made"] += 1
        
        celebrations = [
            "🎉 **Excellent thinking!** You're really getting it!",
            "✨ **Great insight!** That shows deep understanding!",
            "🌟 **Wonderful!** You've made an important connection!",
            "🎯 **Perfect!** That's exactly the kind of thinking we want!"
        ]
        
        import random
        celebration = random.choice(celebrations)
        
        # Add follow-up question to deepen understanding
        follow_up = "\n\nNow that you understand this, what do you think the next step would be?"
        
        return celebration + follow_up
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current learning progress summary"""
        return {
            "questions_explored": self.user_progress["questions_asked"],
            "discoveries_made": self.user_progress["discoveries_made"], 
            "help_requests": self.user_progress["help_requests"],
            "engagement_level": "high" if self.user_progress["questions_asked"] > 3 else "moderate",
            "learning_style": "socratic_discovery"
        }