"""
Socratic Instructor Agent - Study Mode Implementation
Extends InstructorAgent with Socratic questioning methodology based on OpenAI Study Mode principles.

This agent implements the core Study Mode philosophy:
1. Ask questions instead of giving direct answers
2. Guide discovery through progressive questioning  
3. Encourage active participation and critical thinking
4. Provide hints when stuck, not solutions
5. Foster curiosity and deeper understanding
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from langchain_core.messages import HumanMessage
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

        study_mode = bool(learning_context and learning_context.get("study_mode"))
        if study_mode:
            # Produce concise Socratic response but still persist short-term memory like base class
            context = learning_context or {}
            # Save the user query
            try:
                if self.memory:
                    framework_id = context.get("framework")
                    self.memory.append_chat_message(self._user_id or user_id, role="user", content=query, framework=framework_id)
                    self.save_memory(query, user_id=self._user_id or user_id)
            except Exception:
                pass

            result = await self.socratic_response(query, context)

            # Save the assistant response content
            try:
                if self.memory:
                    framework_id = context.get("framework")
                    content = result.get("content", "") if isinstance(result, dict) else str(result)
                    if content:
                        self.memory.append_chat_message(self._user_id or user_id, role="assistant", content=content, framework=framework_id)
                        self.save_memory(content, user_id=self._user_id or user_id)
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
        
        # Create structured Socratic response
        response_content = self.format_socratic_response(
            socratic_questions, query, query_analysis, context
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

    # ──────────────────────────────────────────────────────────────
    # Learning Path generation via RAG (no hardcoded steps)
    # ──────────────────────────────────────────────────────────────
    async def _generate_learning_path_via_rag(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        framework = self._get_framework_from_context(context)
        level, goal = self._parse_quick_check_signals(query)
        rag_query = self._build_rag_query(framework=framework, level=level, goal=goal)

        # Retrieve reference snippets from local vector store
        results = await self._retrieve_docs_async(framework=framework, query=rag_query, top_k=6)
        if not results:
            # LLM-first fallback: let the LLM synthesize a plan from curriculum summary and user context
            curriculum_summary = (context or {}).get("curriculum_summary", "")
            level = (context or {}).get("user_level") or "beginner"
            goal_text = (self._parse_quick_check_signals(query)[1] or query)
            system = (
                "You are a concise learning coach. Create a short, actionable Learning Path (3-5 steps) "
                "for the given framework and learner goal. Use ONLY the provided curriculum summary and context. "
                "Avoid hallucinations. Keep steps crisp and practical."
            )
            user = (
                f"Framework: {framework}\n"
                f"Learner level: {level}\n"
                f"Goal: {goal_text}\n\n"
                f"Curriculum summary (bullet list):\n{curriculum_summary or '- (no summary)'}\n\n"
                "Format:\n**Recommended Learning Path**\n- Step 1: ...\n- Step 2: ...\n- Step 3: ...\n(+ optional Step 4-5)"
            )
            try:
                if hasattr(self.llm, "ainvoke"):
                    resp = await self.llm.ainvoke([HumanMessage(content=system), HumanMessage(content=user)])
                else:
                    resp = self.llm.invoke([HumanMessage(content=system), HumanMessage(content=user)])
                content = getattr(resp, "content", str(resp))
            except Exception:
                content = "**Recommended Learning Path**\n- Step 1: Core concepts\n- Step 2: Minimal example\n- Step 3: Tools\n- Step 4: Memory"

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
        system = (
            "You are a concise learning coach. Create a short, actionable Learning Path (3-5 steps) for the given framework and goal. "
            "Constrain yourself to ONLY use the provided reference snippets. Each step must be grounded by citations [n] that map to the sources list. "
            "Avoid hallucinations. Keep it crisp like a quick plan in bullet points."
        )
        user = (
            f"Framework: {framework}\n"
            f"Learner level: {level or 'beginner'}\n"
            f"Goal: {goal or query}\n\n"
            f"References:\n{context_block}\n\n"
            "Format strictly:\n"
            "**Recommended Learning Path**\n\n"
            "- Step 1: ... [1]\n"
            "- Step 2: ... [2]\n"
            "- Step 3: ... [3]\n"
            "(+ optional Step 4-5 if needed)\n\n"
            "Sources:\n"
            f"{citations_block}"
        )

        try:
            # Use chat-style inputs to encourage structured output
            if hasattr(self.llm, "ainvoke"):
                resp = await self.llm.ainvoke([HumanMessage(content=system), HumanMessage(content=user)])
            else:
                resp = self.llm.invoke([HumanMessage(content=system), HumanMessage(content=user)])
            content = getattr(resp, "content", str(resp))
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
        history_block = ("\nRecent history:\n" + history_context) if history_context else ""
        socratic_prompt = f"""
You are a Socratic tutor teaching {framework} to a {user_level} student.
Student asked: "{query}"
{history_block}

Task: Ask 2-3 short follow-up questions that build directly on the recent context (if any).
Constraints:
- Do NOT repeat generic onboarding questions.
- Keep each question single-sentence, concise.
- Match user's level: {user_level}.
- Focus type: {analysis.get('type', 'concept_exploration')}
- Output only questions as a numbered list.
"""
        
        try:
            # Try async invoke first, fallback to sync
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(socratic_prompt)
            else:
                response = self.llm.invoke(socratic_prompt)
            
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
            resp = await self.llm.ainvoke(prompt) if hasattr(self.llm, "ainvoke") else self.llm.invoke(prompt)
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

        prompt = f"""
You are a Socratic tutor. Framework: {framework}. User level: {user_level}. Language: en.
User message: "{query}"

Task: Ask 2-3 short onboarding questions for the stage: {onboarding_stage}.
{stage_directive}
Constraints:
- No explanations, no answers. Only questions.
- Single-sentence per question; concise, in English.
- If helpful, mention the framework by name once.
Output: numbered list of 2-3 questions only.
"""

        try:
            resp = await self.llm.ainvoke(prompt) if hasattr(self.llm, "ainvoke") else self.llm.invoke(prompt)
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
        context: Dict[str, Any]
    ) -> str:
        """
        Concise Socratic response: 2–3 questions, minimal framing, optional 1-line hint.
        """
        output_lines: List[str] = ["🤔 Let's think together:"]
        # Limit to top 3 concise questions
        for index, question in enumerate(questions[:3], start=1):
            output_lines.append(f"{index}. {question}")

        # One-line hint only if scaffolding is needed
        if analysis.get("requires_scaffolding", False):
            if analysis.get("type") == "problem_solving":
                output_lines.append("Hint: break the first step down and try it immediately.")
            elif analysis.get("type") == "concept_exploration":
                output_lines.append("Hint: start from what you already know about the concept.")
            else:
                output_lines.append("Hint: consider pros and cons of each direction.")

        # Hard cap to keep responses short
        content = "\n\n".join([output_lines[0], "\n".join(output_lines[1:])]).strip()
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