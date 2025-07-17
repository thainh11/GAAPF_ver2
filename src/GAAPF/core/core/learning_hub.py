"""
Learning Hub Core for GAAPF Architecture

This module provides the LearningHub class that serves as the central
coordination system for the GAAPF architecture, managing agent
constellations, user profiles, and learning sessions.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import time
import asyncio

from langchain_core.language_models.base import BaseLanguageModel
from .constellation import Constellation, create_constellation_for_context
from .temporal_state import TemporalState
from .knowledge_graph import KnowledgeGraph
from .analytics_engine import AnalyticsEngine
from .session_manager import SessionManager
from ..config.user_profiles import UserProfile
from ..config.framework_configs import FrameworkConfig
from ..memory.long_term_memory import LongTermMemory
from .framework_onboarding import FrameworkOnboarding

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningHub:
    """
    Central coordination system for the GAAPF architecture.
    
    The LearningHub is responsible for:
    1. Managing agent constellations
    2. Tracking user profiles and learning sessions
    3. Coordinating the learning experience
    4. Connecting UI layer with agent constellation system
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        user_profiles_path: Optional[Union[Path, str]] = Path('user_profiles'),
        frameworks_path: Optional[Union[Path, str]] = Path('frameworks'),
        memory_path: Optional[Union[Path, str]] = Path('memory'),
        memory: Optional[LongTermMemory] = None,
        is_logging: bool = False,
        *args, **kwargs
    ):
        """
        Initialize the LearningHub.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model to use for all agents
        user_profiles_path : Path or str, optional
            Path to user profiles directory
        frameworks_path : Path or str, optional
            Path to framework configurations directory
        memory_path : Path or str, optional
            Path to memory files directory
        memory : LongTermMemory, optional
            Pre-initialized long-term memory instance
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        # Initialize paths
        self.user_profiles_path = Path(user_profiles_path) if isinstance(user_profiles_path, str) else user_profiles_path
        self.frameworks_path = Path(frameworks_path) if isinstance(frameworks_path, str) else frameworks_path
        self.memory_path = Path(memory_path) if isinstance(memory_path, str) else memory_path
        
        # Create directories if they don't exist
        self.user_profiles_path.mkdir(parents=True, exist_ok=True)
        self.frameworks_path.mkdir(parents=True, exist_ok=True)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        self.llm = llm
        self.is_logging = is_logging
        self.memory = memory
        
        # Initialize managers and components
        self.user_profile_manager = UserProfile(
            user_profiles_path=self.user_profiles_path,
            is_logging=is_logging
        )
        
        self.framework_config_manager = FrameworkConfig(
            frameworks_path=self.frameworks_path,
            is_logging=is_logging
        )
        
        self.temporal_state = TemporalState(
            is_logging=is_logging
        )
        
        self.knowledge_graph = KnowledgeGraph(
            is_logging=is_logging
        )
        
        self.analytics_engine = AnalyticsEngine(
            is_logging=is_logging
        )
        
        self.framework_onboarding = FrameworkOnboarding(
            memory=self.memory,
            knowledge_graph=self.knowledge_graph,
            cache_dir=self.frameworks_path.parent / "data/framework_cache",
            is_logging=self.is_logging
        )
        
        # Initialize session manager for persistence
        self.session_manager = SessionManager(
            is_logging=is_logging
        )
        
        # Active sessions and constellations
        self.active_sessions = {}
        self.active_constellations = {}
        
        # Load any existing active sessions on startup
        self._load_existing_sessions()
        
        if self.is_logging:
            logger.info("LearningHub initialized")
    
    async def start_session(
        self,
        user_id: str,
        framework_id: str,
        module_id: Optional[str] = None
    ) -> Dict:
        """
        Start or resume a learning session.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        framework_id : str
            Identifier for the framework
        module_id : str, optional
            Identifier for the module to start with
            
        Returns:
        -------
        Dict
            Session information
        """
        # Check if session already exists for this user+framework
        session_id = self.session_manager.get_session_id(user_id, framework_id)
        if session_id in self.active_sessions:
            return self.resume_session(user_id, framework_id)
        
        # Generate deterministic session ID
        session_id = self.session_manager.get_session_id(user_id, framework_id)
        
        # Get user profile
        user_profile = self.user_profile_manager.get_profile(user_id)
        
        # Create default profile if user doesn't exist
        if user_profile is None:
            # Create a new profile using the proper method
            user_profile = self.user_profile_manager.create_profile(
                user_id=user_id,
                name=f"User {user_id}",
                experience_level="beginner",
                learning_style={
                    "preferred_mode": "balanced",
                    "pace": "moderate",
                    "interaction_style": "guided",
                    "detail_level": "balanced"
                },
                goals=[],
                frameworks=[framework_id]
            )
            
            if self.is_logging:
                logger.info(f"Created new profile for user {user_id}")
        
        # Get framework configuration
        framework_config = self.framework_config_manager.get_framework(framework_id)
        
        # Check if framework configuration exists
        if framework_config is None:
            if self.is_logging:
                logger.info(f"Framework {framework_id} not found locally. Initializing with onboarding process.")
            
            curriculum = await self.framework_onboarding.initialize_framework(
                framework_name=framework_id,
                user_id=user_id,
                user_config=user_profile,
                initialization_mode="quick",
                is_background_collection=True
            )

            new_config = {
                "name": framework_id,
                "description": curriculum.get("description", ""),
                "modules": curriculum.get("modules", {}),
            }
            self.framework_config_manager.save_framework(framework_id, new_config)
            framework_config = new_config
        
        # Determine starting module if not provided
        if not module_id:
            # Check if user has completed modules in this framework
            completed_modules = []
            if "progress" in user_profile and framework_id in user_profile["progress"]:
                framework_progress = user_profile["progress"][framework_id]
                # Get modules with completion percentage > 80%
                for mod_id, mod_progress in framework_progress.items():
                    if isinstance(mod_progress, dict) and mod_progress.get("completion_percentage", 0) > 80:
                        completed_modules.append(mod_id)
            
            framework_modules = framework_config.get("modules", {})
            
            # Find first uncompleted module
            for module_id in framework_modules:
                if module_id not in completed_modules:
                    break
            
            # If all modules completed, use the first one
            if not module_id and framework_modules:
                module_id = list(framework_modules.keys())[0]
        
        # Create learning context
        learning_context = {
            "user_id": user_id,
            "session_id": session_id,
            "framework_id": framework_id,
            "framework_config": framework_config,
            "current_module": module_id,
            "user_profile": user_profile,
            "learning_stage": "exploration",
            "current_activity": "introduction",
            "session_start_time": time.time(),
            "interaction_count": 0,
            "messages": [],  # Initialize empty conversation history
            "created_at": time.time()
        }
        
        # Create constellation for this context
        constellation = create_constellation_for_context(
            llm=self.llm,
            learning_context=learning_context,
            user_id=user_id,
            memory_path=self.memory_path,
            is_logging=self.is_logging
        )
        
        # Store session and constellation
        self.active_sessions[session_id] = learning_context
        self.active_constellations[session_id] = constellation
        
        # Save session to persistent storage
        self.session_manager.save_session_state(learning_context)
        
        # Update user profile with session info
        self.user_profile_manager.update_profile(
            user_id,
            {
                "last_session": session_id,
                "last_session_time": int(time.time())
            }
        )
        
        # Log session start
        if self.is_logging:
            logger.info(f"Started new session {session_id} for user {user_id} with framework {framework_id}")
        
        # Return session info
        return {
            "session_id": session_id,
            "user_id": user_id,
            "framework_id": framework_id,
            "module_id": module_id,
            "constellation_type": constellation.constellation_type,
            "is_resumed": False
        }
    
    def resume_session(self, user_id: str, framework_id: str) -> Dict:
        """
        Resume an existing learning session.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        framework_id : str
            Identifier for the framework
            
        Returns:
        -------
        Dict
            Session information
        """
        # Load session state from persistent storage
        session_data = self.session_manager.load_session_state(user_id, framework_id)
        
        if not session_data:
            error_msg = f"No existing session found for user {user_id} and framework {framework_id}"
            if self.is_logging:
                logger.error(error_msg)
            return {"error": error_msg, "session_id": None}
        
        session_id = session_data["session_id"]
        
        # Get updated user profile and framework config
        user_profile = self.user_profile_manager.get_profile(user_id)
        
        # Create default profile if user doesn't exist
        if user_profile is None:
            # Create a new profile using the proper method
            user_profile = self.user_profile_manager.create_profile(
                user_id=user_id,
                name=f"User {user_id}",
                experience_level="beginner",
                learning_style={
                    "preferred_mode": "balanced",
                    "pace": "moderate",
                    "interaction_style": "guided",
                    "detail_level": "balanced"
                },
                goals=[],
                frameworks=[framework_id]
            )
            
            if self.is_logging:
                logger.info(f"Created new profile for user {user_id}")
        
        framework_config = self.framework_config_manager.get_framework(framework_id)
        
        # Update session data with current information
        session_data["user_profile"] = user_profile
        session_data["framework_config"] = framework_config
        session_data["session_resume_time"] = time.time()
        
        # Load conversation history
        conversation_history = self.session_manager.load_conversation_history(session_id)
        session_data["messages"] = conversation_history
        
        # Create constellation for this context
        constellation = create_constellation_for_context(
            llm=self.llm,
            learning_context=session_data,
            user_id=user_id,
            memory_path=self.memory_path,
            is_logging=self.is_logging
        )
        
        # Store session and constellation in active memory
        self.active_sessions[session_id] = session_data
        self.active_constellations[session_id] = constellation
        
        # Update user profile with session info
        session_id = session_data["session_id"]
        self.user_profile_manager.update_profile(
            user_id,
            {
                "last_session": session_id,
                "last_session_time": int(time.time())
            }
        )
        
        # Log session resume
        if self.is_logging:
            logger.info(f"Resumed session {session_id} for user {user_id} with framework {framework_id}")
            logger.info(f"Loaded {len(conversation_history)} previous messages")
        
        # Return session info
        return {
            "session_id": session_id,
            "user_id": user_id,
            "framework_id": framework_id,
            "module_id": session_data.get("current_module"),
            "constellation_type": constellation.constellation_type,
            "is_resumed": True,
            "message_count": len(conversation_history)
        }
    
    def _load_existing_sessions(self):
        """Load any existing sessions from persistent storage on startup."""
        try:
            # This could be enhanced to load all user sessions if needed
            # For now, sessions will be loaded on-demand when users request them
            if self.is_logging:
                logger.info("Session loading capability initialized")
        except Exception as e:
            logger.error(f"Error loading existing sessions: {e}")
    
    def save_session_state(self, session_id: str):
        """Save current session state to persistent storage."""
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            self.session_manager.save_session_state(session_data)
            
            # Also save conversation history if messages exist
            if "messages" in session_data and session_data["messages"]:
                self.session_manager.save_conversation_history(session_id, session_data["messages"])
    
    def process_interaction(
        self,
        session_id: str,
        interaction_data: Dict
    ) -> Dict:
        """
        Process a learning interaction with enhanced context and guidance.
        ENHANCED: Better context propagation and agent coordination with comprehensive logging
        
        Parameters:
        ----------
        session_id : str
            Identifier for the session
        interaction_data : Dict
            Data about the interaction
            
        Returns:
        -------
        Dict
            Response data with enhanced learning guidance
        """
        start_time = time.time()
        
        # Check if session exists
        if session_id not in self.active_sessions:
            error_msg = f"Session {session_id} not found"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg}
        
        if self.is_logging:
            logger.info(f"🔄 Processing interaction for session {session_id}")
            logger.info(f"📊 Interaction type: {interaction_data.get('type', 'unknown')}")
            logger.info(f"📝 Query length: {len(interaction_data.get('query', ''))}")
        
        # Get learning context and constellation
        learning_context = self.active_sessions[session_id]
        constellation = self.active_constellations[session_id]
        
        if self.is_logging:
            logger.info(f"👤 User: {learning_context.get('user_id', 'unknown')}")
            logger.info(f"📚 Framework: {learning_context.get('framework_id', 'unknown')}")
            logger.info(f"📖 Module: {learning_context.get('current_module', 'unknown')}")
        
        # Update interaction count
        learning_context["interaction_count"] += 1
        
        if self.is_logging:
            logger.info(f"📊 Interaction count: {learning_context['interaction_count']}")
        
        # ENHANCED: Better query analysis for agent routing
        query = interaction_data.get("query", "").lower()
        practice_keywords = [
            "practice", "exercise", "coding", "code", "implement", 
            "hands-on", "try", "example", "demo", "tutorial", 
            "workshop", "show me how", "help me implement",
            "let me try", "i want to practice", "give me", "write code"
        ]
        
        # Update interaction type based on query analysis
        if any(keyword in query for keyword in practice_keywords):
            interaction_data["type"] = "practice"
            learning_context["current_activity"] = "practice"
            learning_context["learning_stage"] = "practice"
            
            if self.is_logging:
                logger.info("🛠️ Practice interaction detected - updating learning stage")
        
        # Enhance learning context with progress tracking
        context_start = time.time()
        enhanced_context = self._enhance_learning_context_for_interaction(learning_context, interaction_data)
        
        if self.is_logging:
            context_time = time.time() - context_start
            logger.info(f"🧠 Learning context enhanced in {context_time:.3f}s")
        
        # ENHANCED: Add framework-specific context (framework-agnostic)
        framework_id = enhanced_context.get("framework_id")
        if framework_id:
            framework_config = enhanced_context.get("framework_config", {})
            current_module = enhanced_context.get("current_module", "")
            
            # Add module context for ANY framework
            if current_module and current_module in framework_config.get("modules", {}):
                module_data = framework_config["modules"][current_module]
                enhanced_context["module_concepts"] = module_data.get("concepts", [])
                enhanced_context["module_complexity"] = module_data.get("complexity", "basic")
                enhanced_context["module_duration"] = module_data.get("estimated_duration", 30)
                enhanced_context["module_prerequisites"] = module_data.get("prerequisites", [])
                enhanced_context["module_title"] = module_data.get("title", current_module.title())
                enhanced_context["module_description"] = module_data.get("description", "")
            
            # Add framework metadata
            enhanced_context["framework_name"] = framework_config.get("name", framework_id.title())
            enhanced_context["framework_version"] = framework_config.get("version", "")
            enhanced_context["framework_description"] = framework_config.get("description", "")
        
        # Process with temporal state
        temporal_start = time.time()
        temporal_context = self.temporal_state.process_interaction(
            learning_context=enhanced_context,
            interaction_data=interaction_data
        )
        
        if self.is_logging:
            temporal_time = time.time() - temporal_start
            logger.info(f"⏰ Temporal state processed in {temporal_time:.3f}s")
        
        # Update learning context with temporal insights
        enhanced_context.update(temporal_context)
        
        # Process with constellation using enhanced context
        constellation_start = time.time()
        response = constellation.process_interaction(
            interaction_data=interaction_data,
            learning_context=enhanced_context
        )
        
        if self.is_logging:
            constellation_time = time.time() - constellation_start
            response_agent = response.get('agent_type', 'unknown')
            logger.info(f"🌟 Constellation processed with agent '{response_agent}' in {constellation_time:.3f}s")
        
        # Update knowledge graph
        kg_start = time.time()
        self.knowledge_graph.update_from_interaction(
            user_id=enhanced_context["user_id"],
            learning_context=enhanced_context,
            interaction_data=interaction_data,
            response=response
        )
        
        if self.is_logging:
            kg_time = time.time() - kg_start
            logger.info(f"🧭 Knowledge graph updated in {kg_time:.3f}s")
        
        # Update analytics
        analytics_start = time.time()
        self.analytics_engine.track_interaction(
            user_id=enhanced_context["user_id"],
            session_id=session_id,
            learning_context=enhanced_context,
            interaction_data=interaction_data,
            response=response
        )
        
        # FIX: Update user profile with current session info after each interaction
        # This ensures the "Last Session" status is properly updated
        self.user_profile_manager.update_profile(
            enhanced_context["user_id"],
            {
                "last_session": session_id,
                "last_session_time": int(time.time())
            }
        )
        
        if self.is_logging:
            analytics_time = time.time() - analytics_start
            logger.info(f"📈 Analytics updated in {analytics_time:.3f}s")
        
        # Check if constellation should be updated
        if self._should_update_constellation(enhanced_context, response):
            # Create new constellation
            new_constellation = create_constellation_for_context(
                llm=self.llm,
                learning_context=enhanced_context,
                user_id=enhanced_context["user_id"],
                memory_path=self.memory_path,
                is_logging=self.is_logging
            )
            
            # Update active constellation
            self.active_constellations[session_id] = new_constellation
            
            # Add constellation update info to response
            response["constellation_updated"] = True
            response["new_constellation_type"] = new_constellation.constellation_type
        
        # Store conversation messages for persistence
        user_message = {
            "role": "user",
            "content": interaction_data.get("query", ""),
            "timestamp": time.time()
        }
        
        ai_message = {
            "role": "assistant", 
            "content": response.get("content", ""),
            "agent_type": response.get("agent_type", "unknown"),
            "timestamp": time.time()
        }
        
        # Add messages to session history
        if "messages" not in enhanced_context:
            enhanced_context["messages"] = []
        
        enhanced_context["messages"].append(user_message)
        enhanced_context["messages"].append(ai_message)
        
        # Update session with enhanced context
        self.active_sessions[session_id] = enhanced_context
        
        # Save session state and conversation history periodically
        save_start = time.time()
        if enhanced_context["interaction_count"] % 5 == 0:  # Save every 5 interactions
            self.save_session_state(session_id)
            
            if self.is_logging:
                logger.info(f"💾 Periodic session state saved (interaction #{enhanced_context['interaction_count']})")

        # Also save to conversation files immediately for each interaction
        try:
            self.session_manager.save_conversation_history(session_id, enhanced_context["messages"])
            if self.is_logging:
                save_time = time.time() - save_start
                logger.info(f"💬 Conversation saved after interaction {enhanced_context['interaction_count']} in {save_time:.3f}s")
        except Exception as e:
            logger.error(f"❌ Failed to save conversation: {e}")
        
        # Format response with enhanced learning guidance
        format_start = time.time()
        formatted_response = self._format_enhanced_response(response, enhanced_context, session_id)
        
        if self.is_logging:
            format_time = time.time() - format_start
            total_time = time.time() - start_time
            logger.info(f"📋 Response formatted in {format_time:.3f}s")
            logger.info(f"✅ Total interaction processing completed in {total_time:.3f}s")
        
        return formatted_response
    
    def _enhance_learning_context_for_interaction(self, learning_context: Dict, interaction_data: Dict) -> Dict:
        """
        Enhance learning context specifically for the current interaction.
        
        Parameters:
        ----------
        learning_context : Dict
            Original learning context
        interaction_data : Dict
            Current interaction data
            
        Returns:
        -------
        Dict
            Enhanced learning context
        """
        enhanced_context = learning_context.copy()
        
        # Add interaction metadata
        enhanced_context["current_query"] = interaction_data.get("query", "")
        enhanced_context["interaction_timestamp"] = time.time()
        
        # Enhance with curriculum progress analysis
        framework_config = learning_context.get("framework_config", {})
        current_module = learning_context.get("current_module", "")
        user_profile = learning_context.get("user_profile", {})
        
        # Calculate learning progress metrics
        if framework_config:
            modules = framework_config.get("modules", {})
            completed_modules = user_profile.get("completed_modules", [])
            
            enhanced_context["progress_metrics"] = {
                "total_modules": len(modules),
                "completed_modules": len(completed_modules),
                "completion_percentage": (len(completed_modules) / len(modules)) * 100 if modules else 0,
                "current_module_progress": self._estimate_module_progress(learning_context)
            }
        
        # Add conversation analysis
        messages = learning_context.get("messages", [])
        if messages:
            enhanced_context["conversation_analysis"] = {
                "total_messages": len(messages),
                "recent_interaction_topics": self._analyze_recent_topics(messages),
                "conversation_depth": self._assess_conversation_depth(messages),
                "learning_momentum": self._calculate_learning_momentum(messages)
            }
        
        # Add learning stage insights
        interaction_count = learning_context.get("interaction_count", 0)
        enhanced_context["learning_insights"] = {
            "engagement_pattern": self._analyze_engagement_pattern(interaction_count),
            "suggested_transition": self._suggest_learning_stage_transition(learning_context),
            "readiness_for_next": self._assess_readiness_for_progression(learning_context)
        }
        
        return enhanced_context
    
    def _estimate_module_progress(self, learning_context: Dict) -> float:
        """Estimate progress within the current module."""
        interaction_count = learning_context.get("interaction_count", 0)
        learning_stage = learning_context.get("learning_stage", "exploration")
        
        # Simple heuristic based on interactions and stage
        base_progress = min(interaction_count / 10.0, 0.8)  # Max 80% from interactions alone
        
        stage_bonuses = {
            "exploration": 0.0,
            "concept": 0.1,
            "practice": 0.2,
            "assessment": 0.3
        }
        
        return min(base_progress + stage_bonuses.get(learning_stage, 0.0), 1.0)
    
    def _analyze_recent_topics(self, messages: List[Dict]) -> List[str]:
        """Analyze recent conversation topics."""
        topics = []
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        
        topic_keywords = {
            "concepts": ["concept", "theory", "principle", "idea"],
            "practice": ["practice", "exercise", "code", "example", "implementation"],
            "questions": ["how", "what", "why", "when", "where"],
            "frameworks": ["langchain", "llm", "agent", "chain", "prompt", "memory"]
        }
        
        for message in recent_messages:
            content = message.get("content", "").lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in content for keyword in keywords):
                    if topic not in topics:
                        topics.append(topic)
        
        return topics
    
    def _assess_conversation_depth(self, messages: List[Dict]) -> str:
        """Assess the depth of the conversation."""
        if len(messages) < 4:
            return "surface"
        elif len(messages) < 10:
            return "moderate"
        else:
            return "deep"
    
    def _calculate_learning_momentum(self, messages: List[Dict]) -> str:
        """Calculate learning momentum based on message patterns."""
        if len(messages) < 2:
            return "starting"
        
        recent_count = len([msg for msg in messages[-10:] if msg.get("role") == "user"])
        
        if recent_count >= 5:
            return "high"
        elif recent_count >= 3:
            return "moderate"
        else:
            return "low"
    
    def _analyze_engagement_pattern(self, interaction_count: int) -> str:
        """Analyze user engagement pattern."""
        if interaction_count >= 12:
            return "highly_engaged"
        elif interaction_count >= 6:
            return "moderately_engaged"
        elif interaction_count >= 3:
            return "getting_engaged"
        else:
            return "just_starting"
    
    def _suggest_learning_stage_transition(self, learning_context: Dict) -> Optional[str]:
        """Suggest if learning stage should transition."""
        current_stage = learning_context.get("learning_stage", "exploration")
        interaction_count = learning_context.get("interaction_count", 0)
        
        stage_transitions = {
            "exploration": ("concept", 3),
            "concept": ("practice", 6),
            "practice": ("assessment", 10)
        }
        
        if current_stage in stage_transitions:
            next_stage, required_interactions = stage_transitions[current_stage]
            if interaction_count >= required_interactions:
                return next_stage
        
        return None
    
    def _assess_readiness_for_progression(self, learning_context: Dict) -> bool:
        """Assess if user is ready for module progression."""
        interaction_count = learning_context.get("interaction_count", 0)
        learning_stage = learning_context.get("learning_stage", "exploration")
        
        # Simple readiness assessment
        return (interaction_count >= 8 and learning_stage in ["practice", "assessment"])
    
    def _format_enhanced_response(self, response: Dict, enhanced_context: Dict, session_id: str) -> Dict:
        """Format response with enhanced learning guidance information."""
        
        # Base response structure
        formatted_response = {
            "primary_response": {
                "content": response.get("content", "No response available"),
                "agent_type": response.get("agent_type", "unknown"),
                "constellation_type": response.get("constellation_type", self.active_constellations[session_id].constellation_type)
            },
            "updated_context": {},
            "constellation_updated": response.get("constellation_updated", False),
            "curriculum_guided": response.get("curriculum_guided", False)
        }
        
        # Generate enhanced learning guidance based on context
        learning_guidance = self._generate_contextual_learning_guidance(enhanced_context, response)
        if learning_guidance:
            formatted_response["learning_guidance"] = learning_guidance
        else:
            formatted_response["learning_guidance"] = {}
            
        # Extract concepts for current module if available
        if enhanced_context.get("current_module") and enhanced_context.get("framework_config"):
            framework_config = enhanced_context["framework_config"]
            current_module = enhanced_context["current_module"]
            modules = framework_config.get("modules", {})
            
            if current_module in modules:
                module_data = modules[current_module]
                concepts = module_data.get("concepts", [])
                if concepts:
                    formatted_response["learning_guidance"]["current_concepts"] = concepts[:3]  # Limit to first 3
        
        # Add existing guidance from response
        if response.get("guidance"):
            formatted_response["learning_guidance"].update(response["guidance"])
        
        if response.get("learning_recommendations"):
            formatted_response["learning_recommendations"] = response["learning_recommendations"]
        
        if response.get("suggested_activities"):
            formatted_response["suggested_activities"] = response["suggested_activities"]
        
        # Add curriculum metadata
        if response.get("curriculum_metadata"):
            formatted_response["curriculum_metadata"] = response["curriculum_metadata"]
        
        # Add progress information
        progress_metrics = enhanced_context.get("progress_metrics", {})
        if progress_metrics:
            formatted_response["progress_info"] = progress_metrics
        
        # Add conversation insights
        conversation_analysis = enhanced_context.get("conversation_analysis", {})
        if conversation_analysis:
            formatted_response["conversation_insights"] = conversation_analysis
        
        # Add learning insights
        learning_insights = enhanced_context.get("learning_insights", {})
        if learning_insights:
            formatted_response["learning_insights"] = learning_insights
        
        # Add constellation update info
        if response.get("new_constellation_type"):
            formatted_response["new_constellation_type"] = response["new_constellation_type"]
        
        return formatted_response
    
    def _generate_contextual_learning_guidance(self, enhanced_context: Dict, response: Dict) -> Dict:
        """Generate enhanced contextual learning guidance based on session state and user behavior."""
        guidance = {}
        
        # Extract context information
        learning_stage = enhanced_context.get("learning_stage", "exploration")
        interaction_count = enhanced_context.get("interaction_count", 0)
        current_module = enhanced_context.get("current_module", "")
        user_profile = enhanced_context.get("user_profile", {})
        user_level = user_profile.get("experience_level", "beginner")
        conversation_analysis = enhanced_context.get("conversation_analysis", {})
        learning_insights = enhanced_context.get("learning_insights", {})
        
        # Analyze conversation patterns for personalized guidance
        conversation_depth = conversation_analysis.get("conversation_depth", "surface")
        learning_momentum = conversation_analysis.get("learning_momentum", "starting")
        recent_topics = conversation_analysis.get("recent_interaction_topics", [])
        engagement_pattern = learning_insights.get("engagement_pattern", "just_starting")
        
        # Generate adaptive next steps based on multiple factors
        next_steps = self._generate_adaptive_next_steps(
            learning_stage, interaction_count, user_level, 
            conversation_depth, learning_momentum, engagement_pattern
        )
        
        # Generate personalized learning tips
        learning_tips = self._generate_personalized_tips(
            user_level, learning_stage, conversation_depth, recent_topics
        )
        
        # Add module-specific guidance with enhanced context
        module_guidance = self._generate_module_guidance(
            current_module, enhanced_context, interaction_count
        )
        
        # Generate progress encouragement with personalization
        progress_note = self._generate_progress_encouragement(
            interaction_count, learning_stage, engagement_pattern, user_level
        )
        
        # Determine optimal learning path recommendations
        learning_path = self._suggest_learning_path(
            learning_stage, interaction_count, user_level, conversation_analysis
        )
        
        # Add readiness assessment for stage transitions
        readiness_assessment = self._assess_learning_readiness(
            enhanced_context, conversation_analysis
        )
        
        guidance.update({
            "next_steps": next_steps,
            "learning_tips": learning_tips,
            "progress_note": progress_note,
            "learning_stage": learning_stage,
            "session_depth": conversation_depth,
            "learning_momentum": learning_momentum,
            "engagement_level": engagement_pattern,
            "module_guidance": module_guidance,
            "learning_path": learning_path,
            "readiness_assessment": readiness_assessment,
            "personalization": {
                "user_level": user_level,
                "interaction_count": interaction_count,
                "recent_focus": recent_topics[:3] if recent_topics else []
            }
        })
        
        return guidance
    
    def _generate_adaptive_next_steps(self, learning_stage: str, interaction_count: int, 
                                     user_level: str, conversation_depth: str, 
                                     learning_momentum: str, engagement_pattern: str) -> List[str]:
        """Generate adaptive next steps based on multiple learning factors."""
        next_steps = []
        
        # Base steps by learning stage
        if learning_stage == "exploration":
            if user_level == "beginner":
                next_steps.extend([
                    "Ask about fundamental concepts and terminology",
                    "Request simple examples to build understanding",
                    "Explore basic features step-by-step"
                ])
            else:
                next_steps.extend([
                    "Dive into advanced concepts and architecture",
                    "Compare with frameworks you already know",
                    "Explore unique features and capabilities"
                ])
        
        elif learning_stage == "concept":
            if conversation_depth == "deep":
                next_steps.extend([
                    "Apply concepts in practical scenarios",
                    "Explore edge cases and advanced patterns",
                    "Connect concepts to real-world applications"
                ])
            else:
                next_steps.extend([
                    "Ask for detailed explanations of key concepts",
                    "Request examples that demonstrate the concepts",
                    "Clarify any confusing aspects"
                ])
        
        elif learning_stage == "practice":
            if engagement_pattern == "highly_engaged":
                next_steps.extend([
                    "Take on challenging implementation projects",
                    "Optimize and refactor existing code",
                    "Explore advanced patterns and best practices"
                ])
            else:
                next_steps.extend([
                    "Start with guided coding exercises",
                    "Practice with step-by-step tutorials",
                    "Build confidence with smaller tasks"
                ])
        
        # Adjust based on momentum
        if learning_momentum == "high" and interaction_count > 10:
            next_steps.append("Consider moving to more advanced topics")
        elif learning_momentum == "low":
            next_steps.append("Take time to review and consolidate your understanding")
        
        return next_steps[:4]  # Limit to 4 most relevant steps
    
    def _generate_personalized_tips(self, user_level: str, learning_stage: str, 
                                   conversation_depth: str, recent_topics: List[str]) -> List[str]:
        """Generate personalized learning tips based on user context."""
        tips = []
        
        # Level-specific tips
        if user_level == "beginner":
            tips.extend([
                "Take your time to understand each concept thoroughly",
                "Don't hesitate to ask for simpler explanations",
                "Practice with small examples before tackling larger projects"
            ])
        elif user_level == "intermediate":
            tips.extend([
                "Connect new concepts to your existing knowledge",
                "Focus on understanding the 'why' behind patterns",
                "Experiment with variations of examples"
            ])
        else:  # advanced
            tips.extend([
                "Consider architectural implications and trade-offs",
                "Explore performance optimization opportunities",
                "Think about scalability and maintainability"
            ])
        
        # Stage-specific tips
        if learning_stage == "exploration" and conversation_depth == "surface":
            tips.append("Ask follow-up questions to deepen your understanding")
        elif learning_stage == "practice":
            tips.append("Learn from mistakes - they're valuable learning opportunities")
        
        # Topic-specific tips
        if recent_topics:
            tips.append(f"Consider how {recent_topics[0]} relates to other concepts you've learned")
        
        return tips[:3]  # Limit to 3 most relevant tips
    
    def _generate_module_guidance(self, current_module: str, enhanced_context: Dict, 
                                 interaction_count: int) -> Dict:
        """Generate enhanced module-specific guidance."""
        guidance = {}
        
        if not current_module or current_module == "introduction":
            return guidance
        
        module_info = enhanced_context.get("current_module_info", {})
        if module_info:
            concepts = module_info.get("concepts", [])
            objectives = module_info.get("objectives", [])
            
            if concepts:
                guidance["key_concepts"] = concepts[:3]
                guidance["focus_suggestion"] = f"Focus on mastering: {', '.join(concepts[:2])}"
            
            if objectives:
                guidance["learning_objectives"] = objectives[:2]
            
            # Progress within module
            module_progress = enhanced_context.get("progress_metrics", {}).get("current_module_progress", 0)
            if module_progress > 0:
                guidance["module_progress"] = f"{module_progress:.0%} complete"
                
                if module_progress < 0.3:
                    guidance["stage_suggestion"] = "Continue exploring the fundamentals"
                elif module_progress < 0.7:
                    guidance["stage_suggestion"] = "Ready for hands-on practice"
                else:
                    guidance["stage_suggestion"] = "Consider moving to the next module"
        
        return guidance
    
    def _generate_progress_encouragement(self, interaction_count: int, learning_stage: str, 
                                       engagement_pattern: str, user_level: str) -> str:
        """Generate personalized progress encouragement."""
        if interaction_count == 0:
            return f"Welcome! As a {user_level} learner, you're about to embark on an exciting journey."
        
        # Base encouragement on engagement and progress
        if engagement_pattern == "highly_engaged":
            if interaction_count > 15:
                return "Outstanding dedication! Your deep engagement is building strong expertise."
            else:
                return "Excellent engagement! You're making rapid progress."
        elif engagement_pattern == "moderately_engaged":
            return "Great steady progress! You're building solid understanding."
        else:
            return "Good start! Every question and interaction builds your knowledge."
    
    def _suggest_learning_path(self, learning_stage: str, interaction_count: int, 
                              user_level: str, conversation_analysis: Dict) -> Dict:
        """Suggest optimal learning path based on current context."""
        path = {}
        
        # Determine next recommended stage
        if learning_stage == "exploration" and interaction_count >= 5:
            path["next_stage"] = "concept"
            path["transition_suggestion"] = "Ready to dive deeper into core concepts"
        elif learning_stage == "concept" and interaction_count >= 10:
            path["next_stage"] = "practice"
            path["transition_suggestion"] = "Time for hands-on implementation"
        elif learning_stage == "practice" and interaction_count >= 15:
            path["next_stage"] = "assessment"
            path["transition_suggestion"] = "Ready to test your knowledge"
        
        # Suggest learning activities
        conversation_depth = conversation_analysis.get("conversation_depth", "surface")
        if conversation_depth == "surface":
            path["recommended_activity"] = "Ask more detailed questions to deepen understanding"
        elif conversation_depth == "deep":
            path["recommended_activity"] = "Apply your knowledge in practical exercises"
        
        return path
    
    def _assess_learning_readiness(self, enhanced_context: Dict, conversation_analysis: Dict) -> Dict:
        """Assess readiness for various learning activities."""
        assessment = {}
        
        interaction_count = enhanced_context.get("interaction_count", 0)
        learning_stage = enhanced_context.get("learning_stage", "exploration")
        conversation_depth = conversation_analysis.get("conversation_depth", "surface")
        
        # Readiness for stage progression
        if learning_stage == "exploration" and interaction_count >= 3:
            assessment["concept_readiness"] = conversation_depth in ["medium", "deep"]
        elif learning_stage == "concept" and interaction_count >= 8:
            assessment["practice_readiness"] = conversation_depth == "deep"
        elif learning_stage == "practice" and interaction_count >= 12:
            assessment["assessment_readiness"] = True
        
        # Readiness for advanced topics
        if interaction_count > 10 and conversation_depth == "deep":
            assessment["advanced_topics_readiness"] = True
        
        return assessment
    
    def end_session(self, session_id: str) -> Dict:
        """
        End a learning session.
        
        Parameters:
        ----------
        session_id : str
            Identifier for the session
            
        Returns:
        -------
        Dict
            Session summary
        """
        # Check if session exists
        if session_id not in self.active_sessions:
            error_msg = f"Session {session_id} not found"
            if self.is_logging:
                logger.error(error_msg)
            return {"error": error_msg}
        
        # Get learning context
        learning_context = self.active_sessions[session_id]
        
        # Calculate session duration
        session_duration = time.time() - learning_context.get("session_start_time", time.time())
        
        # Generate session summary
        summary = {
            "session_id": session_id,
            "user_id": learning_context["user_id"],
            "framework_id": learning_context["framework_id"],
            "module_id": learning_context["current_module"],
            "duration_seconds": session_duration,
            "interaction_count": learning_context["interaction_count"],
            "metrics": {
                "duration_minutes": round(session_duration / 60, 1),
                "interactions_per_minute": round(learning_context["interaction_count"] / (session_duration / 60), 1) if session_duration > 0 else 0
            }
        }
        
        # Save final session state before cleanup
        self.save_session_state(session_id)
        
        # Update user profile with session data
        try:
            # Use update_profile instead of save_profile
            self.user_profile_manager.update_profile(
                learning_context["user_id"],
                {
                    "last_session": session_id,
                    "last_activity": int(time.time())
                }
            )
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
        
        # Clean up active session (but keep persistent data)
        del self.active_sessions[session_id]
        if session_id in self.active_constellations:
            del self.active_constellations[session_id]
        
        # Log session end
        if self.is_logging:
            logger.info(f"Ended session {session_id} for user {learning_context['user_id']}")
        
        return summary
    
    def get_session_info(self, session_id: str) -> Dict:
        """
        Get information about a session.
        
        Parameters:
        ----------
        session_id : str
            Identifier for the session
            
        Returns:
        -------
        Dict
            Session information
        """
        # Check if session exists
        if session_id not in self.active_sessions:
            error_msg = f"Session {session_id} not found"
            if self.is_logging:
                logger.error(error_msg)
            return {"error": error_msg}
        
        # Get learning context and constellation
        learning_context = self.active_sessions[session_id]
        constellation = self.active_constellations[session_id]
        
        # Get constellation info
        constellation_info = constellation.get_constellation_info()
        
        # Calculate session duration
        session_duration = time.time() - learning_context.get("session_start_time", time.time())
        
        # Create session info
        info = {
            "session_id": session_id,
            "user_id": learning_context["user_id"],
            "framework_id": learning_context["framework_id"],
            "module_id": learning_context["current_module"],
            "learning_stage": learning_context["learning_stage"],
            "current_activity": learning_context["current_activity"],
            "duration_seconds": session_duration,
            "interaction_count": learning_context["interaction_count"],
            "constellation": constellation_info,
            "messages": learning_context.get("messages", [])  # FIX: Include messages for CLI sync
        }
        
        return info
    
    def get_user_learning_summary(self, user_id: str, framework_id: Optional[str] = None) -> Dict:
        """
        Get a summary of a user's learning progress.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        framework_id : str, optional
            Identifier for a specific framework
            
        Returns:
        -------
        Dict
            Learning summary
        """
        # Get user profile
        user_profile = self.user_profile_manager.get_profile(user_id)
        
        # Get analytics for this user
        user_analytics = self.analytics_engine.get_user_analytics(user_id)
        
        # Create summary
        summary = {
            "user_id": user_id,
            "experience_level": user_profile.get("experience_level", "beginner"),
            "total_sessions": len(user_profile.get("session_history", [])),
            "completed_modules": user_profile.get("completed_modules", []),
            "analytics": user_analytics
        }
        
        # Filter by framework if specified
        if framework_id:
            # Get framework configuration
            framework_config = self.framework_config_manager.get_framework(framework_id)
            
            # Calculate progress percentage
            total_modules = len(framework_config.get("modules", {}))
            completed_modules = [m for m in user_profile.get("completed_modules", []) 
                                if m in framework_config.get("modules", {})]
            
            progress_pct = (len(completed_modules) / total_modules * 100) if total_modules > 0 else 0
            
            # Add framework-specific info
            summary["framework_id"] = framework_id
            summary["framework_name"] = framework_config.get("name", framework_id)
            summary["total_modules"] = total_modules
            summary["completed_modules_count"] = len(completed_modules)
            summary["progress_percentage"] = progress_pct
            
            # Filter analytics
            if "framework_analytics" in user_analytics:
                summary["analytics"] = user_analytics.get("framework_analytics", {}).get(framework_id, {})
        
        return summary
    
    def _should_update_constellation(self, learning_context: Dict, response: Dict) -> bool:
        """
        Determine if the constellation should be updated.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        response : Dict
            Response from current constellation
            
        Returns:
        -------
        bool
            True if constellation should be updated, False otherwise
        """
        # Check for explicit update request in response
        if response.get("update_constellation", False):
            return True
        
        # Check for learning stage change
        if learning_context.get("learning_stage_changed", False):
            return True
        
        # Check for activity change
        if learning_context.get("current_activity_changed", False):
            return True
        
        # Check for module change
        if learning_context.get("current_module_changed", False):
            return True
        
        # No update needed
        return False