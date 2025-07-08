"""
Constellation Manager for GAAPF Architecture

This module provides the Constellation class that manages the formation
and coordination of agent constellations based on learning context.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool

from ..agents import SpecializedAgent
from ..graph.constellation_graph import ConstellationGraph
from .constellation_types import get_constellation_type, get_recommended_constellation_types
from .learning_flow_orchestrator import LearningFlowOrchestrator
from .orchestration import llm_orchestrate_agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockAgent(SpecializedAgent):
    """Mock agent for fallback when agent initialization fails."""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        agent_type: str,
        tools: List[Union[str, BaseTool]] = [],
        memory_path: Optional[Path] = None,
        config: Dict = None,
        is_logging: bool = False,
        *args, **kwargs
    ):
        """Initialize a mock agent."""
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            config=config,
            agent_type=agent_type,
            description=f"A fallback agent for {agent_type}",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        if self.is_logging:
            logger.info(f"Initialized MockAgent for {agent_type}")
    
    def _generate_system_prompt(self, learning_context: Optional[Dict] = None) -> str:
        """Generate a system prompt for this agent."""
        
        user_name = learning_context.get("user_profile", {}).get("name", "learner")
        framework_name = learning_context.get("framework_config", {}).get("name", "the framework")
        
        return f"""You are a helpful AI assistant. The specialized agent ({self.agent_type}) is currently unavailable. Please apologize to the user and ask them if they would like to try another activity, such as reviewing a concept or trying a different exercise."""

class Constellation:
    """
    Manages the formation and coordination of agent constellations.
    
    The Constellation class is responsible for:
    1. Creating specialized agent teams based on learning context
    2. Managing agent handoffs and coordination
    3. Optimizing constellation composition for learning effectiveness
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        constellation_type: str = "learning",
        user_id: str = "default_user",
        memory_path: Optional[Path] = None,
        is_logging: bool = False,
        *args, **kwargs
    ):
        """
        Initialize a Constellation with a specific type and configuration.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model to use for all agents in this constellation
        constellation_type : str, optional
            Type of constellation to create
        user_id : str, optional
            Identifier for the user
        memory_path : Path, optional
            Base path for agent memory files
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        self.llm = llm
        self.constellation_type = constellation_type
        self.user_id = user_id
        self.memory_path = memory_path
        self.is_logging = is_logging
        
        # Get constellation configuration
        self.constellation_config = get_constellation_type(constellation_type)
        if not self.constellation_config:
            raise ValueError(f"Constellation type '{constellation_type}' not found")
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
        
        # Create constellation graph for agent coordination
        self.graph = ConstellationGraph(
            agents=self.agents,
            constellation_type=constellation_type,
            constellation_config=self.constellation_config,
            is_logging=is_logging
        )
        
        if self.is_logging:
            logger.info(f"Initialized {constellation_type} constellation with {len(self.agents)} agents")
    
    def process_interaction(self, interaction_data: Dict, learning_context: Dict) -> Dict:
        """
        Process a learning interaction through the constellation using LLM-driven orchestration.
        """
        from .learning_guidance import LearningGuidance
        guidance = LearningGuidance(is_logging=self.is_logging)
        enhanced_context = self._enhance_learning_context(learning_context, interaction_data)

        # --- LLM-DRIVEN ORCHESTRATION ---
        available_agents = list(self.agents.keys())
        orchestration = llm_orchestrate_agent(
            self.llm, interaction_data, enhanced_context, available_agents, self.is_logging
        )

        primary_agent_type = None
        if orchestration and orchestration.get("primary_agent") in self.agents:
            primary_agent_type = orchestration["primary_agent"]
            # Add the task to the learning context instead of modifying the query
            enhanced_context["agent_task"] = orchestration.get("task")
        else:
            # Fallback to the first primary agent if orchestration fails or suggests an unavailable agent
            if self.is_logging:
                logger.warning("LLM orchestration failed or suggested an invalid agent. Using fallback.")
            primary_agents = [
                agent_config["type"]
                for agent_config in self.constellation_config.get("agents", [])
                if agent_config.get("role") == "primary"
            ]
            primary_agent_type = primary_agents[0] if primary_agents else list(self.agents.keys())[0]


        response = self.graph.process(
            user_id=self.user_id,
            interaction_data=interaction_data,
            learning_context=enhanced_context,
            primary_agent=primary_agent_type
        )

        if not response.get("content"):
            response["content"] = self._generate_fallback_response(interaction_data, enhanced_context)
            response["agent_type"] = "system"
            response["constellation_type"] = self.constellation_type

        if response.get("agent_type") == "instructor":
            response = self._enhance_instructor_response(response, enhanced_context, guidance)

        enhanced_response = guidance.enhance_response_with_guidance(response, enhanced_context)
        enhanced_response = self._add_curriculum_tracking(enhanced_response, enhanced_context, interaction_data)
        return enhanced_response
    
    def _enhance_learning_context(self, learning_context: Dict, interaction_data: Dict) -> Dict:
        """
        Enhance learning context with additional curriculum and progress information.
        
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
        
        # Add conversation history context
        messages = learning_context.get("messages", [])
        if messages:
            enhanced_context["conversation_length"] = len(messages)
            enhanced_context["recent_topics"] = self._extract_recent_topics(messages)
        
        # Add curriculum progression context
        framework_config = learning_context.get("framework_config", {})
        current_module = learning_context.get("current_module", "")
        
        if framework_config and current_module:
            modules = framework_config.get("modules", {})
            if current_module in modules:
                module_info = modules[current_module]
                enhanced_context["current_module_info"] = module_info
                enhanced_context["current_concepts"] = module_info.get("concepts", [])
                enhanced_context["module_complexity"] = module_info.get("complexity", "basic")
                enhanced_context["estimated_duration"] = module_info.get("estimated_duration", 30)
        
        # Add interaction pattern analysis
        interaction_count = learning_context.get("interaction_count", 0)
        enhanced_context["interaction_patterns"] = {
            "total_interactions": interaction_count,
            "session_depth": "deep" if interaction_count > 10 else "medium" if interaction_count > 5 else "surface",
            "engagement_level": "high" if interaction_count > 8 else "medium" if interaction_count > 3 else "low"
        }
        
        return enhanced_context
    
    def _extract_recent_topics(self, messages: List[Dict]) -> List[str]:
        """
        Extract recent topics from conversation history.
        
        Parameters:
        ----------
        messages : List[Dict]
            Conversation messages
            
        Returns:
        -------
        List[str]
            List of recent topics
        """
        topics = []
        
        # Look at last 6 messages for recent topics
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        
        for message in recent_messages:
            content = message.get("content", "").lower()
            
            # Simple keyword extraction for topics
            topic_keywords = ["langchain", "llm", "chain", "prompt", "memory", "agent", "rag", "embedding", "vector"]
            
            for keyword in topic_keywords:
                if keyword in content and keyword not in topics:
                    topics.append(keyword)
        
        return topics
    
    def _add_curriculum_tracking(self, response: Dict, learning_context: Dict, interaction_data: Dict) -> Dict:
        """
        Add curriculum progression tracking to the response.
        
        Parameters:
        ----------
        response : Dict
            Original response
        learning_context : Dict
            Enhanced learning context
        interaction_data : Dict
            Current interaction data
            
        Returns:
        -------
        Dict
            Response with curriculum tracking
        """
        # Add curriculum metadata
        response["curriculum_metadata"] = {
            "current_module": learning_context.get("current_module", ""),
            "learning_stage": learning_context.get("learning_stage", "exploration"),
            "concepts_covered": learning_context.get("current_concepts", []),
            "interaction_count": learning_context.get("interaction_count", 0),
            "engagement_level": learning_context.get("interaction_patterns", {}).get("engagement_level", "low")
        }
        
        # Add progression suggestions
        framework_config = learning_context.get("framework_config", {})
        current_module = learning_context.get("current_module", "")
        
        if framework_config and current_module:
            modules = framework_config.get("modules", {})
            module_order = list(modules.keys())
            
            if current_module in module_order:
                current_index = module_order.index(current_module)
                
                # Suggest next module if appropriate
                if current_index < len(module_order) - 1:
                    next_module = module_order[current_index + 1]
                    response["curriculum_metadata"]["next_module"] = next_module
                    response["curriculum_metadata"]["next_module_title"] = modules[next_module].get("title", next_module)
                    response["curriculum_metadata"]["next_module_info"] = modules[next_module]
        
        return response
    
    def _enhance_instructor_response(self, response: Dict, learning_context: Dict, guidance) -> Dict:
        """
        Enhance instructor response with curriculum-aware guidance.
        
        Parameters:
        ----------
        response : Dict
            Original instructor response
        learning_context : Dict
            Enhanced learning context
        guidance : LearningGuidance
            Learning guidance instance
            
        Returns:
        -------
        Dict
            Enhanced instructor response
        """
        # Extract key learning context
        current_module = learning_context.get("current_module", "")
        framework_config = learning_context.get("framework_config", {})
        interaction_count = learning_context.get("interaction_count", 0)
        user_profile = learning_context.get("user_profile", {})
        
        # Get module information
        modules = framework_config.get("modules", {})
        module_info = modules.get(current_module, {}) if current_module else {}
        concepts = module_info.get("concepts", [])
        module_title = module_info.get("title", current_module)
        
        # Original content
        original_content = response.get("content", "")
        
        # Enhanced instructor content
        enhanced_content = original_content
        
        # Only enhance if we have sufficient context and the response isn't already comprehensive
        if concepts and current_module and len(original_content.split()) < 100:  # Enhance shorter responses
            
            # Add module context if not present
            if module_title.lower() not in original_content.lower():
                enhanced_content += f"\n\nWe're currently working through **{module_title}**, where you'll master key concepts like {', '.join(concepts[:3])}."
            
            # Add engagement-appropriate guidance
            experience_level = user_profile.get("experience_level", "beginner")
            
            if interaction_count <= 2:  # Initial interactions
                enhanced_content += f"\n\n**🌟 Getting Started:**\nSince you're at an {experience_level} level, I'll guide you step-by-step through these concepts. Feel free to ask questions at any time - that's how we learn best!"
                
            elif interaction_count <= 5:  # Early learning phase
                enhanced_content += f"\n\n**📚 Building Understanding:**\nYou're making great progress! As we continue, try to think about how these concepts might apply to real projects you're interested in."
                
            else:  # Active learning phase
                enhanced_content += f"\n\n**🚀 Deep Dive Ready:**\nYou've been actively engaged - excellent! Ready to explore more advanced aspects of these concepts or move toward hands-on practice?"
        
        # Update response content
        response["content"] = enhanced_content
        
        # Add instructor-specific metadata
        response["instructor_enhancements"] = {
            "module_context_added": module_title.lower() not in original_content.lower() if concepts else False,
            "engagement_guidance_added": True,
            "concepts_referenced": concepts[:3],
            "interaction_phase": "initial" if interaction_count <= 2 else "early" if interaction_count <= 5 else "active"
        }
        
        return response
    
    def get_constellation_info(self) -> Dict:
        """
        Get information about this constellation.
        
        Returns:
        -------
        Dict
            Constellation information
        """
        agent_info = {}
        for agent_type, agent in self.agents.items():
            agent_info[agent_type] = {
                "role": self._get_agent_role(agent_type),
                "capabilities": agent.get_agent_capabilities()
            }
        
        info = {
            "constellation_type": self.constellation_type,
            "description": self.constellation_config.get("description", ""),
            "primary_goal": self.constellation_config.get("primary_goal", ""),
            "secondary_goals": self.constellation_config.get("secondary_goals", []),
            "agents": agent_info
        }
        
        return info
    
    def _initialize_agents(self):
        """Initialize all agents for this constellation."""
        for agent_config in self.constellation_config.get("agents", []):
            agent_type = agent_config["type"]
            agent_role = agent_config["role"]
            
            # Create agent-specific memory path
            agent_memory_path = self._create_agent_memory_path(agent_type)
            
            # Create agent instance
            agent = self._create_agent(agent_type, agent_memory_path)
            
            if agent:
                self.agents[agent_type] = agent
                if self.is_logging:
                    logger.info(f"Added {agent_type} agent with {agent_role} role to constellation")

    def _create_agent_memory_path(self, agent_type: str) -> Path:
        """
        Create a proper memory path for a specific agent.
        
        Parameters:
        ----------
        agent_type : str
            Type of agent to create memory path for
            
        Returns:
        -------
        Path
            Agent-specific memory file path
        """
        # Sanitize agent type for file naming
        sanitized_agent_type = agent_type.strip().lower().replace(" ", "_")
        
        # Determine base memory directory
        if self.memory_path:
            # If memory_path is provided, use it as base directory
            if str(self.memory_path).endswith('.json'):
                # If it's a JSON file, use its directory
                base_memory_dir = self.memory_path.parent
            else:
                # Use it as directory
                base_memory_dir = Path(self.memory_path)
        else:
            # Default to 'memory' directory
            base_memory_dir = Path('memory')
        
        # Ensure memory directory exists
        base_memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Create agent-specific memory file path
        memory_filename = f"{sanitized_agent_type}_memory.json"
        agent_memory_path = base_memory_dir / memory_filename
        
        if self.is_logging:
            logger.info(f"Created memory path for {agent_type}: {agent_memory_path}")
        
        return agent_memory_path

    def _create_agent(self, agent_type: str, memory_path: Optional[Path] = None) -> Optional[SpecializedAgent]:
        """
        Create a specialized agent instance.
        """
        import time
        start_time = time.time()
        try:
            from importlib import import_module
            import os
            # Defensive: sanitize agent_type
            sanitized_agent_type = agent_type.strip().lower().replace(" ", "_")
            # Log agent_type and memory_path
            logger.info(f"Attempting to create agent: agent_type='{sanitized_agent_type}', memory_path='{memory_path}'")
            
            # Ensure memory path is properly configured
            if not memory_path:
                memory_path = self._create_agent_memory_path(sanitized_agent_type)
            elif not str(memory_path).endswith('.json'):
                logger.warning(f"memory_path '{memory_path}' is not a .json file. Creating proper path.")
                memory_path = self._create_agent_memory_path(sanitized_agent_type)
            
            # Ensure memory directory exists
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Dynamically import the agent class
            module = import_module(f"src.GAAPF.core.agents.{sanitized_agent_type}")
            
            # Get the agent class (assuming a convention like 'CodeAssistant' -> 'CodeAssistantAgent')
            class_name = f"{''.join(word.capitalize() for word in sanitized_agent_type.split('_'))}Agent"
            agent_class = getattr(module, class_name)
            
            # Create agent instance
            agent_instance = agent_class(
                llm=self.llm,
                memory_path=memory_path,
                is_logging=self.is_logging
            )
            
            # Log success with elapsed time
            elapsed = time.time() - start_time
            logger.info(f"Successfully created {agent_type} agent in {elapsed:.3f} seconds")
            
            return agent_instance

        except (ImportError, AttributeError, Exception) as e:
            # Log the error with elapsed time
            elapsed = time.time() - start_time
            logger.error(f"Error creating {agent_type} agent: {e} (elapsed {elapsed:.3f} seconds)")

            # Fallback to MockAgent if specific agent fails
            logger.info(f"Creating mock agent for {agent_type} as fallback")
            from .constellation import MockAgent
            return MockAgent(
                llm=self.llm,
                agent_type=agent_type,
                memory_path=memory_path,
                is_logging=self.is_logging
            )
        
        except Exception as e:
            logger.error(f"Critical error in _create_agent for {agent_type}: {e}")
            return None
    
    def _get_agent_role(self, agent_type: str) -> str:
        """Get the role of a specific agent."""
        for agent_config in self.constellation_config.get("agents", []):
            if agent_config["type"] == agent_type:
                return agent_config.get("role", "support")
        return "support"
    
    def _generate_fallback_response(self, interaction_data: Dict, learning_context: Dict) -> str:
        """Generate a fallback response using the LLM."""
        user_query = interaction_data.get("query", "your request")
        prompt = f"""
        An error occurred while processing the user's request: '{user_query}'.
        As a helpful assistant, apologize for the issue and ask the user to try rephrasing their question or trying a different activity.
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return "I'm sorry, but I encountered an error and cannot process your request at the moment. Please try again later."

def create_constellation_for_context(
    llm: BaseLanguageModel,
    learning_context: Dict,
    user_id: str = "default_user",
    memory_path: Optional[Path] = None,
    is_logging: bool = False
) -> Constellation:
    """
    Create an appropriate constellation based on learning context.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use for all agents
    learning_context : Dict
        Current learning context
    user_id : str, optional
        Identifier for the user
    memory_path : Path, optional
        Base path for agent memory files
    is_logging : bool, optional
        Flag to enable detailed logging
        
    Returns:
    -------
    Constellation
        Created constellation instance
    """
    # Get recommended constellation types
    recommended_types = get_recommended_constellation_types(learning_context)
    
    # Use first recommended type
    constellation_type = recommended_types[0] if recommended_types else "learning"
    
    # Create constellation
    constellation = Constellation(
        llm=llm,
        constellation_type=constellation_type,
        user_id=user_id,
        memory_path=memory_path,
        is_logging=is_logging
    )
    
    return constellation