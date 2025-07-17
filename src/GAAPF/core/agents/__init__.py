"""
Specialized Agents for GAAPF Architecture

This module provides specialized AI agents for different learning tasks.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool

from ..agent.agent import Agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpecializedAgent(Agent):
    """
    Base class for specialized learning agents in the GAAPF architecture.
    
    This class extends the basic Agent class with specialized capabilities
    for educational and learning contexts.
    """
    
    # Class attributes for agent registry support
    DESCRIPTION = "Base specialized learning agent"
    CAPABILITIES = ["learning_support", "educational_guidance"]
    PRIORITY = 0  # Default priority for agent selection
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[Union[str, BaseTool]] = [],
        memory_path: Optional[Path] = None,
        config: Dict = None,
        agent_type: str = "specialized",
        description: str = "Specialized learning agent",
        is_logging: bool = False,
        *args, **kwargs
    ):
        """
        Initialize a specialized agent.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model to use for this agent
        tools : List[Union[str, BaseTool]], optional
            Tools available to this agent
        memory_path : Path, optional
            Path to agent memory file
        config : Dict, optional
            Agent-specific configuration
        agent_type : str, optional
            Type identifier for this agent
        description : str, optional
            Description of the agent's capabilities
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        # Initialize the base agent
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            description=description,
            is_reset_memory=False,
            is_logging=is_logging,
            *args, **kwargs
        )
        
        # Store specialized agent attributes
        self.config = config or {}
        self.agent_type = agent_type
        self._agent_type = agent_type  # Set the agent type for memory context
        
        if self.is_logging:
            logger.info(f"Initialized {agent_type} agent with config: {self.config}")
    
    def get_agent_capabilities(self) -> Dict:
        """
        Get the capabilities of this agent.
        
        Returns:
        -------
        Dict
            Dictionary describing the agent's capabilities
        """
        return {
            "agent_type": self.agent_type,
            "description": self.description,
            "tools": [str(tool) for tool in self.tools],
            "config": self.config
        }
    
    def _generate_system_prompt(self, learning_context: Optional[Dict] = None) -> str:
        """
        Generate a system prompt for this agent.
        This should be overridden by specialized agents.
        
        Parameters:
        ----------
        learning_context : Dict, optional
            Current learning context, if available.

        Returns:
        -------
        str
            System prompt for the agent
        """
        if not learning_context:
            return f"""You are a {self.agent_type} agent. {self.description}

Your role is to assist users with learning and understanding various topics.

**Your Capabilities:**
- Provide expert guidance in your specialization area
- Adapt explanations to user's level and needs  
- Offer practical examples and hands-on guidance
- Maintain supportive and encouraging interactions

When helping users, focus on being clear, helpful, and educational in your responses.
"""
        
        # Extract context information with fallbacks
        user_profile = learning_context.get("user_profile", {})
        framework_config = learning_context.get("framework_config", {})
        
        user_name = user_profile.get("name", "learner")
        user_level = user_profile.get("experience_level", "intermediate")
        framework_name = framework_config.get("name", "the framework")
        current_module = learning_context.get("current_module", "current topics")
        agent_task = learning_context.get("agent_task")
        
        task_instruction = f"\\nYour specific task for this interaction is: {agent_task}" if agent_task else ""

        return f"""You are a {self.agent_type} agent helping {user_name}, a {user_level} level learner.{task_instruction}

**Current Context:**
- Student: {user_name} ({user_level} level)
- Learning Focus: {framework_name}
- Current Module: {current_module}
- Your Role: {self.description}

**Your Mission:**
As a {self.agent_type} agent, provide specialized assistance to help {user_name} learn {framework_name} effectively.

**Your Approach:**
1. **Personalized:** Address {user_name} by name and adapt to their {user_level} level
2. **Focused:** Keep responses relevant to {framework_name} and current learning goals
3. **Supportive:** Maintain encouraging and helpful tone
4. **Expert:** Leverage your {self.agent_type} specialization to provide valuable insights

Build upon previous conversation context and guide {user_name} toward their learning objectives.
"""
    
    def _enhance_query_with_context(self, query: str, learning_context: Dict) -> str:
        """
        Enhance a user query with comprehensive learning context including conversation history.
        This can be overridden by specialized agents for specific enhancements.
        
        Parameters:
        ----------
        query : str
            Original user query
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        str
            Enhanced query with context
        """
        # Get comprehensive user and framework info
        user_profile = learning_context.get("user_profile", {})
        framework_config = learning_context.get("framework_config", {})
        
        user_name = user_profile.get("name", "learner")
        user_level = user_profile.get("experience_level", "beginner")
        current_module = learning_context.get("current_module", "introduction")
        framework_name = framework_config.get("name", "the framework")
        
        # Get learning progression context
        session_id = learning_context.get("session_id", "")
        interaction_count = learning_context.get("interaction_count", 0)
        learning_stage = learning_context.get("learning_stage", "exploration")
        
        # Get conversation history context
        messages = learning_context.get("messages", [])
        conversation_context = self._build_conversation_context(messages)
        
        # Get progress and curriculum context
        progress_metrics = learning_context.get("progress_metrics", {})
        current_concepts = learning_context.get("current_concepts", [])
        recent_topics = learning_context.get("recent_topics", [])
        
        # Get enhanced context from enhanced learning context
        conversation_analysis = learning_context.get("conversation_analysis", {})
        learning_insights = learning_context.get("learning_insights", {})
        
        # Build comprehensive context prefix
        context_prefix = f"""
LEARNING SESSION CONTEXT:
You are helping {user_name}, a {user_level} level learner who is studying {framework_name}.

CURRENT STATUS:
- Framework: {framework_name}
- User Level: {user_level}
- Current Module: {current_module}
- Learning Stage: {learning_stage}
- Session Interactions: {interaction_count}
- User Name: {user_name}

CURRICULUM CONTEXT:
- Current Concepts: {', '.join(current_concepts) if current_concepts else 'Not specified'}
- Recent Topics: {', '.join(recent_topics) if recent_topics else 'None'}

PROGRESS TRACKING:
- Module Progress: {progress_metrics.get('current_module_progress', 0):.1%} if progress_metrics else 'Starting'
- Overall Progress: {progress_metrics.get('completion_percentage', 0):.1f}% if progress_metrics else '0%'
- Engagement: {learning_insights.get('engagement_pattern', 'starting').replace('_', ' ').title() if learning_insights else 'Starting'}

CONVERSATION CONTINUITY:
{conversation_context}

AGENT ROLE: As a {self.agent_type} agent, provide responses that:
1. Address {user_name} by name when appropriate
2. Are tailored to a {user_level} learner
3. Build upon previous conversation topics
4. Focus on {framework_name} framework concepts
5. Support the current learning stage: {learning_stage}
6. Reference current module: {current_module}
7. Maintain conversation continuity and learning progression

USER QUERY: {query}

RESPONSE GUIDELINES:
- Build on the conversation history shown above
- Provide {self.agent_type}-specific expertise
- Guide toward appropriate next learning steps
- Reference relevant concepts from the current module
- Maintain encouraging and supportive tone
"""
        
        return context_prefix
    
    def _build_conversation_context(self, messages: List[Dict]) -> str:
        """
        Build conversation context from message history.
        
        Parameters:
        ----------
        messages : List[Dict]
            List of conversation messages
            
        Returns:
        -------
        str
            Formatted conversation context
        """
        if not messages or len(messages) < 2:
            return "- This is the beginning of your learning conversation"
        
        # Get recent messages for context (last 6 messages, 3 exchanges)
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        
        context_lines = ["Recent Conversation Summary:"]
        
        # Group messages into exchanges
        exchanges = []
        current_exchange = {}
        
        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                if current_exchange:
                    exchanges.append(current_exchange)
                current_exchange = {"user": content[:150]}  # Limit length
            elif role == "assistant" and "user" in current_exchange:
                current_exchange["assistant"] = content[:150]  # Limit length
        
        if current_exchange:
            exchanges.append(current_exchange)
        
        # Format recent exchanges
        for i, exchange in enumerate(exchanges[-3:], 1):  # Last 3 exchanges
            if "user" in exchange:
                context_lines.append(f"  Exchange {i}:")
                context_lines.append(f"    User: {exchange['user']}...")
                if "assistant" in exchange:
                    context_lines.append(f"    AI: {exchange['assistant']}...")
        
        # Add conversation insights
        total_messages = len(messages)
        context_lines.append(f"- Total conversation length: {total_messages} messages")
        
        if total_messages >= 6:
            context_lines.append("- This is an ongoing, in-depth learning conversation")
        elif total_messages >= 3:
            context_lines.append("- Learning conversation is developing momentum")
        
        return "\n".join(context_lines)
    
    def _process_response(self, response: Any, learning_context: Dict) -> Dict:
        """
        Process and structure the agent's response with enhanced metadata.
        This can be overridden by specialized agents for specific processing.
        
        Parameters:
        ----------
        response : Any
            Raw response from the agent
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        Dict
            Structured response data with enhanced metadata
        """
        # Extract content from response
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        # Build structured response
        structured_response = {
            "agent_type": self.agent_type,
            "content": content,
            "timestamp": learning_context.get("interaction_timestamp"),
            "learning_metadata": {
                "current_module": learning_context.get("current_module", ""),
                "learning_stage": learning_context.get("learning_stage", "exploration"),
                "interaction_count": learning_context.get("interaction_count", 0),
                "user_level": learning_context.get("user_profile", {}).get("experience_level", "beginner")
            }
        }
        
        # Add conversation metadata if available
        conversation_analysis = learning_context.get("conversation_analysis", {})
        if conversation_analysis:
            structured_response["conversation_metadata"] = {
                "conversation_depth": conversation_analysis.get("conversation_depth", "surface"),
                "learning_momentum": conversation_analysis.get("learning_momentum", "starting"),
                "recent_topics": conversation_analysis.get("recent_interaction_topics", [])
            }
        
        # Add learning insights if available
        learning_insights = learning_context.get("learning_insights", {})
        if learning_insights:
            structured_response["learning_insights"] = {
                "engagement_pattern": learning_insights.get("engagement_pattern", "just_starting"),
                "suggested_transition": learning_insights.get("suggested_transition"),
                "readiness_for_next": learning_insights.get("readiness_for_next", False)
            }
        
        return structured_response


# Import all the specialized agent classes
from .assessment import AssessmentAgent
from .code_assistant import CodeAssistantAgent
from .documentation_expert import DocumentationExpertAgent
from .instructor import InstructorAgent
from .knowledge_synthesizer import KnowledgeSynthesizerAgent
from .mentor import MentorAgent
from .motivational_coach import MotivationalCoachAgent
from .practice_facilitator import PracticeFacilitatorAgent
from .progress_tracker import ProgressTrackerAgent
from .project_guide import ProjectGuideAgent
from .research_assistant import ResearchAssistantAgent
from .troubleshooter import TroubleshooterAgent

__all__ = [
    "SpecializedAgent",
    "AssessmentAgent",
    "CodeAssistantAgent", 
    "DocumentationExpertAgent",
    "InstructorAgent",
    "KnowledgeSynthesizerAgent",
    "MentorAgent",
    "MotivationalCoachAgent",
    "PracticeFacilitatorAgent",
    "ProgressTrackerAgent",
    "ProjectGuideAgent",
    "ResearchAssistantAgent",
    "TroubleshooterAgent"
]