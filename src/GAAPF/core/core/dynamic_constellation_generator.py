"""Dynamic Constellation Generator for GAAPF Architecture

This module provides LLM-driven dynamic constellation generation that can:
1. Generate custom constellation types based on learning context
2. Dynamically compose agent teams for specific learning scenarios
3. Adapt constellation parameters using LLM intelligence
4. Provide intelligent recommendations beyond fixed rules

This extends the existing constellation_types.py with dynamic capabilities.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path

# Import existing constellation types as fallback
from .constellation_types import CONSTELLATION_TYPES, get_constellation_type

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicConstellationGenerator:
    """
    LLM-powered dynamic constellation generator.
    
    This class uses an LLM to generate custom constellation types and agent
    compositions based on specific learning contexts, going beyond the fixed
    constellation types to provide truly adaptive learning experiences.
    """
    
    def __init__(
        self, 
        llm: BaseLanguageModel,
        fallback_to_fixed: bool = True,
        cache_enabled: bool = True,
        is_logging: bool = False
    ):
        """
        Initialize the dynamic constellation generator.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model to use for constellation generation
        fallback_to_fixed : bool, optional
            Whether to fallback to fixed constellation types on failure
        cache_enabled : bool, optional
            Whether to cache generated constellations
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        self.llm = llm
        self.fallback_to_fixed = fallback_to_fixed
        self.cache_enabled = cache_enabled
        self.is_logging = is_logging
        self._constellation_cache = {}
        self._cache_ttl = 3600  # 1 hour cache TTL
        
        # Available agent types with their capabilities
        self.available_agents = {
            "instructor": {
                "capabilities": ["concept_explanation", "theoretical_teaching", "curriculum_guidance"],
                "best_for": ["learning", "concept_introduction", "theory"]
            },
            "practice_facilitator": {
                "capabilities": ["exercise_creation", "hands_on_guidance", "skill_development"],
                "best_for": ["practice", "skill_building", "exercises"]
            },
            "code_assistant": {
                "capabilities": ["code_generation", "implementation_help", "debugging_support"],
                "best_for": ["coding", "implementation", "development"]
            },
            "assessment": {
                "capabilities": ["knowledge_evaluation", "progress_tracking", "feedback_provision"],
                "best_for": ["evaluation", "testing", "progress_check"]
            },
            "project_guide": {
                "capabilities": ["project_planning", "milestone_tracking", "integration_guidance"],
                "best_for": ["projects", "application", "real_world_tasks"]
            },
            "troubleshooter": {
                "capabilities": ["error_diagnosis", "problem_solving", "debugging"],
                "best_for": ["debugging", "error_resolution", "troubleshooting"]
            },
            "mentor": {
                "capabilities": ["motivation", "guidance", "learning_strategy"],
                "best_for": ["support", "motivation", "learning_path"]
            },
            "documentation_expert": {
                "capabilities": ["reference_provision", "documentation_search", "api_guidance"],
                "best_for": ["reference", "documentation", "api_help"]
            },
            "knowledge_synthesizer": {
                "capabilities": ["concept_connection", "knowledge_integration", "pattern_recognition"],
                "best_for": ["synthesis", "connections", "big_picture"]
            },
            "research_assistant": {
                "capabilities": ["information_gathering", "resource_finding", "research_support"],
                "best_for": ["research", "information", "resources"]
            },
            "motivational_coach": {
                "capabilities": ["encouragement", "motivation_maintenance", "confidence_building"],
                "best_for": ["motivation", "encouragement", "confidence"]
            },
            "progress_tracker": {
                "capabilities": ["progress_monitoring", "milestone_tracking", "analytics"],
                "best_for": ["tracking", "progress", "analytics"]
            }
        }
        
        if self.is_logging:
            logger.info(f"✅ Initialized DynamicConstellationGenerator with {len(self.available_agents)} agent types")
    
    def generate_dynamic_constellation(
        self, 
        learning_context: Dict,
        user_query: str = "",
        custom_requirements: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Generate a dynamic constellation based on learning context and requirements.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context including user profile, progress, etc.
        user_query : str, optional
            Current user query for context
        custom_requirements : Dict, optional
            Custom requirements for the constellation
            
        Returns:
        -------
        Optional[Dict]
            Generated constellation configuration or None if generation fails
        """
        start_time = time.time()
        
        if self.is_logging:
            logger.info(f"🎯 Generating dynamic constellation for context: {learning_context.get('current_activity', 'unknown')}")
        
        # Check cache first
        cache_key = self._generate_cache_key(learning_context, user_query, custom_requirements)
        if self.cache_enabled and cache_key in self._constellation_cache:
            cached_result, timestamp = self._constellation_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                if self.is_logging:
                    logger.info("📋 Using cached constellation")
                return cached_result
        
        try:
            # Generate constellation using LLM
            constellation = self._llm_generate_constellation(
                learning_context, user_query, custom_requirements
            )
            
            if constellation:
                # Cache successful result
                if self.cache_enabled:
                    self._constellation_cache[cache_key] = (constellation, time.time())
                
                generation_time = time.time() - start_time
                if self.is_logging:
                    logger.info(f"✅ Dynamic constellation generated successfully in {generation_time:.2f}s")
                
                return constellation
            else:
                raise ValueError("LLM returned empty constellation")
                
        except Exception as e:
            if self.is_logging:
                logger.error(f"❌ Dynamic constellation generation failed: {e}")
            
            # Fallback to fixed constellation types if enabled
            if self.fallback_to_fixed:
                return self._get_fallback_constellation(learning_context)
            
            return None
    
    def _llm_generate_constellation(
        self,
        learning_context: Dict,
        user_query: str,
        custom_requirements: Optional[Dict]
    ) -> Optional[Dict]:
        """
        Use LLM to generate a custom constellation configuration.
        """
        # Prepare context for LLM
        context_summary = self._prepare_context_summary(learning_context)
        agent_info = self._prepare_agent_info()
        requirements_info = self._prepare_requirements_info(custom_requirements)
        
        prompt = f"""You are an expert AI system architect specializing in multi-agent learning systems.
Your task is to design a custom constellation (team) of AI agents optimized for a specific learning scenario.

**LEARNING CONTEXT:**
{context_summary}

**USER QUERY:**
"{user_query}"

**AVAILABLE AGENTS:**
{agent_info}

**CUSTOM REQUIREMENTS:**
{requirements_info}

**YOUR TASK:**
Design an optimal constellation by:
1. Analyzing the learning context and user needs
2. Selecting 3-6 agents that work best together for this scenario
3. Assigning appropriate roles (primary, secondary, support)
4. Defining clear goals and agent descriptions
5. Ensuring the team composition addresses the specific learning needs

**RESPONSE FORMAT:**
Respond with a JSON object following this exact structure:

{{
  "name": "<Descriptive constellation name>",
  "description": "<Brief description of constellation purpose>",
  "primary_goal": "<Main objective of this constellation>",
  "secondary_goals": [
    "<Secondary goal 1>",
    "<Secondary goal 2>"
  ],
  "agents": [
    {{
      "type": "<agent_type>",
      "role": "primary|secondary|support",
      "description": "<Specific role description for this scenario>"
    }}
  ],
  "optimization_focus": "<What this constellation is optimized for>",
  "confidence": <0.0-1.0 confidence score>
}}

**GUIDELINES:**
- Include 1-2 primary agents (main drivers)
- Include 1-3 secondary agents (supporting roles)
- Include 1-2 support agents (auxiliary help)
- Ensure agent types exist in the available agents list
- Tailor descriptions to the specific learning scenario
- Focus on synergy between selected agents
"""
        
        try:
            messages = [
                SystemMessage(
                    content="You are an expert multi-agent system architect. Respond only with valid JSON."
                ),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # Extract JSON from response
            constellation = self._extract_json_from_response(content)
            
            if constellation and self._validate_constellation(constellation):
                return constellation
            else:
                raise ValueError("Invalid constellation structure")
                
        except Exception as e:
            if self.is_logging:
                logger.error(f"LLM constellation generation failed: {e}")
            return None
    
    def _prepare_context_summary(self, learning_context: Dict) -> str:
        """
        Prepare a concise summary of the learning context for the LLM.
        """
        summary_parts = []
        
        # User information
        user_profile = learning_context.get("user_profile", {})
        if user_profile:
            level = user_profile.get("level", "unknown")
            preferences = user_profile.get("learning_preferences", {})
            summary_parts.append(f"- User Level: {level}")
            if preferences:
                summary_parts.append(f"- Learning Style: {preferences.get('style', 'adaptive')}")
        
        # Current learning state
        current_module = learning_context.get("current_module", "")
        current_activity = learning_context.get("current_activity", "")
        learning_stage = learning_context.get("learning_stage", "")
        
        if current_module:
            summary_parts.append(f"- Current Module: {current_module}")
        if current_activity:
            summary_parts.append(f"- Current Activity: {current_activity}")
        if learning_stage:
            summary_parts.append(f"- Learning Stage: {learning_stage}")
        
        # Framework information
        framework_config = learning_context.get("framework_config", {})
        if framework_config:
            framework_name = framework_config.get("name", "unknown")
            summary_parts.append(f"- Framework: {framework_name}")
        
        # Progress information
        interaction_count = learning_context.get("interaction_count", 0)
        if interaction_count > 0:
            summary_parts.append(f"- Session Interactions: {interaction_count}")
        
        return "\n".join(summary_parts) if summary_parts else "No specific context available"
    
    def _prepare_agent_info(self) -> str:
        """
        Prepare agent information for the LLM.
        """
        agent_descriptions = []
        for agent_type, info in self.available_agents.items():
            capabilities = ", ".join(info["capabilities"])
            best_for = ", ".join(info["best_for"])
            agent_descriptions.append(
                f"- **{agent_type}**: Capabilities: {capabilities}. Best for: {best_for}"
            )
        
        return "\n".join(agent_descriptions)
    
    def _prepare_requirements_info(self, custom_requirements: Optional[Dict]) -> str:
        """
        Prepare custom requirements information for the LLM.
        """
        if not custom_requirements:
            return "No specific custom requirements"
        
        req_parts = []
        for key, value in custom_requirements.items():
            req_parts.append(f"- {key}: {value}")
        
        return "\n".join(req_parts)
    
    def _extract_json_from_response(self, content: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response using multiple patterns.
        """
        import re
        
        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _validate_constellation(self, constellation: Dict) -> bool:
        """
        Validate the generated constellation structure.
        """
        required_fields = ["name", "description", "primary_goal", "agents"]
        
        # Check required fields
        for field in required_fields:
            if field not in constellation:
                return False
        
        # Validate agents
        agents = constellation.get("agents", [])
        if not agents or len(agents) < 2 or len(agents) > 8:
            return False
        
        # Check agent structure and types
        for agent in agents:
            if not isinstance(agent, dict):
                return False
            if "type" not in agent or "role" not in agent:
                return False
            if agent["type"] not in self.available_agents:
                return False
            if agent["role"] not in ["primary", "secondary", "support"]:
                return False
        
        # Ensure at least one primary agent
        primary_agents = [a for a in agents if a["role"] == "primary"]
        if not primary_agents:
            return False
        
        return True
    
    def _get_fallback_constellation(self, learning_context: Dict) -> Optional[Dict]:
        """
        Get a fallback constellation from fixed types.
        """
        # Use existing recommendation logic as fallback
        from .constellation_types import get_recommended_constellation_types
        
        recommended_types = get_recommended_constellation_types(learning_context)
        if recommended_types:
            constellation_type = recommended_types[0]
            return get_constellation_type(constellation_type)
        
        # Ultimate fallback
        return get_constellation_type("learning")
    
    def _generate_cache_key(self, learning_context: Dict, user_query: str, custom_requirements: Optional[Dict]) -> str:
        """
        Generate a cache key for constellation caching.
        """
        key_components = [
            learning_context.get("current_activity", ""),
            learning_context.get("learning_stage", ""),
            learning_context.get("current_module", ""),
            user_query[:100] if user_query else "",
            str(custom_requirements) if custom_requirements else ""
        ]
        return hash(tuple(key_components))
    
    def get_dynamic_agent_composition(
        self,
        task_description: str,
        learning_context: Dict,
        max_agents: int = 5
    ) -> List[Dict]:
        """
        Get dynamic agent composition for a specific task.
        
        Parameters:
        ----------
        task_description : str
            Description of the task to be performed
        learning_context : Dict
            Current learning context
        max_agents : int, optional
            Maximum number of agents to include
            
        Returns:
        -------
        List[Dict]
            List of agent configurations for the task
        """
        constellation = self.generate_dynamic_constellation(
            learning_context=learning_context,
            user_query=task_description,
            custom_requirements={"max_agents": max_agents}
        )
        
        if constellation and "agents" in constellation:
            return constellation["agents"]
        
        # Fallback to basic composition
        return [
            {"type": "instructor", "role": "primary", "description": "Provides guidance and instruction"},
            {"type": "code_assistant", "role": "secondary", "description": "Assists with implementation"}
        ]


# Convenience functions for backward compatibility and easy integration

def create_dynamic_constellation_generator(
    llm: BaseLanguageModel,
    **kwargs
) -> DynamicConstellationGenerator:
    """
    Create a dynamic constellation generator instance.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    **kwargs
        Additional arguments for the generator
        
    Returns:
    -------
    DynamicConstellationGenerator
        Configured generator instance
    """
    return DynamicConstellationGenerator(llm=llm, **kwargs)


def get_dynamic_constellation_type(
    llm: BaseLanguageModel,
    learning_context: Dict,
    user_query: str = "",
    fallback_to_fixed: bool = True
) -> Optional[Dict]:
    """
    Get a dynamic constellation type for the given context.
    
    This function provides a simple interface for getting dynamic constellations
    without managing the generator instance.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    learning_context : Dict
        Current learning context
    user_query : str, optional
        Current user query
    fallback_to_fixed : bool, optional
        Whether to fallback to fixed types on failure
        
    Returns:
    -------
    Optional[Dict]
        Generated constellation configuration
    """
    generator = DynamicConstellationGenerator(
        llm=llm,
        fallback_to_fixed=fallback_to_fixed
    )
    
    return generator.generate_dynamic_constellation(
        learning_context=learning_context,
        user_query=user_query
    )