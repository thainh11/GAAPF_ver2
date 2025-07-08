import logging
from typing import Dict, List, Optional, Union, Any, TypedDict
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from ..graph.function_graph import node, NodeWrapper
from langgraph.graph import END
from src.GAAPF.prompts.instructor import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstructorState(TypedDict):
    query: str
    analysis: str
    response: str

class InstructorAgent(SpecializedAgent):
    """
    Enhanced Specialized agent focused on providing expert instruction and explanations.
    
    The InstructorAgent is responsible for:
    1. Explaining core framework concepts and principles
    2. Providing theoretical background and context
    3. Breaking down complex ideas into understandable components
    4. Adapting explanations to the user's experience level
    5. Breaking down complex topics into understandable chunks
    6. Guiding users through structured learning paths
    7. Providing proactive guidance and next steps
    8. Maintaining conversation continuity and context
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[Union[str, BaseTool]] = [],
        memory_path: Optional[Path] = None,
        config: Dict = None,
        is_logging: bool = False,
        *args, **kwargs
    ):
        """
        Initialize the InstructorAgent.
        
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
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        # Set default config if not provided
        if config is None:
            config = {
                "explanation_depth": "balanced",  # basic, balanced, detailed, advanced
                "pace": "moderate",  # gradual, moderate, rapid
                "theoretical_focus": "medium",  # low, medium, high
                "use_analogies": True,
                "use_examples": True,
                "curriculum_driven": True,  # Enable curriculum-driven responses
                "interactive_style": True,   # Enable interactive teaching style
                "teaching_style": "socratic",  # socratic, direct, narrative
                "proactive_guidance": True,  # Enable proactive guidance
                "context_aware": True,       # Enable context awareness
            }
        
        # Set default tools if not provided
        if not tools:
            tools = [
                "websearch_tools",
                "deepsearch"
            ]
        
        # Define flow
        analysis_node  = NodeWrapper(self.analyze_query,  name="analyze_query")
        response_node  = NodeWrapper(self.generate_response, name="generate_response")
        flow = [analysis_node >> response_node, response_node >> END]

        # Initialize the base specialized agent
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            config=config,
            agent_type="instructor",
            description="Expert in providing clear explanations and structured learning with proactive guidance",
            is_logging=is_logging,
            flow=flow,
            state_schema=InstructorState,
            *args, **kwargs
        )
        
        if self.is_logging:
            logger.info(f"✅ Initialized InstructorAgent with enhanced config: {self.config}")
    
    def analyze_query(self, state: InstructorState) -> Dict:
        """Enhanced query analysis with context awareness and intent detection."""
        user_query = state["query"]
        
        if self.is_logging:
            logger.info(f"🔍 InstructorAgent analyzing query: {user_query[:100]}...")
        
        # Enhanced intent detection for readiness signals
        readiness_signals = {
            "ready", "i'm ready", "let's go", "let's start", "continue", 
            "yes", "proceed", "next", "go ahead", "start", "begin"
        }
        
        practice_signals = {
            "practice", "hands-on", "exercise", "coding", "code", "implement", 
            "build", "create", "show me how", "example", "demo"
        }
        
        query_lower = user_query.lower().strip()
        
        # Detect readiness intent
        if any(signal in query_lower for signal in readiness_signals):
            if self.is_logging:
                logger.info("🎯 Detected readiness signal - preparing proactive continuation")
            
            system_message = (
                "The user has indicated readiness to continue learning. You are an expert instructor "
                "who should provide proactive guidance and immediately continue with the next logical "
                "step in their curriculum. DO NOT ask 'what are you ready for?' - instead, confidently "
                "guide them to the next learning objective with specific actionable tasks. "
                "Be encouraging and provide clear next steps."
            )
        elif any(signal in query_lower for signal in practice_signals):
            if self.is_logging:
                logger.info("🛠️ Detected practice intent - preparing hands-on guidance")
            
            system_message = (
                "The user wants hands-on practice. Provide immediate practical exercises, "
                "code examples, or implementation tasks related to their current learning module. "
                "Be specific and actionable in your guidance."
            )
        else:
            system_message = (
                "You are an expert instructor. Provide clear, helpful, and proactive guidance "
                "that builds on previous conversation context. Always include next steps "
                "and learning recommendations."
            )
        
        return {"analysis": system_message}

    def generate_response(self, state: InstructorState) -> Dict:
        """Enhanced response generation with proactive guidance."""
        system_prompt = state["analysis"]
        user_query = state["query"]

        if self.is_logging:
            logger.info("🧠 InstructorAgent generating enhanced response with proactive guidance")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]

        response_message = self.llm.invoke(messages)
        
        if self.is_logging:
            logger.info("✅ InstructorAgent response generated successfully")
            
        return {"response": response_message.content}

    def _generate_system_prompt(self, learning_context: Dict = None) -> str:
        """
        Generate a dynamic, curriculum-aware system prompt for the Instructor.
        Enhanced with proactive guidance and context awareness.
        
        Parameters:
        ----------
        learning_context : Dict, optional
            The current learning context containing user and session details.
            
        Returns:
        -------
        str
            A detailed, dynamic system prompt with enhanced guidance capabilities.
        """
        if self.is_logging:
            logger.info("📝 Generating enhanced system prompt with learning context")
            
        return generate_system_prompt(self.config, learning_context)

    def _enhance_query_with_context(self, query: str, learning_context: Dict) -> str:
        """
        Enhanced query context enrichment with conversation continuity.
        """
        if self.is_logging:
            logger.info("🔗 Enhancing query with comprehensive learning context")
            
        # Get context information
        user_profile = learning_context.get("user_profile", {})
        framework_config = learning_context.get("framework_config", {})
        current_module = learning_context.get("current_module", "")
        interaction_count = learning_context.get("interaction_count", 0)
        messages = learning_context.get("messages", [])
        
        user_name = user_profile.get("name", "learner")
        framework_name = framework_config.get("name", "the framework")
        
        # Build conversation context
        conversation_summary = ""
        if messages and len(messages) > 1:
            recent_messages = messages[-4:]  # Last 4 messages for context
            conversation_summary = f"\nRecent conversation context:\n"
            for msg in recent_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:150]
                conversation_summary += f"- {role}: {content}...\n"
        
        # Enhanced context for instructors
        enhanced_query = f"""
INSTRUCTOR CONTEXT:
- Student: {user_name} (learning {framework_name})
- Current Module: {current_module}
- Interaction #{interaction_count}
- Teaching Style: Proactive and supportive

{conversation_summary}

INSTRUCTOR DIRECTIVE:
You are {user_name}'s dedicated instructor for {framework_name}. Your role is to:
1. Provide proactive, curriculum-driven guidance
2. Build upon previous interactions naturally
3. Offer specific next steps and actionable advice
4. Maintain encouraging and supportive tone
5. Detect intent (readiness, practice requests) and respond appropriately

STUDENT QUERY: {query}

RESPONSE REQUIREMENTS:
- Be proactive and guide the learning journey
- Reference current module and previous context when relevant
- Always include concrete next steps
- Use encouraging language
- Provide specific, actionable guidance
"""
        
        if self.is_logging:
            logger.info(f"🎯 Enhanced query prepared for {user_name} in {current_module}")
            
        return enhanced_query

    def _process_response(self, response: Any, learning_context: Dict) -> Dict:
        """
        Enhanced response processing with proactive guidance injection.
        
        Parameters:
        ----------
        response : Any
            Raw response from the agent
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        Dict
            Enhanced and structured response with proactive guidance
        """
        if self.is_logging:
            logger.info("⚡ Processing instructor response with enhancements")
            
        # Get base processed response
        processed = super()._process_response(response, learning_context)
        
        # Add instructor-specific metadata
        processed["instruction_type"] = "proactive_guidance"
        processed["concepts_covered"] = self._extract_concepts_from_response(
            processed["content"],
            learning_context
        )
        
        # Enhanced content with proactive learning guidance
        if self.config.get("proactive_guidance", True):
            enhanced_content = self._enhance_response_with_proactive_guidance(
                processed["content"], 
                learning_context
            )
            processed["content"] = enhanced_content
        
        if self.is_logging:
            logger.info("✅ Instructor response processing completed with proactive enhancements")
            
        return processed
    
    def _extract_concepts_from_response(self, response_content: str, learning_context: Dict) -> List[str]:
        """
        Extract the concepts covered in the response.
        """
        if self.is_logging:
            logger.info("📚 Extracting concepts from instructor response")
            
        # Get concepts for the current module
        framework_config = learning_context.get("framework_config", {})
        current_module = learning_context.get("current_module", "")
        
        all_concepts = []
        
        # Get module concepts
        if current_module and "modules" in framework_config:
            module_info = framework_config.get("modules", {}).get(current_module, {})
            all_concepts.extend(module_info.get("concepts", []))
        
        # Filter concepts that appear in the response
        covered_concepts = []
        for concept in all_concepts:
            if concept.lower() in response_content.lower():
                covered_concepts.append(concept)
        
        if self.is_logging:
            logger.info(f"📖 Extracted {len(covered_concepts)} concepts: {covered_concepts}")
            
        return covered_concepts
    
    def _enhance_response_with_proactive_guidance(self, content: str, learning_context: Dict) -> str:
        """
        Enhanced response with comprehensive proactive guidance.
        
        Parameters:
        ----------
        content : str
            Original response content
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        str
            Enhanced content with proactive guidance
        """
        if self.is_logging:
            logger.info("🚀 Adding proactive guidance to instructor response")
            
        # Extract key information from learning context
        current_module = learning_context.get("current_module", "")
        framework_config = learning_context.get("framework_config", {})
        interaction_count = learning_context.get("interaction_count", 0)
        user_profile = learning_context.get("user_profile", {})
        messages = learning_context.get("messages", [])
        
        # Get module and framework information
        modules = framework_config.get("modules", {})
        module_info = modules.get(current_module, {}) if current_module else {}
        concepts = module_info.get("concepts", [])
        framework_name = framework_config.get("name", "the framework")
        user_name = user_profile.get("name", "learner")
        
        # Only enhance if we have sufficient context
        if not concepts or not current_module:
            if self.is_logging:
                logger.info("⚠️ Limited context available - providing basic guidance")
            return content + "\n\n**🎯 Ready for the Next Step?**\nLet me know what you'd like to explore next!"
        
        # Detect user intent from last message
        last_query = ""
        if messages:
            last_message = messages[-1] if isinstance(messages[-1], dict) else None
            if last_message:
                last_query = last_message.get("content", "").lower()
        
        # Enhanced guidance based on interaction patterns
        guidance_section = "\n\n"
        
        # Proactive next steps based on curriculum progression
        if interaction_count <= 2:  # Early interactions
            guidance_section += f"**🌟 Welcome to Your {framework_name} Learning Journey, {user_name}!**\n\n"
            guidance_section += "**🎯 Your Next Learning Steps:**\n"
            if concepts:
                guidance_section += f"1. **Master the Basics**: Focus on {concepts[0]} to build your foundation\n"
                if len(concepts) > 1:
                    guidance_section += f"2. **Explore Further**: Once comfortable, we'll dive into {concepts[1]}\n"
            guidance_section += f"3. **Hands-On Practice**: Ask me 'Show me how to implement {concepts[0] if concepts else 'this'}'\n"
            guidance_section += f"4. **Ask Questions**: I'm here to clarify any {framework_name} concepts\n"
            
        elif interaction_count <= 5:  # Building understanding
            guidance_section += f"**🚀 Great Progress, {user_name}!**\n\n"
            guidance_section += "**🎯 Ready for More Advanced Topics?**\n"
            if len(concepts) > 1:
                guidance_section += f"1. **Deepen Your Skills**: Let's explore {concepts[1]} in more detail\n"
            guidance_section += f"2. **Practical Application**: Try building something with {framework_name}\n"
            guidance_section += f"3. **Connect Concepts**: See how different {framework_name} components work together\n"
            
        else:  # Advanced learning stage
            guidance_section += f"**💪 You're Making Excellent Progress, {user_name}!**\n\n"
            guidance_section += "**🎯 Advanced Learning Opportunities:**\n"
            guidance_section += f"1. **Advanced Patterns**: Explore complex {framework_name} implementations\n"
            guidance_section += f"2. **Real-World Projects**: Build production-ready applications\n"
            guidance_section += f"3. **Best Practices**: Learn optimization and advanced techniques\n"
        
        # Add interactive prompts
        guidance_section += "\n**💬 Interactive Learning:**\n"
        guidance_section += f"- Say 'I'm ready' to continue with the next topic\n"
        guidance_section += f"- Ask 'Show me an example' for hands-on practice\n"
        guidance_section += f"- Type 'Explain [concept]' for detailed explanations\n"
        
        # Add module-specific guidance
        if module_info.get("learning_objectives"):
            guidance_section += f"\n**📋 Current Module Objectives:**\n"
            for i, objective in enumerate(module_info["learning_objectives"][:3], 1):
                guidance_section += f"{i}. {objective}\n"
        
        if self.is_logging:
            logger.info("✅ Enhanced proactive guidance added to response")
            
        return content + guidance_section 