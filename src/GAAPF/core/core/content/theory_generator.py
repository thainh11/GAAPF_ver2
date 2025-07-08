"""
Theory content generation module for the GAAPF framework.

This module provides AI-powered generation of theoretical content
tailored to user learning levels and curriculum requirements.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

class TheoryGenerator:
    """
    AI-powered theory content generator.
    
    Generates comprehensive theoretical content including:
    - Concept explanations
    - Learning objectives
    - Key points and summaries
    - Progressive difficulty levels
    """
    
    def __init__(self, llm: BaseLanguageModel, framework_collector=None, is_logging: bool = False):
        """
        Initialize the theory generator.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model for content generation
        framework_collector : FrameworkCollector, optional
            Framework-specific information collector
        is_logging : bool
            Enable detailed logging
        """
        self.llm = llm
        self.framework_collector = framework_collector
        self.is_logging = is_logging
        
        # Content templates
        self.theory_templates = {
            "beginner": self._get_beginner_template(),
            "intermediate": self._get_intermediate_template(),
            "advanced": self._get_advanced_template()
        }
        
        if self.is_logging:
            logger.info("TheoryGenerator initialized")
    
    async def generate_theory_content(
        self,
        topic: str,
        framework_name: str,
        learning_level: str,
        learning_objectives: List[str],
        user_context: Dict[str, Any],
        content_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive theory content for a specific topic.
        
        Parameters:
        ----------
        topic : str
            The topic to generate content for
        framework_name : str
            Target framework
        learning_level : str
            User's learning level (beginner/intermediate/advanced)
        learning_objectives : List[str]
            Specific learning objectives to address
        user_context : Dict[str, Any]
            User's learning context and preferences
        content_type : str
            Type of content to generate
            
        Returns:
        -------
        Dict[str, Any]
            Generated theory content with metadata
        """
        try:
            if self.is_logging:
                logger.info(f"Generating theory content for topic: {topic}")
            
            # Get framework-specific information
            framework_info = await self._get_framework_context(framework_name)
            
            # Select appropriate template
            template = self.theory_templates.get(learning_level, self.theory_templates["beginner"])
            
            # Generate content
            content = await self._generate_structured_content(
                topic=topic,
                framework_name=framework_name,
                framework_info=framework_info,
                learning_level=learning_level,
                learning_objectives=learning_objectives,
                user_context=user_context,
                template=template
            )
            
            # Enhance content with examples
            enhanced_content = await self._enhance_with_examples(
                content, framework_name, learning_level
            )
            
            # Add metadata
            result = {
                "content": enhanced_content,
                "metadata": {
                    "topic": topic,
                    "framework_name": framework_name,
                    "learning_level": learning_level,
                    "content_type": content_type,
                    "generated_at": datetime.now().isoformat(),
                    "objectives_covered": learning_objectives,
                    "estimated_reading_time": self._calculate_reading_time(enhanced_content)
                }
            }
            
            if self.is_logging:
                logger.info(f"Theory content generated successfully for topic: {topic}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating theory content: {str(e)}")
            raise
    
    async def generate_concept_explanation(
        self,
        concept: str,
        framework_name: str,
        context: Dict[str, Any],
        explanation_depth: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Generate explanation for a specific concept.
        
        Parameters:
        ----------
        concept : str
            The concept to explain
        framework_name : str
            Target framework
        context : Dict[str, Any]
            Learning context
        explanation_depth : str
            Depth of explanation (brief/detailed/comprehensive)
            
        Returns:
        -------
        Dict[str, Any]
            Concept explanation with examples
        """
        try:
            prompt = ChatPromptTemplate.from_template(
                """
                Explain the concept of "{concept}" in the context of {framework_name}.
                
                Explanation depth: {explanation_depth}
                Learning level: {learning_level}
                
                Provide a {explanation_depth} explanation that includes:
                1. Clear definition
                2. Why it's important in {framework_name}
                3. How it relates to other concepts
                4. Practical applications
                5. Common misconceptions (if any)
                
                Format your response as JSON with the following structure:
                {{
                    "definition": "Clear, concise definition",
                    "importance": "Why this concept matters",
                    "relationships": ["Related concept 1", "Related concept 2"],
                    "applications": ["Application 1", "Application 2"],
                    "misconceptions": ["Common misconception 1"],
                    "key_points": ["Key point 1", "Key point 2"],
                    "difficulty_level": "estimated difficulty (1-10)"
                }}
                """
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "concept": concept,
                "framework_name": framework_name,
                "explanation_depth": explanation_depth,
                "learning_level": context.get("learning_level", "beginner")
            })
            
            return {
                "concept": concept,
                "explanation": result,
                "metadata": {
                    "framework_name": framework_name,
                    "explanation_depth": explanation_depth,
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating concept explanation: {str(e)}")
            raise
    
    async def generate_progressive_content(
        self,
        base_topic: str,
        framework_name: str,
        progression_levels: List[str],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate progressive content that builds complexity gradually.
        
        Parameters:
        ----------
        base_topic : str
            Base topic to build upon
        framework_name : str
            Target framework
        progression_levels : List[str]
            Ordered list of progression levels
        user_context : Dict[str, Any]
            User learning context
            
        Returns:
        -------
        Dict[str, Any]
            Progressive content structure
        """
        try:
            progressive_content = {}
            
            for i, level in enumerate(progression_levels):
                # Generate content for each level
                level_content = await self.generate_theory_content(
                    topic=f"{base_topic} - {level}",
                    framework_name=framework_name,
                    learning_level=user_context.get("learning_level", "beginner"),
                    learning_objectives=[f"Understand {level} aspects of {base_topic}"],
                    user_context=user_context,
                    content_type="progressive"
                )
                
                # Add prerequisites from previous levels
                if i > 0:
                    level_content["prerequisites"] = [
                        f"{base_topic} - {prev_level}" 
                        for prev_level in progression_levels[:i]
                    ]
                
                progressive_content[level] = level_content
            
            return {
                "base_topic": base_topic,
                "progression": progressive_content,
                "total_levels": len(progression_levels),
                "metadata": {
                    "framework_name": framework_name,
                    "generated_at": datetime.now().isoformat(),
                    "progression_type": "linear"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating progressive content: {str(e)}")
            raise
    
    async def _generate_structured_content(
        self,
        topic: str,
        framework_name: str,
        framework_info: Dict,
        learning_level: str,
        learning_objectives: List[str],
        user_context: Dict,
        template: ChatPromptTemplate
    ) -> Dict[str, Any]:
        """Generate structured content using the specified template."""
        try:
            chain = template | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "topic": topic,
                "framework_name": framework_name,
                "framework_info": json.dumps(framework_info, indent=2),
                "learning_level": learning_level,
                "learning_objectives": ", ".join(learning_objectives),
                "user_experience": user_context.get("experience_level", "beginner"),
                "preferred_learning_style": user_context.get("learning_style", "visual")
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in structured content generation: {str(e)}")
            raise
    
    async def _enhance_with_examples(
        self,
        content: Dict[str, Any],
        framework_name: str,
        learning_level: str
    ) -> Dict[str, Any]:
        """Enhance content with relevant examples."""
        try:
            # Add practical examples to each section
            enhanced_content = content.copy()
            
            if "sections" in content:
                for section in enhanced_content["sections"]:
                    if "examples" not in section:
                        examples = await self._generate_section_examples(
                            section.get("title", ""),
                            section.get("content", ""),
                            framework_name,
                            learning_level
                        )
                        section["examples"] = examples
            
            return enhanced_content
            
        except Exception as e:
            logger.error(f"Error enhancing content with examples: {str(e)}")
            return content
    
    async def _generate_section_examples(
        self,
        section_title: str,
        section_content: str,
        framework_name: str,
        learning_level: str
    ) -> List[Dict[str, str]]:
        """Generate examples for a specific section."""
        try:
            prompt = ChatPromptTemplate.from_template(
                """
                Generate practical examples for the following section about {framework_name}:
                
                Section Title: {section_title}
                Section Content: {section_content}
                Learning Level: {learning_level}
                
                Provide 2-3 practical examples that illustrate the concepts in this section.
                Each example should be relevant, clear, and appropriate for the learning level.
                
                Format as JSON:
                {{
                    "examples": [
                        {{
                            "title": "Example title",
                            "description": "Brief description",
                            "scenario": "Practical scenario or use case"
                        }}
                    ]
                }}
                """
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "section_title": section_title,
                "section_content": section_content,
                "framework_name": framework_name,
                "learning_level": learning_level
            })
            
            return result.get("examples", [])
            
        except Exception as e:
            logger.error(f"Error generating section examples: {str(e)}")
            return []
    
    async def _get_framework_context(self, framework_name: str) -> Dict[str, Any]:
        """Get framework-specific context information."""
        try:
            if self.framework_collector:
                return await self.framework_collector.get_framework_info(framework_name)
            return {"name": framework_name, "description": f"Framework: {framework_name}"}
        except Exception as e:
            logger.error(f"Error getting framework context: {str(e)}")
            return {"name": framework_name, "description": f"Framework: {framework_name}"}
    
    def _calculate_reading_time(self, content: Dict[str, Any]) -> int:
        """Calculate estimated reading time in minutes."""
        try:
            # Rough estimation: 200 words per minute reading speed
            total_text = json.dumps(content)
            word_count = len(total_text.split())
            reading_time = max(1, word_count // 200)
            return reading_time
        except Exception:
            return 5  # Default to 5 minutes
    
    def _get_beginner_template(self) -> ChatPromptTemplate:
        """Get template for beginner-level content."""
        return ChatPromptTemplate.from_template(
            """
            Generate comprehensive beginner-friendly theory content about "{topic}" in {framework_name}.
            
            Framework Information: {framework_info}
            Learning Objectives: {learning_objectives}
            User Experience: {user_experience}
            Learning Style: {preferred_learning_style}
            
            Create content that is:
            - Clear and easy to understand
            - Well-structured with logical flow
            - Includes practical context
            - Avoids complex jargon
            - Provides step-by-step explanations
            
            Format your response as JSON with this structure:
            {{
                "title": "Content title",
                "introduction": "Engaging introduction that sets context",
                "learning_objectives": ["Objective 1", "Objective 2"],
                "sections": [
                    {{
                        "title": "Section title",
                        "content": "Detailed content",
                        "key_points": ["Key point 1", "Key point 2"],
                        "difficulty": "beginner"
                    }}
                ],
                "summary": "Concise summary of key concepts",
                "next_steps": ["Suggested next learning steps"],
                "glossary": {{
                    "term1": "definition1",
                    "term2": "definition2"
                }}
            }}
            """
        )
    
    def _get_intermediate_template(self) -> ChatPromptTemplate:
        """Get template for intermediate-level content."""
        return ChatPromptTemplate.from_template(
            """
            Generate intermediate-level theory content about "{topic}" in {framework_name}.
            
            Framework Information: {framework_info}
            Learning Objectives: {learning_objectives}
            User Experience: {user_experience}
            Learning Style: {preferred_learning_style}
            
            Create content that:
            - Builds on foundational knowledge
            - Introduces more complex concepts
            - Provides detailed explanations
            - Includes comparative analysis
            - Shows real-world applications
            
            Format your response as JSON with this structure:
            {{
                "title": "Content title",
                "prerequisites": ["Required knowledge"],
                "introduction": "Introduction that connects to prior knowledge",
                "learning_objectives": ["Objective 1", "Objective 2"],
                "sections": [
                    {{
                        "title": "Section title",
                        "content": "Detailed content",
                        "key_points": ["Key point 1", "Key point 2"],
                        "difficulty": "intermediate",
                        "connections": ["Related concepts"]
                    }}
                ],
                "practical_applications": ["Application 1", "Application 2"],
                "summary": "Comprehensive summary",
                "advanced_topics": ["Topics for further study"],
                "glossary": {{
                    "term1": "definition1",
                    "term2": "definition2"
                }}
            }}
            """
        )
    
    def _get_advanced_template(self) -> ChatPromptTemplate:
        """Get template for advanced-level content."""
        return ChatPromptTemplate.from_template(
            """
            Generate advanced-level theory content about "{topic}" in {framework_name}.
            
            Framework Information: {framework_info}
            Learning Objectives: {learning_objectives}
            User Experience: {user_experience}
            Learning Style: {preferred_learning_style}
            
            Create content that:
            - Assumes strong foundational knowledge
            - Explores complex interactions and edge cases
            - Provides deep technical insights
            - Includes performance considerations
            - Shows advanced patterns and best practices
            
            Format your response as JSON with this structure:
            {{
                "title": "Content title",
                "prerequisites": ["Advanced prerequisites"],
                "introduction": "Technical introduction",
                "learning_objectives": ["Advanced objective 1", "Advanced objective 2"],
                "sections": [
                    {{
                        "title": "Section title",
                        "content": "Deep technical content",
                        "key_points": ["Advanced point 1", "Advanced point 2"],
                        "difficulty": "advanced",
                        "technical_details": ["Technical detail 1"],
                        "performance_notes": ["Performance consideration 1"]
                    }}
                ],
                "advanced_patterns": ["Pattern 1", "Pattern 2"],
                "best_practices": ["Best practice 1", "Best practice 2"],
                "edge_cases": ["Edge case 1", "Edge case 2"],
                "summary": "Technical summary",
                "research_directions": ["Advanced topic 1", "Advanced topic 2"],
                "glossary": {{
                    "advanced_term1": "technical_definition1",
                    "advanced_term2": "technical_definition2"
                }}
            }}
            """
        ) 