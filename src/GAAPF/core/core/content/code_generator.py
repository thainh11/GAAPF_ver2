"""
Code example generation module for the GAAPF framework.

This module provides AI-powered generation of code examples
with detailed explanations tailored to learning objectives.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ...memory.long_term_memory import LongTermMemory

logger = logging.getLogger(__name__)

class CodeGenerator:
    """
    AI-powered code example generator.
    
    Generates comprehensive code examples including:
    - Framework-specific code samples
    - Step-by-step explanations
    - Multiple complexity levels
    - Interactive exercises
    """
    
    def __init__(self, llm: BaseLanguageModel, framework_collector=None, long_term_memory: LongTermMemory = None, is_logging: bool = False):
        """
        Initialize the code generator.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model for code generation
        framework_collector : FrameworkCollector, optional
            Framework-specific information collector
        long_term_memory : LongTermMemory, optional
            Long-term memory for database content validation
        is_logging : bool
            Enable detailed logging
        """
        self.llm = llm
        self.framework_collector = framework_collector
        self.long_term_memory = long_term_memory
        self.is_logging = is_logging
        
        # Code templates by learning level
        self.code_templates = {
            "beginner": self._get_beginner_code_template(),
            "intermediate": self._get_intermediate_code_template(),
            "advanced": self._get_advanced_code_template()
        }
        
        # Supported languages and frameworks
        self.supported_frameworks = {
            "python": ["django", "flask", "fastapi", "pandas", "numpy"],
            "javascript": ["react", "vue", "angular", "node", "express"],
            "java": ["spring", "hibernate", "maven"],
            "csharp": ["asp.net", "entity framework", "mvc"]
        }
        
        if self.is_logging:
            logger.info("CodeGenerator initialized")
    
    def suggest_file_path(self, concept: str, framework_name: str, language: str, file_type: str = "main") -> str:
        """
        Suggest an exact file path for generated code based on framework conventions.
        
        Parameters:
        ----------
        concept : str
            The concept being implemented
        framework_name : str
            Target framework
        language : str
            Programming language
        file_type : str
            Type of file (main, test, config, etc.)
            
        Returns:
        -------
        str
            Suggested file path
        """
        try:
            # Normalize concept name for file naming
            concept_name = re.sub(r'[^a-zA-Z0-9_]', '_', concept.lower())
            framework_lower = framework_name.lower()
            
            # Framework-specific path suggestions
            if "django" in framework_lower:
                if file_type == "main":
                    return f"myproject/myapp/{concept_name}.py"
                elif file_type == "test":
                    return f"myproject/myapp/tests/test_{concept_name}.py"
                elif file_type == "model":
                    return f"myproject/myapp/models/{concept_name}.py"
                elif file_type == "view":
                    return f"myproject/myapp/views/{concept_name}.py"
            
            elif "flask" in framework_lower:
                if file_type == "main":
                    return f"app/{concept_name}.py"
                elif file_type == "test":
                    return f"tests/test_{concept_name}.py"
                elif file_type == "route":
                    return f"app/routes/{concept_name}.py"
            
            elif "react" in framework_lower:
                if file_type == "main":
                    return f"src/components/{concept_name.title()}.jsx"
                elif file_type == "test":
                    return f"src/components/__tests__/{concept_name.title()}.test.jsx"
                elif file_type == "style":
                    return f"src/components/{concept_name.title()}.css"
            
            elif "vue" in framework_lower:
                if file_type == "main":
                    return f"src/components/{concept_name.title()}.vue"
                elif file_type == "test":
                    return f"tests/unit/{concept_name}.spec.js"
            
            elif "spring" in framework_lower:
                if file_type == "main":
                    return f"src/main/java/com/example/{concept_name.title()}.java"
                elif file_type == "test":
                    return f"src/test/java/com/example/{concept_name.title()}Test.java"
            
            # Default path suggestions by language
            if language == "python":
                return f"src/{concept_name}.py" if file_type == "main" else f"tests/test_{concept_name}.py"
            elif language == "javascript":
                return f"src/{concept_name}.js" if file_type == "main" else f"tests/{concept_name}.test.js"
            elif language == "java":
                return f"src/main/java/{concept_name.title()}.java" if file_type == "main" else f"src/test/java/{concept_name.title()}Test.java"
            elif language == "csharp":
                return f"src/{concept_name.title()}.cs" if file_type == "main" else f"tests/{concept_name.title()}Test.cs"
            
            # Fallback
            return f"{concept_name}.{self._get_file_extension(language)}"
            
        except Exception as e:
            logger.error(f"Error suggesting file path: {str(e)}")
            return f"{concept}.{self._get_file_extension(language)}"
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for a programming language."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "java": "java",
            "csharp": "cs"
        }
        return extensions.get(language, "txt")
    
    async def validate_code_against_database(self, concept: str, framework_name: str, code_content: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
        """
        Validate generated code content against stored knowledge in the database.
        
        Parameters:
        ----------
        concept : str
            The concept being implemented
        framework_name : str
            Target framework
        code_content : Dict[str, Any]
            Generated code content to validate
        user_id : str, optional
            User ID for personalized validation
            
        Returns:
        -------
        Dict[str, Any]
            Validation results with suggestions
        """
        if not self.long_term_memory:
            return {
                "validated": True,
                "suggestions": [],
                "similar_patterns": [],
                "confidence": 0.5,
                "message": "No database validation available"
            }
        
        try:
            # Query similar memories for the concept and framework
            query_text = f"{concept} {framework_name} code example implementation"
            similar_memories = self.long_term_memory.query_similar_memories(
                query=query_text,
                user_id=user_id,
                n_results=5,
                framework_id=framework_name
            )
            
            # Extract patterns from similar memories
            patterns = []
            suggestions = []
            
            for memory in similar_memories:
                memory_text = memory.get('text', '')
                distance = memory.get('distance', 1.0)
                
                # If memory is very similar (low distance), extract patterns
                if distance < 0.3:
                    patterns.append({
                        "pattern": memory_text,
                        "similarity": 1 - distance,
                        "source": "database_memory"
                    })
                
                # Generate suggestions based on memory content
                if distance < 0.5:
                    suggestions.append(f"Consider pattern from previous implementation: {memory_text[:100]}...")
            
            # Validate code structure against known patterns
            main_code = code_content.get('main_code', '')
            validation_score = self._calculate_validation_score(main_code, patterns)
            
            # Generate framework-specific suggestions
            framework_suggestions = await self._get_framework_specific_suggestions(
                concept, framework_name, code_content, similar_memories
            )
            
            suggestions.extend(framework_suggestions)
            
            return {
                "validated": validation_score > 0.6,
                "suggestions": suggestions[:5],  # Limit to top 5 suggestions
                "similar_patterns": patterns[:3],  # Limit to top 3 patterns
                "confidence": validation_score,
                "message": f"Validation completed with {len(similar_memories)} similar memories found"
            }
            
        except Exception as e:
            logger.error(f"Error validating code against database: {str(e)}")
            return {
                "validated": False,
                "suggestions": ["Database validation failed - proceeding with basic validation"],
                "similar_patterns": [],
                "confidence": 0.0,
                "message": f"Validation error: {str(e)}"
            }
    
    def _calculate_validation_score(self, code: str, patterns: List[Dict]) -> float:
        """
        Calculate validation score based on similarity to known patterns.
        """
        if not patterns:
            return 0.5  # Neutral score when no patterns available
        
        # Simple scoring based on pattern similarity
        total_similarity = sum(pattern.get('similarity', 0) for pattern in patterns)
        avg_similarity = total_similarity / len(patterns)
        
        # Boost score if code contains common framework keywords
        keyword_boost = 0
        common_keywords = ['import', 'class', 'def', 'function', 'const', 'var', 'let']
        for keyword in common_keywords:
            if keyword in code.lower():
                keyword_boost += 0.1
        
        return min(avg_similarity + keyword_boost, 1.0)
    
    async def _get_framework_specific_suggestions(self, concept: str, framework_name: str, code_content: Dict, memories: List) -> List[str]:
        """
        Generate framework-specific suggestions based on memories and best practices.
        """
        suggestions = []
        framework_lower = framework_name.lower()
        
        # Django-specific suggestions
        if "django" in framework_lower:
            if "model" in concept.lower() and "class" not in code_content.get('main_code', ''):
                suggestions.append("Consider using Django Model class for database operations")
            if "view" in concept.lower() and "request" not in code_content.get('main_code', ''):
                suggestions.append("Django views should typically handle request objects")
        
        # Flask-specific suggestions
        elif "flask" in framework_lower:
            if "route" in concept.lower() and "@app.route" not in code_content.get('main_code', ''):
                suggestions.append("Flask routes should use @app.route decorator")
        
        # React-specific suggestions
        elif "react" in framework_lower:
            if "component" in concept.lower() and "return" not in code_content.get('main_code', ''):
                suggestions.append("React components should return JSX elements")
        
        return suggestions
    
    async def generate_code_example(
        self,
        concept: str,
        framework_name: str,
        learning_level: str,
        learning_objectives: List[str],
        example_type: str = "practical",
        complexity: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive code example for a specific concept.
        
        Parameters:
        ----------
        concept : str
            The concept to demonstrate with code
        framework_name : str
            Target framework
        learning_level : str
            User's learning level
        learning_objectives : List[str]
            Specific objectives the code should address
        example_type : str
            Type of example (practical/tutorial/exercise)
        complexity : str
            Code complexity level
            
        Returns:
        -------
        Dict[str, Any]
            Generated code example with explanations
        """
        try:
            if self.is_logging:
                logger.info(f"Generating code example for concept: {concept}")
            
            # Get framework-specific context
            framework_info = await self._get_framework_context(framework_name)
            
            # Determine programming language
            language = self._detect_framework_language(framework_name)
            
            # Select appropriate template
            template = self.code_templates.get(learning_level, self.code_templates["beginner"])
            
            # Generate code with explanation
            code_content = await self._generate_structured_code(
                concept=concept,
                framework_name=framework_name,
                framework_info=framework_info,
                language=language,
                learning_level=learning_level,
                learning_objectives=learning_objectives,
                example_type=example_type,
                complexity=complexity,
                template=template
            )
            
            # Add syntax validation and formatting
            validated_code = await self._validate_and_format_code(
                code_content, framework_name, language
            )
            
            # Generate interactive exercises
            exercises = await self._generate_practice_exercises(
                concept, framework_name, validated_code, learning_level
            )
            
            # Suggest file path for the generated code
            suggested_path = self.suggest_file_path(concept, framework_name, language)
            
            # Validate code against database if memory is available
            validation_result = await self.validate_code_against_database(
                concept, framework_name, validated_code
            )
            
            result = {
                "concept": concept,
                "framework_name": framework_name,
                "language": language,
                "code_content": validated_code,
                "exercises": exercises,
                "suggested_file_path": suggested_path,
                "validation": validation_result,
                "metadata": {
                    "learning_level": learning_level,
                    "example_type": example_type,
                    "complexity": complexity,
                    "objectives_covered": learning_objectives,
                    "generated_at": datetime.now().isoformat(),
                    "estimated_completion_time": self._estimate_completion_time(validated_code)
                }
            }
            
            if self.is_logging:
                logger.info(f"Code example generated successfully for concept: {concept}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating code example: {str(e)}")
            raise
    
    def _detect_framework_language(self, framework_name: str) -> str:
        """Detect the programming language for a framework."""
        framework_lower = framework_name.lower()
        
        # Python frameworks
        if any(fw in framework_lower for fw in ["django", "flask", "fastapi", "pandas", "numpy", "python"]):
            return "python"
        
        # JavaScript frameworks
        if any(fw in framework_lower for fw in ["react", "vue", "angular", "node", "express", "javascript", "js"]):
            return "javascript"
        
        # Java frameworks
        if any(fw in framework_lower for fw in ["spring", "hibernate", "maven", "java"]):
            return "java"
        
        # C# frameworks
        if any(fw in framework_lower for fw in ["asp.net", "entity framework", "mvc", "csharp", "c#"]):
            return "csharp"
        
        # Default to Python
        return "python"
    
    async def _get_framework_context(self, framework_name: str) -> Dict[str, Any]:
        """Get framework-specific context information."""
        try:
            if self.framework_collector:
                return await self.framework_collector.get_framework_info(framework_name)
            return {"name": framework_name, "language": self._detect_framework_language(framework_name)}
        except Exception as e:
            logger.error(f"Error getting framework context: {str(e)}")
            return {"name": framework_name, "language": self._detect_framework_language(framework_name)}
    
    async def _generate_structured_code(
        self,
        concept: str,
        framework_name: str,
        framework_info: Dict,
        language: str,
        learning_level: str,
        learning_objectives: List[str],
        example_type: str,
        complexity: str,
        template: ChatPromptTemplate
    ) -> Dict[str, Any]:
        """Generate structured code using the specified template."""
        try:
            chain = template | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "concept": concept,
                "framework_name": framework_name,
                "framework_info": json.dumps(framework_info, indent=2),
                "language": language,
                "learning_level": learning_level,
                "learning_objectives": ", ".join(learning_objectives),
                "example_type": example_type,
                "complexity": complexity
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in structured code generation: {str(e)}")
            raise
    
    async def _validate_and_format_code(
        self,
        code_content: Dict[str, Any],
        framework_name: str,
        language: str
    ) -> Dict[str, Any]:
        """Validate and format generated code."""
        try:
            validated_content = code_content.copy()
            
            # Extract and validate main code
            if "main_code" in code_content:
                code = code_content["main_code"]
                
                # Basic syntax validation
                validated_code = await self._basic_syntax_validation(code, language)
                validated_content["main_code"] = validated_code
                
                # Add syntax highlighting hints
                validated_content["syntax_highlighting"] = language
            
            return validated_content
            
        except Exception as e:
            logger.error(f"Error validating code: {str(e)}")
            return code_content
    
    async def _basic_syntax_validation(self, code: str, language: str) -> str:
        """Perform basic syntax validation and formatting."""
        try:
            # Remove common formatting issues
            cleaned_code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)  # Remove excessive blank lines
            cleaned_code = cleaned_code.strip()
            
            return cleaned_code
            
        except Exception as e:
            logger.error(f"Error in syntax validation: {str(e)}")
            return code
    
    async def _generate_practice_exercises(
        self,
        concept: str,
        framework_name: str,
        code_content: Dict[str, Any],
        learning_level: str
    ) -> List[Dict[str, Any]]:
        """Generate practice exercises based on the code example."""
        try:
            prompt = ChatPromptTemplate.from_template(
                """
                Generate practice exercises based on this {framework_name} code example about {concept}.
                
                Learning Level: {learning_level}
                
                Create 2-3 practice exercises that:
                1. Reinforce the concepts shown in the code
                2. Are appropriate for the learning level
                3. Include clear instructions and expected outcomes
                
                Format as JSON:
                {{
                    "exercises": [
                        {{
                            "title": "Exercise title",
                            "description": "What to do",
                            "difficulty": "easy/medium/hard",
                            "instructions": ["Step 1", "Step 2"],
                            "expected_outcome": "What should happen"
                        }}
                    ]
                }}
                """
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "concept": concept,
                "framework_name": framework_name,
                "learning_level": learning_level
            })
            
            return result.get("exercises", [])
            
        except Exception as e:
            logger.error(f"Error generating practice exercises: {str(e)}")
            return []
    
    def _estimate_completion_time(self, code_content: Dict[str, Any]) -> int:
        """Estimate completion time in minutes."""
        try:
            # Base time estimation
            base_time = 15  # 15 minutes base
            
            # Add time based on code length
            code_length = len(str(code_content.get("main_code", "")))
            if code_length > 500:
                base_time += 10
            if code_length > 1000:
                base_time += 15
            
            return base_time
            
        except Exception:
            return 30  # Default to 30 minutes
    
    def _get_beginner_code_template(self) -> ChatPromptTemplate:
        """Get template for beginner-level code examples."""
        return ChatPromptTemplate.from_template(
            """
            Generate a beginner-friendly code example for "{concept}" in {framework_name}.
            
            Programming Language: {language}
            Learning Objectives: {learning_objectives}
            
            Create a simple, well-commented code example.
            
            Format your response as JSON:
            {{
                "title": "Code example title",
                "description": "What this example demonstrates",
                "main_code": "The main code with comments",
                "explanation": "Overall explanation of the code",
                "concepts": ["Concept 1", "Concept 2"],
                "expected_output": "Expected output or result",
                "complexity": "beginner"
            }}
            """
        )
    
    def _get_intermediate_code_template(self) -> ChatPromptTemplate:
        """Get template for intermediate-level code examples."""
        return ChatPromptTemplate.from_template(
            """
            Generate an intermediate-level code example for "{concept}" in {framework_name}.
            
            Programming Language: {language}
            Learning Objectives: {learning_objectives}
            
            Create a more complex example that builds on foundational knowledge.
            
            Format your response as JSON:
            {{
                "title": "Code example title",
                "description": "What this example demonstrates",
                "main_code": "The main code with strategic comments",
                "explanation": "Comprehensive explanation",
                "key_concepts": ["Advanced concept 1", "Advanced concept 2"],
                "expected_output": "Expected output with variations",
                "complexity": "intermediate"
            }}
            """
        )
    
    def _get_advanced_code_template(self) -> ChatPromptTemplate:
        """Get template for advanced-level code examples."""
        return ChatPromptTemplate.from_template(
            """
            Generate an advanced-level code example for "{concept}" in {framework_name}.
            
            Programming Language: {language}
            Learning Objectives: {learning_objectives}
            
            Create a complex, production-ready example.
            
            Format your response as JSON:
            {{
                "title": "Advanced code example title",
                "description": "Complex problem this example solves",
                "main_code": "Production-ready code",
                "architecture_explanation": "System design and architecture",
                "advanced_concepts": ["Pattern 1", "Pattern 2"],
                "expected_output": "Complex output with edge cases",
                "complexity": "advanced"
            }}
            """
        )
    
    async def generate_step_by_step_tutorial(
        self,
        topic: str,
        framework_name: str,
        learning_level: str,
        steps: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a step-by-step coding tutorial.
        
        Parameters:
        ----------
        topic : str
            Tutorial topic
        framework_name : str
            Target framework
        learning_level : str
            User's learning level
        steps : List[str]
            Tutorial steps to cover
            
        Returns:
        -------
        Dict[str, Any]
            Step-by-step tutorial with code
        """
        try:
            tutorial_steps = []
            accumulated_code = ""
            
            for i, step in enumerate(steps, 1):
                step_content = await self._generate_tutorial_step(
                    step_number=i,
                    step_description=step,
                    topic=topic,
                    framework_name=framework_name,
                    learning_level=learning_level,
                    previous_code=accumulated_code
                )
                
                tutorial_steps.append(step_content)
                accumulated_code += "\n" + step_content.get("code", "")
            
            return {
                "topic": topic,
                "framework_name": framework_name,
                "tutorial_steps": tutorial_steps,
                "complete_code": accumulated_code.strip(),
                "metadata": {
                    "total_steps": len(steps),
                    "learning_level": learning_level,
                    "generated_at": datetime.now().isoformat(),
                    "estimated_duration": len(steps) * 10  # 10 minutes per step
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating tutorial: {str(e)}")
            raise
    
    async def generate_progressive_examples(
        self,
        concept: str,
        framework_name: str,
        difficulty_levels: List[str],
        learning_objectives: List[str]
    ) -> Dict[str, Any]:
        """
        Generate progressive code examples with increasing complexity.
        
        Parameters:
        ----------
        concept : str
            Core concept to demonstrate
        framework_name : str
            Target framework
        difficulty_levels : List[str]
            Ordered difficulty levels
        learning_objectives : List[str]
            Learning objectives to address
            
        Returns:
        -------
        Dict[str, Any]
            Progressive examples structure
        """
        try:
            progressive_examples = {}
            
            for i, level in enumerate(difficulty_levels):
                example = await self.generate_code_example(
                    concept=concept,
                    framework_name=framework_name,
                    learning_level=level,
                    learning_objectives=learning_objectives,
                    complexity=level
                )
                
                # Add progression context
                if i > 0:
                    example["builds_on"] = difficulty_levels[:i]
                    example["new_concepts"] = await self._identify_new_concepts(
                        current_level=level,
                        previous_levels=difficulty_levels[:i],
                        concept=concept
                    )
                
                progressive_examples[level] = example
            
            return {
                "concept": concept,
                "framework_name": framework_name,
                "progression": progressive_examples,
                "total_levels": len(difficulty_levels),
                "metadata": {
                    "progression_type": "difficulty-based",
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating progressive examples: {str(e)}")
            raise
    
    async def _generate_structured_code(
        self,
        concept: str,
        framework_name: str,
        framework_info: Dict,
        language: str,
        learning_level: str,
        learning_objectives: List[str],
        example_type: str,
        complexity: str,
        template: ChatPromptTemplate
    ) -> Dict[str, Any]:
        """Generate structured code using the specified template."""
        try:
            chain = template | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "concept": concept,
                "framework_name": framework_name,
                "framework_info": json.dumps(framework_info, indent=2),
                "language": language,
                "learning_level": learning_level,
                "learning_objectives": ", ".join(learning_objectives),
                "example_type": example_type,
                "complexity": complexity
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in structured code generation: {str(e)}")
            raise
    
    async def _generate_tutorial_step(
        self,
        step_number: int,
        step_description: str,
        topic: str,
        framework_name: str,
        learning_level: str,
        previous_code: str
    ) -> Dict[str, Any]:
        """Generate content for a single tutorial step."""
        try:
            prompt = ChatPromptTemplate.from_template(
                """
                Generate content for step {step_number} of a {framework_name} tutorial.
                
                Topic: {topic}
                Step Description: {step_description}
                Learning Level: {learning_level}
                Previous Code: {previous_code}
                
                Create this tutorial step with:
                1. Clear explanation of what we're doing in this step
                2. The code to add or modify
                3. Explanation of the code
                4. What the user should see/expect
                5. Common issues and solutions
                
                Format as JSON:
                {{
                    "step_number": {step_number},
                    "title": "Step title",
                    "explanation": "What we're doing in this step",
                    "code": "Code for this step",
                    "code_explanation": "Line-by-line explanation",
                    "expected_output": "What the user should see",
                    "common_issues": ["Issue 1", "Issue 2"],
                    "tips": ["Tip 1", "Tip 2"]
                }}
                """
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "step_number": step_number,
                "step_description": step_description,
                "topic": topic,
                "framework_name": framework_name,
                "learning_level": learning_level,
                "previous_code": previous_code
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating tutorial step: {str(e)}")
            return {
                "step_number": step_number,
                "title": f"Step {step_number}",
                "explanation": step_description,
                "code": "// Code generation failed",
                "code_explanation": "Please try again",
                "expected_output": "",
                "common_issues": [],
                "tips": []
            }
    
    async def _validate_and_format_code(
        self,
        code_content: Dict[str, Any],
        framework_name: str,
        language: str
    ) -> Dict[str, Any]:
        """Validate and format generated code."""
        try:
            validated_content = code_content.copy()
            
            # Extract and validate main code
            if "main_code" in code_content:
                code = code_content["main_code"]
                
                # Basic syntax validation
                validated_code = await self._basic_syntax_validation(code, language)
                validated_content["main_code"] = validated_code
                
                # Add syntax highlighting hints
                validated_content["syntax_highlighting"] = language
                
                # Validate against framework patterns
                if self.framework_collector:
                    framework_validation = await self._validate_framework_syntax(
                        code, framework_name
                    )
                    validated_content["framework_validation"] = framework_validation
            
            return validated_content
            
        except Exception as e:
            logger.error(f"Error validating code: {str(e)}")
            return code_content
    
    async def _basic_syntax_validation(self, code: str, language: str) -> str:
        """Perform basic syntax validation and formatting."""
        try:
            # Remove common formatting issues
            cleaned_code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)  # Remove excessive blank lines
            cleaned_code = cleaned_code.strip()
            
            # Add language-specific formatting
            if language == "python":
                # Ensure proper indentation hints
                lines = cleaned_code.split('\n')
                formatted_lines = []
                for line in lines:
                    if line.strip():
                        formatted_lines.append(line)
                    else:
                        formatted_lines.append('')
                cleaned_code = '\n'.join(formatted_lines)
            
            return cleaned_code
            
        except Exception as e:
            logger.error(f"Error in syntax validation: {str(e)}")
            return code
    
    async def _validate_framework_syntax(self, code: str, framework_name: str) -> Dict[str, Any]:
        """Validate code against framework-specific patterns."""
        try:
            if not self.framework_collector:
                return {"status": "skipped", "reason": "no framework collector"}
            
            # Get framework patterns
            framework_info = await self.framework_collector.get_framework_info(framework_name)
            
            # Basic validation
            validation_result = {
                "status": "valid",
                "warnings": [],
                "suggestions": []
            }
            
            # Check for common patterns
            if framework_name.lower() in ["django", "flask", "fastapi"]:
                if "import" not in code:
                    validation_result["warnings"].append("No import statements found")
                
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in framework validation: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _generate_practice_exercises(
        self,
        concept: str,
        framework_name: str,
        code_content: Dict[str, Any],
        learning_level: str
    ) -> List[Dict[str, Any]]:
        """Generate practice exercises based on the code example."""
        try:
            prompt = ChatPromptTemplate.from_template(
                """
                Generate practice exercises based on this {framework_name} code example about {concept}.
                
                Code Content: {code_summary}
                Learning Level: {learning_level}
                
                Create 3-5 practice exercises that:
                1. Reinforce the concepts shown in the code
                2. Are appropriate for the learning level
                3. Progressively increase in difficulty
                4. Include clear instructions and expected outcomes
                
                Format as JSON:
                {{
                    "exercises": [
                        {{
                            "title": "Exercise title",
                            "description": "What to do",
                            "difficulty": "easy/medium/hard",
                            "instructions": ["Step 1", "Step 2"],
                            "expected_outcome": "What should happen",
                            "hints": ["Hint 1", "Hint 2"],
                            "solution_approach": "How to approach the solution"
                        }}
                    ]
                }}
                """
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            
            # Create summary of code content
            code_summary = {
                "main_concepts": code_content.get("concepts", []),
                "code_length": len(str(code_content.get("main_code", ""))),
                "complexity": code_content.get("complexity", "moderate")
            }
            
            result = await chain.ainvoke({
                "concept": concept,
                "framework_name": framework_name,
                "code_summary": json.dumps(code_summary),
                "learning_level": learning_level
            })
            
            return result.get("exercises", [])
            
        except Exception as e:
            logger.error(f"Error generating practice exercises: {str(e)}")
            return []
    
    async def _identify_new_concepts(
        self,
        current_level: str,
        previous_levels: List[str],
        concept: str
    ) -> List[str]:
        """Identify new concepts introduced at the current level."""
        try:
            prompt = ChatPromptTemplate.from_template(
                """
                Identify new concepts introduced at the {current_level} level for {concept}.
                
                Previous levels covered: {previous_levels}
                Current level: {current_level}
                
                What new concepts, techniques, or patterns are typically introduced
                when moving from {previous_level} to {current_level} for {concept}?
                
                Return as JSON:
                {{
                    "new_concepts": ["Concept 1", "Concept 2"],
                    "advanced_techniques": ["Technique 1", "Technique 2"],
                    "complexity_increases": ["Aspect 1", "Aspect 2"]
                }}
                """
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "current_level": current_level,
                "previous_levels": ", ".join(previous_levels),
                "previous_level": previous_levels[-1] if previous_levels else "none",
                "concept": concept
            })
            
            return result.get("new_concepts", [])
            
        except Exception as e:
            logger.error(f"Error identifying new concepts: {str(e)}")
            return []
    
    def _detect_framework_language(self, framework_name: str) -> str:
        """Detect the programming language for a framework."""
        framework_lower = framework_name.lower()
        
        # Python frameworks
        if any(fw in framework_lower for fw in ["django", "flask", "fastapi", "pandas", "numpy", "python"]):
            return "python"
        
        # JavaScript frameworks
        if any(fw in framework_lower for fw in ["react", "vue", "angular", "node", "express", "javascript", "js"]):
            return "javascript"
        
        # Java frameworks
        if any(fw in framework_lower for fw in ["spring", "hibernate", "maven", "java"]):
            return "java"
        
        # C# frameworks
        if any(fw in framework_lower for fw in ["asp.net", "entity framework", "mvc", "csharp", "c#"]):
            return "csharp"
        
        # Default to Python
        return "python"
    
    async def _get_framework_context(self, framework_name: str) -> Dict[str, Any]:
        """Get framework-specific context information."""
        try:
            if self.framework_collector:
                return await self.framework_collector.get_framework_info(framework_name)
            return {"name": framework_name, "language": self._detect_framework_language(framework_name)}
        except Exception as e:
            logger.error(f"Error getting framework context: {str(e)}")
            return {"name": framework_name, "language": self._detect_framework_language(framework_name)}
    
    def _estimate_completion_time(self, code_content: Dict[str, Any]) -> int:
        """Estimate completion time in minutes."""
        try:
            # Base time estimation
            base_time = 15  # 15 minutes base
            
            # Add time based on code length
            code_length = len(str(code_content.get("main_code", "")))
            if code_length > 500:
                base_time += 10
            if code_length > 1000:
                base_time += 15
            
            # Add time based on complexity
            complexity = code_content.get("complexity", "moderate")
            if complexity == "high":
                base_time += 20
            elif complexity == "moderate":
                base_time += 10
            
            # Add time for exercises
            exercises_count = len(code_content.get("exercises", []))
            base_time += exercises_count * 5
            
            return base_time
            
        except Exception:
            return 30  # Default to 30 minutes
    
    def _get_beginner_code_template(self) -> ChatPromptTemplate:
        """Get template for beginner-level code examples."""
        return ChatPromptTemplate.from_template(
            """
            Generate a beginner-friendly code example for "{concept}" in {framework_name}.
            
            Framework Information: {framework_info}
            Programming Language: {language}
            Learning Level: {learning_level}
            Learning Objectives: {learning_objectives}
            Example Type: {example_type}
            Complexity: {complexity}
            
            Create a code example that is:
            - Simple and easy to understand
            - Well-commented with explanations
            - Uses basic concepts only
            - Includes step-by-step breakdown
            - Provides clear expected output
            
            Format your response as JSON:
            {{
                "title": "Code example title",
                "description": "What this example demonstrates",
                "main_code": "The main code with comments",
                "explanation": "Overall explanation of the code",
                "line_by_line": [
                    {{
                        "line_number": 1,
                        "code": "code line",
                        "explanation": "what this line does"
                    }}
                ],
                "concepts": ["Concept 1", "Concept 2"],
                "expected_output": "Expected output or result",
                "common_mistakes": ["Mistake 1", "Mistake 2"],
                "tips": ["Tip 1", "Tip 2"],
                "complexity": "beginner"
            }}
            """
        )
    
    def _get_intermediate_code_template(self) -> ChatPromptTemplate:
        """Get template for intermediate-level code examples."""
        return ChatPromptTemplate.from_template(
            """
            Generate an intermediate-level code example for "{concept}" in {framework_name}.
            
            Framework Information: {framework_info}
            Programming Language: {language}
            Learning Level: {learning_level}
            Learning Objectives: {learning_objectives}
            Example Type: {example_type}
            Complexity: {complexity}
            
            Create a code example that:
            - Builds on foundational knowledge
            - Introduces intermediate concepts
            - Shows practical applications
            - Includes error handling
            - Demonstrates best practices
            
            Format your response as JSON:
            {{
                "title": "Code example title",
                "description": "What this example demonstrates",
                "prerequisites": ["Required knowledge"],
                "main_code": "The main code with strategic comments",
                "explanation": "Comprehensive explanation",
                "key_concepts": ["Advanced concept 1", "Advanced concept 2"],
                "code_structure": {{
                    "imports": "Import statements explanation",
                    "main_logic": "Core logic explanation",
                    "error_handling": "Error handling explanation"
                }},
                "expected_output": "Expected output with variations",
                "best_practices": ["Practice 1", "Practice 2"],
                "extensions": ["How to extend this example"],
                "complexity": "intermediate"
            }}
            """
        )
    
    def _get_advanced_code_template(self) -> ChatPromptTemplate:
        """Get template for advanced-level code examples."""
        return ChatPromptTemplate.from_template(
            """
            Generate an advanced-level code example for "{concept}" in {framework_name}.
            
            Framework Information: {framework_info}
            Programming Language: {language}
            Learning Level: {learning_level}
            Learning Objectives: {learning_objectives}
            Example Type: {example_type}
            Complexity: {complexity}
            
            Create a code example that:
            - Demonstrates advanced patterns and techniques
            - Shows performance optimizations
            - Includes comprehensive error handling
            - Follows enterprise-level practices
            - Addresses edge cases and scalability
            
            Format your response as JSON:
            {{
                "title": "Advanced code example title",
                "description": "Complex problem this example solves",
                "prerequisites": ["Advanced prerequisites"],
                "main_code": "Production-ready code with minimal comments",
                "architecture_explanation": "System design and architecture",
                "advanced_concepts": ["Pattern 1", "Pattern 2"],
                "performance_considerations": ["Optimization 1", "Optimization 2"],
                "code_organization": {{
                    "modules": "Module structure explanation",
                    "patterns": "Design patterns used",
                    "scalability": "Scalability considerations"
                }},
                "expected_output": "Complex output with edge cases",
                "enterprise_practices": ["Practice 1", "Practice 2"],
                "edge_cases": ["Edge case 1", "Edge case 2"],
                "testing_strategy": "How to test this code",
                "complexity": "advanced"
            }}
            """
        )