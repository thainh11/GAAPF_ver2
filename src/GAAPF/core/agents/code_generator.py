"""
Code Generation Agent - GAAPF Phase 2
Specialized agent for generating and validating Python code.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import tempfile
import os

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import BaseTool

from . import SpecializedAgent
from ..core.content.code_generator import CodeGenerator
from ..tools.sandbox_runner import SandboxRunner

# Setup logging
logger = logging.getLogger(__name__)


class CodeGenerationAgent(SpecializedAgent):
    """
    Specialized agent for generating and validating Python code.
    
    This agent combines LLM-based code generation with automated validation
    through testing and linting in a sandbox environment.
    """
    
    # Class attributes for agent registry
    DESCRIPTION = "AI agent specialized in generating and validating Python code"
    CAPABILITIES = ["generate_code", "validate_code", "create_tests", "code_quality"]
    PRIORITY = 5  # Higher priority for code generation tasks
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[Union[str, BaseTool]] = [],
        memory_path: Optional[Path] = None,
        config: Dict = None,
        is_logging: bool = False,
        max_retries: int = 3,
        quality_threshold: int = 70,
        *args, **kwargs
    ):
        """
        Initialize CodeGenerationAgent.
        
        Args:
            llm: Language model for code generation
            tools: Additional tools (inherited from base)
            memory_path: Path for agent memory
            config: Agent configuration
            is_logging: Enable detailed logging
            max_retries: Maximum self-correction attempts
            quality_threshold: Minimum quality score to accept
        """
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            config=config,
            agent_type="code_generator",
            description=self.DESCRIPTION,
            is_logging=is_logging,
            *args, **kwargs
        )
        
        # Code generation specific attributes
        self.max_retries = max_retries
        self.quality_threshold = quality_threshold
        self.sandbox = SandboxRunner(timeout=15, memory_limit_mb=512)
        
        # Initialize the existing CodeGenerator for content creation
        self.content_generator = CodeGenerator(
            llm=llm,
            is_logging=is_logging
        )
        
        # Setup JSON parser for structured output
        self.json_parser = JsonOutputParser()
        
        if self.is_logging:
            logger.info(f"CodeGenerationAgent initialized with quality_threshold={quality_threshold}")
    
    def _generate_system_prompt(self, learning_context: Optional[Dict] = None) -> str:
        """Generate system prompt for code generation tasks."""
        return """You are a Python Code Generation Agent specialized in creating high-quality, tested code.

**Your Core Responsibilities:**
1. Generate clean, well-documented Python code based on specifications
2. Create comprehensive unit tests for generated code
3. Follow Python best practices and PEP 8 style guidelines
4. Ensure code is secure and handles edge cases appropriately

**Code Generation Guidelines:**
- Write clear, readable code with meaningful variable names
- Include proper docstrings for functions and classes
- Add type hints where appropriate
- Handle errors gracefully with try-catch blocks
- Generate accompanying unit tests using pytest

**Quality Standards:**
- All generated code must be syntactically correct
- Tests should cover main functionality and edge cases
- Code should be secure (no eval, exec, or dangerous operations)
- Follow the principle of least privilege

**Output Format:**
Always return your response as JSON with the following structure:
{
    "code": "# Your generated Python code here",
    "test_code": "# Corresponding pytest test code",
    "file_name": "suggested_filename.py",
    "test_file_name": "test_suggested_filename.py",
    "explanation": "Brief explanation of the implementation",
    "dependencies": ["list", "of", "required", "packages"]
}

Be precise, secure, and always generate working code."""
    
    async def generate_code(self, specification: str, context: Dict = None) -> Dict[str, Any]:
        """
        Generate Python code based on specification with validation.
        
        Args:
            specification: Natural language description of desired code
            context: Additional context (framework, user_level, etc.)
            
        Returns:
            Dict containing generated code, tests, and quality metrics
        """
        if self.is_logging:
            logger.info(f"Generating code for: {specification}")
        
        context = context or {}
        
        # Prepare the prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._generate_system_prompt()),
            ("human", """Generate Python code for the following specification:

Specification: {specification}

Context:
- Framework: {framework}
- User Level: {user_level}
- Additional Requirements: {additional_requirements}

Please generate high-quality Python code with comprehensive tests.""")
        ])
        
        # Extract context with defaults
        framework = context.get("framework", "general")
        user_level = context.get("user_level", "intermediate")
        additional_requirements = context.get("additional_requirements", "None")
        
        # Generate initial code
        attempt = 0
        best_result = None
        best_quality = 0
        
        while attempt < self.max_retries:
            try:
                if self.is_logging:
                    logger.info(f"Code generation attempt {attempt + 1}")
                
                # Generate code using LLM
                chain = prompt_template | self.llm | self.json_parser
                
                llm_result = await chain.ainvoke({
                    "specification": specification,
                    "framework": framework,
                    "user_level": user_level,
                    "additional_requirements": additional_requirements
                })
                
                # Validate the generated code
                validation_result = await self._validate_generated_code(llm_result)
                
                current_quality = validation_result["quality_score"]
                
                if current_quality >= self.quality_threshold:
                    # Quality threshold met, return result
                    result = {
                        **llm_result,
                        **validation_result,
                        "generation_attempts": attempt + 1,
                        "specification": specification
                    }
                    
                    if self.is_logging:
                        logger.info(f"Code generation successful with quality score: {current_quality}")
                    
                    return result
                
                # Track best result so far
                if current_quality > best_quality:
                    best_quality = current_quality
                    best_result = {
                        **llm_result,
                        **validation_result,
                        "generation_attempts": attempt + 1,
                        "specification": specification
                    }
                
                # If not good enough, prepare for retry
                if attempt < self.max_retries - 1:
                    error_feedback = self._prepare_retry_feedback(validation_result)
                    specification = f"{specification}\n\nPrevious attempt had issues: {error_feedback}\nPlease fix these issues."
                
                attempt += 1
                
            except Exception as e:
                logger.error(f"Code generation attempt {attempt + 1} failed: {str(e)}")
                attempt += 1
                
                if attempt >= self.max_retries:
                    # Return best result or error
                    if best_result:
                        best_result["generation_error"] = f"Max retries reached. Best quality: {best_quality}"
                        return best_result
                    else:
                        return {
                            "code": "# Code generation failed",
                            "test_code": "# No tests generated",
                            "file_name": "failed.py",
                            "test_file_name": "test_failed.py",
                            "explanation": f"Code generation failed: {str(e)}",
                            "dependencies": [],
                            "tests_passed": False,
                            "quality_score": 0,
                            "lint_errors": [str(e)],
                            "generation_attempts": attempt,
                            "generation_error": str(e)
                        }
        
        # Return best result if we reach here
        if best_result:
            best_result["generation_error"] = f"Quality threshold not met. Best score: {best_quality}"
            return best_result
        
        # Fallback error result
        return {
            "code": "# Code generation failed",
            "test_code": "# No tests generated",
            "file_name": "failed.py",
            "test_file_name": "test_failed.py",
            "explanation": "Code generation failed after all attempts",
            "dependencies": [],
            "tests_passed": False,
            "quality_score": 0,
            "lint_errors": ["Generation failed"],
            "generation_attempts": self.max_retries,
            "generation_error": "All generation attempts failed"
        }
    
    async def _validate_generated_code(self, llm_result: Dict) -> Dict[str, Any]:
        """
        Validate generated code by running tests and linting.
        
        Args:
            llm_result: Result from LLM containing code and tests
            
        Returns:
            Dict with validation results and quality metrics
        """
        try:
            # Create temporary files for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write main code file
                code_file = temp_path / llm_result.get("file_name", "generated.py")
                code_file.write_text(llm_result.get("code", ""))
                
                # Write test file
                test_file = temp_path / llm_result.get("test_file_name", "test_generated.py")
                test_file.write_text(llm_result.get("test_code", ""))
                
                # Install dependencies if specified
                dependencies = llm_result.get("dependencies", [])
                if dependencies:
                    deps_str = " ".join(dependencies)
                    install_result = self.sandbox.run_in_sandbox(
                        f"pip install {deps_str}",
                        timeout=30,
                        cwd=temp_path
                    )
                    if install_result["exit_code"] != 0:
                        logger.warning(f"Failed to install dependencies: {install_result['stderr']}")
                
                # Validate using sandbox
                validation = self.sandbox.validate_code(code_file)
                
                return validation
                
        except Exception as e:
            logger.error(f"Code validation failed: {str(e)}")
            return {
                "tests_passed": False,
                "quality_score": 0,
                "lint_errors": [f"Validation error: {str(e)}"],
                "test_output": f"Validation failed: {str(e)}"
            }
    
    def _prepare_retry_feedback(self, validation_result: Dict) -> str:
        """Prepare feedback for retry attempts."""
        feedback_parts = []
        
        if not validation_result.get("tests_passed", False):
            feedback_parts.append("Tests are failing")
        
        lint_errors = validation_result.get("lint_errors", [])
        if lint_errors:
            feedback_parts.append(f"Linting errors: {'; '.join(lint_errors[:3])}")
        
        quality_score = validation_result.get("quality_score", 0)
        if quality_score < self.quality_threshold:
            feedback_parts.append(f"Quality score {quality_score} below threshold {self.quality_threshold}")
        
        return ". ".join(feedback_parts) if feedback_parts else "General quality issues"
    
    async def ainvoke(self, query: str, *args, **kwargs) -> Any:
        """
        Async invoke method for agent compatibility.
        
        Args:
            query: Code generation specification
            
        Returns:
            Generated code result
        """
        context = kwargs.get("learning_context", kwargs.get("context", {})) or {}

        # Persist user message similar to base Agent
        try:
            if getattr(self, "memory", None):
                framework_id = context.get("framework")
                uid = getattr(self, "_user_id", None) or kwargs.get("user_id") or "unknown_user"
                self.memory.append_chat_message(uid, role="user", content=query, framework=framework_id)
                self.save_memory(query, user_id=uid)
        except Exception:
            pass

        result = await self.generate_code(query, context)

        # Persist assistant result summary
        try:
            if getattr(self, "memory", None):
                framework_id = context.get("framework")
                uid = getattr(self, "_user_id", None) or kwargs.get("user_id") or "unknown_user"
                summary = str(result)[:800]
                self.memory.append_chat_message(uid, role="assistant", content=summary, framework=framework_id)
                self.save_memory(summary, user_id=uid)
        except Exception:
            pass

        return result
    
    def invoke(self, query: str, *args, **kwargs) -> Any:
        """
        Sync invoke method for agent compatibility.
        
        Args:
            query: Code generation specification
            
        Returns:
            Generated code result
        """
        return asyncio.run(self.ainvoke(query, *args, **kwargs))