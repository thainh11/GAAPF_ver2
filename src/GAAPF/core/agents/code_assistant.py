import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from src.GAAPF.prompts.code_assistant import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeAssistantAgent(SpecializedAgent):
    """
    Specialized agent focused on providing code examples and implementation guidance.
    
    The CodeAssistantAgent is responsible for:
    1. Creating code examples that demonstrate framework concepts
    2. Explaining code implementation details
    3. Helping users translate concepts into working code
    4. Providing best practices for code implementation
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
        Initialize the CodeAssistantAgent.
        
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
                "code_style": "clean",  # clean, verbose, compact
                "include_comments": True,
                "error_handling": "basic",  # none, basic, comprehensive
                "optimization_level": "standard",  # minimal, standard, optimized
                "show_alternatives": False
            }
        
        # Set default tools if not provided
        if not tools:
            tools = [
                "websearch_tools",
                "computer_tools",
                "terminal_tools",
                "framework_collector",
                "deepsearch"
            ]
        
        # Initialize the base specialized agent
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            config=config,
            agent_type="code_assistant",
            description="Expert in providing code examples and implementation guidance",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        # Enhanced capabilities for Phase 4
        self.code_validation_enabled = True
        self.auto_execution_enabled = config.get("auto_execution", False)
        self.framework_integration_enabled = True
        
        if self.is_logging:
            logger.info(f"Initialized CodeAssistantAgent with config: {self.config}")
            logger.info(f"Enhanced capabilities: validation={self.code_validation_enabled}, "
                       f"auto_execution={self.auto_execution_enabled}, "
                       f"framework_integration={self.framework_integration_enabled}")
    
    async def execute_code_with_validation(self, code: str, language: str, framework_context: Dict = None) -> Dict:
        """
        Execute code with validation and error handling.
        
        Parameters:
        ----------
        code : str
            Code to execute
        language : str
            Programming language
        framework_context : Dict, optional
            Framework-specific context
            
        Returns:
        -------
        Dict
            Execution results with validation
        """
        try:
            if self.is_logging:
                logger.info(f"Executing code with validation for language: {language}")
            
            # Pre-execution validation
            validation_result = await self._validate_code_syntax(code, language, framework_context)
            
            if not validation_result.get("is_valid", False):
                return {
                    "success": False,
                    "error": "Code validation failed",
                    "validation_errors": validation_result.get("errors", []),
                    "suggestions": validation_result.get("suggestions", [])
                }
            
            # Execute the code if validation passes
            if self.auto_execution_enabled:
                execution_result = await self._safe_execute_code(code, language)
                return {
                    "success": True,
                    "validation": validation_result,
                    "execution": execution_result
                }
            else:
                return {
                    "success": True,
                    "validation": validation_result,
                    "message": "Code validated successfully. Execute manually to see results."
                }
                
        except Exception as e:
            logger.error(f"Error in code execution with validation: {str(e)}")
            return {
                "success": False,
                "error": f"Execution error: {str(e)}"
            }
    
    async def _validate_code_syntax(self, code: str, language: str, framework_context: Dict = None) -> Dict:
        """Validate code syntax and framework compliance."""
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": []
            }
            
            # Basic syntax validation
            if language.lower() == "python":
                try:
                    import ast
                    ast.parse(code)
                except SyntaxError as e:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"Syntax error: {str(e)}")
            
            # Framework-specific validation
            if framework_context and self.framework_integration_enabled:
                framework_validation = await self._validate_framework_compliance(
                    code, framework_context
                )
                validation_result["warnings"].extend(framework_validation.get("warnings", []))
                validation_result["suggestions"].extend(framework_validation.get("suggestions", []))
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in code validation: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }
    
    async def _validate_framework_compliance(self, code: str, framework_context: Dict) -> Dict:
        """Validate code against framework best practices."""
        try:
            framework_name = framework_context.get("name", "")
            validation_result = {
                "warnings": [],
                "suggestions": []
            }
            
            # Framework-specific checks
            if "django" in framework_name.lower():
                if "import django" not in code and "from django" not in code:
                    validation_result["warnings"].append("No Django imports detected")
                    validation_result["suggestions"].append("Consider adding Django imports if needed")
            
            elif "flask" in framework_name.lower():
                if "from flask import" not in code and "import flask" not in code:
                    validation_result["warnings"].append("No Flask imports detected")
                    validation_result["suggestions"].append("Consider adding Flask imports if needed")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in framework validation: {str(e)}")
            return {"warnings": [], "suggestions": []}
    
    async def _safe_execute_code(self, code: str, language: str) -> Dict:
        """Safely execute code with proper error handling."""
        try:
            # Use computer tools for safe execution
            execution_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and 'execute_code' in tool.name:
                    execution_tool = tool
                    break
            
            if execution_tool:
                result = await execution_tool.arun(
                    code=code,
                    language=language,
                    timeout=30  # 30 second timeout
                )
                return {
                    "success": True,
                    "output": result.get("output", ""),
                    "errors": result.get("errors", [])
                }
            else:
                return {
                    "success": False,
                    "error": "Code execution tool not available"
                }
                
        except Exception as e:
            logger.error(f"Error in safe code execution: {str(e)}")
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}"
            }
    
    async def generate_enhanced_code_example(self, concept: str, framework_context: Dict, user_level: str) -> Dict:
        """
        Generate enhanced code examples with validation and best practices.
        
        Parameters:
        ----------
        concept : str
            Concept to demonstrate
        framework_context : Dict
            Framework-specific context
        user_level : str
            User's experience level
            
        Returns:
        -------
        Dict
            Enhanced code example
        """
        # This would use the LLM to generate code based on context
        prompt = f"""
        Generate a code example for the concept '{concept}' in the framework '{framework_context.get('name', '')}'.
        The user's experience level is {user_level}.
        Provide the code and a brief explanation.
        """
        response = await self.llm.ainvoke(prompt)
        
        # This is a simplified extraction. A real implementation would parse the response more robustly.
        code_content = response.content
        return {
            "code": code_content,
            "explanation": "This is a generated explanation for the code.",
            "is_validated": False
        }
    
    def _generate_system_prompt(self, learning_context: Dict) -> str:
        """
        Generate a system prompt for this agent.
        
        Returns:
        -------
        str
            System prompt for the agent
        """
        return generate_system_prompt(self.config, learning_context)
    
    def _enhance_query_with_context(self, query: str, learning_context: Dict) -> str:
        """
        Enhance a user query with learning context specific to the code assistant role.
        """
        return query
    
    def _process_response(self, response: Any, learning_context: Dict) -> Dict:
        """
        Process and structure the code assistant's response.
        
        Parameters:
        ----------
        response : Any
            Raw response from the agent
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        Dict
            Processed and structured response
        """
        # Get base processed response
        processed = super()._process_response(response, learning_context)
        
        # Add code assistant-specific metadata
        processed["code_example"] = self._extract_code_from_response(processed["content"])
        processed["language"] = self._detect_language(processed["content"], learning_context)
        
        return processed
    
    def _extract_code_from_response(self, response_content: str) -> List[Dict]:
        """
        Extract code blocks from the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
            
        Returns:
        -------
        List[Dict]
            List of extracted code blocks with metadata
        """
        # In a real implementation, this would parse markdown code blocks
        # For now, we'll use a simple approach
        
        code_blocks = []
        import re
        
        # Find code blocks with language specification
        pattern = r"```([a-zA-Z0-9_+-]+)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, response_content)
        
        for idx, (lang, code) in enumerate(matches):
            code_blocks.append({
                "id": idx,
                "language": lang.strip() if lang else "unknown",
                "code": code.strip(),
                "start_index": response_content.find(f"```{lang}"),
                "end_index": response_content.find("```", response_content.find(f"```{lang}") + 3) + 3
            })
        
        return code_blocks
    
    def _detect_language(self, response_content: str, learning_context: Dict) -> str:
        """
        Detect the programming language used in the response.
        
        Parameters:
        ----------
        response_content : str
            Content of the response
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        str
            Detected programming language
        """
        # First try to get language from code blocks
        code_blocks = self._extract_code_from_response(response_content)
        if code_blocks and code_blocks[0]["language"] != "unknown":
            return code_blocks[0]["language"]
        
        # Fall back to framework language
        framework_config = learning_context.get("framework_config", {})
        return framework_config.get("language", "unknown") 