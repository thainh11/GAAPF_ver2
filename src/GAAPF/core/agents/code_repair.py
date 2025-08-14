"""
Code Repair Agent - GAAPF Phase 2
Specialized agent for analyzing and fixing bugs in existing code files.
"""

import logging
import asyncio
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import tempfile

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import BaseTool

from . import SpecializedAgent
from ..tools.sandbox_runner import SandboxRunner

# Setup logging
logger = logging.getLogger(__name__)

try:
    import unidiff
    UNIDIFF_AVAILABLE = True
except ImportError:
    UNIDIFF_AVAILABLE = False
    logger.warning("unidiff not available, using simple string replacement for patches")


class CodeRepairAgent(SpecializedAgent):
    """
    Specialized agent for analyzing and fixing bugs in existing code files.
    
    This agent can read existing code, identify issues, generate patches,
    and validate fixes through testing.
    """
    
    # Class attributes for agent registry
    DESCRIPTION = "AI agent specialized in analyzing and fixing bugs in Python code"
    CAPABILITIES = ["analyze_code", "fix_bugs", "apply_patches", "validate_fixes"]
    PRIORITY = 5  # Higher priority for code repair tasks
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[Union[str, BaseTool]] = [],
        memory_path: Optional[Path] = None,
        config: Dict = None,
        is_logging: bool = False,
        max_retries: int = 3,
        *args, **kwargs
    ):
        """
        Initialize CodeRepairAgent.
        
        Args:
            llm: Language model for code analysis and repair
            tools: Additional tools (inherited from base)
            memory_path: Path for agent memory
            config: Agent configuration
            is_logging: Enable detailed logging
            max_retries: Maximum repair attempts
        """
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            config=config,
            agent_type="code_repair",
            description=self.DESCRIPTION,
            is_logging=is_logging,
            *args, **kwargs
        )
        
        # Code repair specific attributes
        self.max_retries = max_retries
        self.sandbox = SandboxRunner(timeout=15, memory_limit_mb=512)
        
        # Setup JSON parser for structured output
        self.json_parser = JsonOutputParser()
        
        if self.is_logging:
            logger.info(f"CodeRepairAgent initialized with max_retries={max_retries}")
    
    def _generate_system_prompt(self, learning_context: Optional[Dict] = None) -> str:
        """Generate system prompt for code repair tasks."""
        return """You are a Python Code Repair Agent specialized in analyzing and fixing bugs in existing code.

**Your Core Responsibilities:**
1. Analyze existing Python code to identify bugs and issues
2. Generate precise patches to fix identified problems
3. Ensure fixes maintain code functionality and don't introduce new bugs
4. Provide clear explanations of what was fixed and why

**Code Analysis Guidelines:**
- Read and understand the existing code structure
- Identify syntax errors, logic errors, and potential runtime issues
- Consider edge cases and error handling
- Maintain the original code's intent and functionality
- Follow Python best practices and PEP 8 style guidelines

**Patch Generation Guidelines:**
- Generate minimal, targeted fixes that address specific issues
- Preserve existing functionality while fixing bugs
- Add proper error handling where needed
- Improve code clarity without changing core logic
- Include type hints and documentation improvements if beneficial

**Output Format:**
Always return your response as JSON with the following structure:
{
    "analysis": "Detailed analysis of the issues found in the code",
    "issues_found": ["list", "of", "specific", "issues"],
    "patch": "Unified diff format patch to fix the issues",
    "fixed_code": "Complete fixed version of the code",
    "explanation": "Clear explanation of what was fixed and why",
    "test_suggestions": "Suggested tests to verify the fixes"
}

Be precise, conservative, and ensure all fixes are thoroughly considered."""
    
    async def repair_file(self, file_path: Union[str, Path], context: Dict = None) -> Dict[str, Any]:
        """
        Repair bugs in an existing code file.
        
        Args:
            file_path: Path to the code file to repair
            context: Additional context for repair
            
        Returns:
            Dict containing repair results and validation
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "analysis": "File does not exist",
                "issues_found": ["File not found"],
                "patch": "",
                "fixed_code": "",
                "explanation": "Cannot repair non-existent file",
                "repair_attempts": 0
            }
        
        if self.is_logging:
            logger.info(f"Repairing file: {file_path}")
        
        try:
            # Read the original file
            original_code = file_path.read_text(encoding='utf-8')
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read file: {str(e)}",
                "analysis": f"File reading error: {str(e)}",
                "issues_found": ["File reading error"],
                "patch": "",
                "fixed_code": "",
                "explanation": "Could not read the file",
                "repair_attempts": 0
            }
        
        context = context or {}
        
        # Attempt repair with retries
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            try:
                if self.is_logging:
                    logger.info(f"Repair attempt {attempt + 1}")
                
                # Analyze and repair the code
                repair_result = await self._analyze_and_repair(original_code, file_path, context, last_error)
                
                # Validate the repair
                validation_result = await self._validate_repair(
                    original_code, 
                    repair_result.get("fixed_code", ""),
                    file_path
                )
                
                # Combine results
                result = {
                    **repair_result,
                    **validation_result,
                    "repair_attempts": attempt + 1,
                    "original_file": str(file_path)
                }
                
                # Check if repair was successful
                if validation_result.get("tests_passed", False) or validation_result.get("syntax_valid", False):
                    result["success"] = True
                    
                    if self.is_logging:
                        logger.info(f"File repair successful after {attempt + 1} attempts")
                    
                    return result
                
                # Prepare for retry if needed
                if attempt < self.max_retries - 1:
                    last_error = validation_result.get("test_output", "Validation failed")
                
                attempt += 1
                
            except Exception as e:
                logger.error(f"Repair attempt {attempt + 1} failed: {str(e)}")
                last_error = str(e)
                attempt += 1
        
        # All attempts failed
        return {
            "success": False,
            "error": "All repair attempts failed",
            "analysis": "Unable to successfully repair the file",
            "issues_found": ["Repair failed"],
            "patch": "",
            "fixed_code": original_code,
            "explanation": f"Failed to repair after {self.max_retries} attempts. Last error: {last_error}",
            "repair_attempts": self.max_retries,
            "original_file": str(file_path),
            "tests_passed": False,
            "quality_score": 0,
            "lint_errors": [last_error or "Unknown error"]
        }
    
    async def _analyze_and_repair(self, original_code: str, file_path: Path, context: Dict, previous_error: str = None) -> Dict[str, Any]:
        """
        Analyze code and generate repair.
        
        Args:
            original_code: Original code content
            file_path: Path to the file being repaired
            context: Repair context
            previous_error: Error from previous attempt (for retry)
            
        Returns:
            Dict with analysis and repair results
        """
        # Prepare the prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._generate_system_prompt()),
            ("human", """Analyze and repair the following Python code:

File: {file_path}
Code:
```python
{original_code}
```

Context:
- Framework: {framework}
- Additional Info: {additional_info}
{previous_error_section}

Please analyze the code for bugs and issues, then provide a complete repair.""")
        ])
        
        # Extract context with defaults
        framework = context.get("framework", "general")
        additional_info = context.get("additional_info", "None")
        
        previous_error_section = ""
        if previous_error:
            previous_error_section = f"\nPrevious repair attempt failed with: {previous_error}\nPlease address this issue in your repair."
        
        # Generate repair using LLM
        chain = prompt_template | self.llm | self.json_parser
        
        result = await chain.ainvoke({
            "file_path": str(file_path),
            "original_code": original_code,
            "framework": framework,
            "additional_info": additional_info,
            "previous_error_section": previous_error_section
        })
        
        # Ensure we have a fixed_code field
        if "fixed_code" not in result:
            # Try to apply patch to get fixed code
            if "patch" in result and result["patch"]:
                try:
                    result["fixed_code"] = self._apply_patch(original_code, result["patch"])
                except Exception as e:
                    logger.warning(f"Failed to apply patch: {e}")
                    result["fixed_code"] = original_code
            else:
                result["fixed_code"] = original_code
        
        return result
    
    def _apply_patch(self, original_code: str, patch_str: str) -> str:
        """
        Apply a patch to the original code.
        
        Args:
            original_code: Original code content
            patch_str: Patch in unified diff format
            
        Returns:
            Patched code
        """
        if not UNIDIFF_AVAILABLE:
            # Fallback: return the patch as-is if it looks like complete code
            if patch_str.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                return patch_str
            else:
                return original_code
        
        try:
            # Parse the patch
            patch_set = unidiff.PatchSet(patch_str)
            
            if not patch_set:
                return original_code
            
            # Apply the patch
            lines = original_code.splitlines(keepends=True)
            
            for patched_file in patch_set:
                for hunk in patched_file:
                    # Simple application - this is a basic implementation
                    # In production, you'd want more sophisticated patch application
                    target_line = hunk.target_start - 1
                    
                    # Remove lines
                    for line in hunk:
                        if line.is_removed:
                            if target_line < len(lines):
                                lines.pop(target_line)
                            else:
                                target_line += 1
                        elif line.is_added:
                            lines.insert(target_line, line.value)
                            target_line += 1
                        else:
                            target_line += 1
            
            return ''.join(lines)
            
        except Exception as e:
            logger.warning(f"Patch application failed: {e}")
            # Fallback to simple replacement if patch looks like complete code
            if patch_str.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                return patch_str
            return original_code
    
    async def _validate_repair(self, original_code: str, fixed_code: str, file_path: Path) -> Dict[str, Any]:
        """
        Validate the repair by running tests and checks.
        
        Args:
            original_code: Original code
            fixed_code: Repaired code
            file_path: File path for context
            
        Returns:
            Dict with validation results
        """
        try:
            # Create temporary file for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write fixed code to temporary file
                temp_file = temp_path / file_path.name
                temp_file.write_text(fixed_code)
                
                # Basic syntax check
                syntax_result = self.sandbox.run_in_sandbox(
                    f"python -m py_compile {temp_file.name}",
                    timeout=10,
                    cwd=temp_path
                )
                
                syntax_valid = syntax_result["exit_code"] == 0
                
                # Try to run any existing tests
                test_result = self.sandbox.run_in_sandbox(
                    f"python -m pytest {temp_file.name} -v",
                    timeout=15,
                    cwd=temp_path
                )
                
                tests_passed = test_result["exit_code"] == 0
                
                # Run linting
                lint_result = self.sandbox.run_in_sandbox(
                    f"python -m ruff check {temp_file.name}",
                    timeout=10,
                    cwd=temp_path
                )
                
                lint_errors = []
                if lint_result["exit_code"] != 0 and lint_result["stderr"]:
                    lint_errors = lint_result["stderr"].split('\n')[:3]  # Limit to 3 errors
                
                # Calculate quality score
                quality_score = 0
                if syntax_valid:
                    quality_score += 40  # 40% for valid syntax
                if tests_passed:
                    quality_score += 40  # 40% for passing tests
                if not lint_errors:
                    quality_score += 20  # 20% for clean linting
                
                return {
                    "syntax_valid": syntax_valid,
                    "tests_passed": tests_passed,
                    "quality_score": quality_score,
                    "lint_errors": lint_errors,
                    "test_output": test_result["stdout"] + test_result["stderr"],
                    "syntax_output": syntax_result["stderr"] if not syntax_valid else ""
                }
                
        except Exception as e:
            logger.error(f"Repair validation failed: {str(e)}")
            return {
                "syntax_valid": False,
                "tests_passed": False,
                "quality_score": 0,
                "lint_errors": [f"Validation error: {str(e)}"],
                "test_output": f"Validation failed: {str(e)}",
                "syntax_output": ""
            }
    
    async def ainvoke(self, query: str, *args, **kwargs) -> Any:
        """
        Async invoke method for agent compatibility.
        
        Args:
            query: File path to repair
            
        Returns:
            Repair result
        """
        context = kwargs.get("learning_context", kwargs.get("context", {})) or {}

        # Persist user message similar to base Agent
        try:
            if getattr(self, "memory", None):
                framework_id = context.get("framework")
                uid = getattr(self, "_user_id", None) or kwargs.get("user_id") or "unknown_user"
                self.memory.append_chat_message(uid, role="user", content=str(query), framework=framework_id)
                self.save_memory(str(query), user_id=uid)
        except Exception:
            pass

        result = await self.repair_file(query, context)

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
            query: File path to repair
            
        Returns:
            Repair result
        """
        return asyncio.run(self.ainvoke(query, *args, **kwargs))