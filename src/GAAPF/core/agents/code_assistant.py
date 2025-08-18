import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from ...prompts.code_assistant import generate_system_prompt
from ..utils.retriever import Retriever
from ..tools.vector_store import VectorStore

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
    4. Providing debugging assistance
    5. Offering best practices for code organization
    6. Adapting code complexity to user's skill level
    """
    
    # Class attributes for agent registry
    DESCRIPTION = "Specialized in providing code examples and implementation guidance"
    CAPABILITIES = [
        "code_examples",
        "implementation_guidance",
        "debugging_assistance",
        "best_practices",
        "code_explanation",
        "hands_on_coding"
    ]
    PRIORITY = 9  # High priority for coding tasks
    
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
                "computer_tools",  # Essential for file creation
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
        # Phase 3.3: Validation mode (light checks, no execution)
        self.validation_mode = bool(config.get("validation_mode", False))
        
        # Initialize Retriever for RAG capabilities
        self.retriever = None
        self.current_framework = None
        
        if self.is_logging:
            logger.info(f"Initialized CodeAssistantAgent with config: {self.config}")
            logger.info(f"Enhanced capabilities: validation={self.code_validation_enabled}, "
                       f"auto_execution={self.auto_execution_enabled}, "
                       f"framework_integration={self.framework_integration_enabled}")
    
    def execute_code_safely(self, code: str, language: str = "python") -> Dict:
        """Public method to safely execute code snippets.
        
        Parameters:
        ----------
        code : str
            Code to execute
        language : str, optional
            Programming language (python, bash, sh)
            
        Returns:
        -------
        Dict
            Execution result with success status, output, and errors
        """
        import asyncio
        
        # Run the async method in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self._safe_execute_code(code, language))
    
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
            
            # Phase 3.3: If validation mode is enabled, perform light run checks only
            if self.validation_mode:
                light_check = await self._light_run_check(code, language)
                return {
                    "success": bool(light_check.get("ok", False)),
                    "validation": validation_result,
                    "light_check": light_check,
                    "message": "Light validation only (no execution)",
                }

            # Execute the code if validation passes
            if self.auto_execution_enabled:
                execution_result = await self._safe_execute_code(code, language)
                # Phase 3.2: If execution failed, diagnose and propose fixes
                if not execution_result.get("success", False):
                    error_text = "\n".join(execution_result.get("errors", [])) if execution_result.get("errors") else str(execution_result)
                    advice = await self._diagnose_error_and_suggest_fix(code, error_text, language, framework_context or {})
                    return {
                        "success": False,
                        "validation": validation_result,
                        "execution": execution_result,
                        "suggestions": advice.get("suggestions")
                    }
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
        """Safely execute code with proper error handling using available tools."""
        try:
            # Get the appropriate execution tool from tools manager
            bash_command_tool = self.tools_manager.get_tool("run_bash_command")
            
            if not bash_command_tool:
                return {
                    "success": False,
                    "error": "Execution tools not available"
                }
            
            # Handle different languages
            if language.lower() in ['python', 'py']:
                # For Python code, create a temporary file and execute it
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                try:
                    # Execute Python file
                    result = bash_command_tool(f"python {temp_file}")
                    # Phase 3.2: Detect error output pattern from tool and convert to structured error
                    if isinstance(result, str) and result.strip().lower().startswith("error:"):
                        return {
                            "success": False,
                            "output": "",
                            "errors": [result]
                        }
                    return {
                        "success": True,
                        "output": result,
                        "errors": []
                    }
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        
            elif language.lower() in ['bash', 'sh', 'shell', 'powershell', 'ps1', 'cmd']:
                 # For shell commands, execute directly
                 # On Windows, adapt bash commands to PowerShell equivalents
                 import platform
                 if platform.system() == "Windows":
                     # Convert common bash commands to PowerShell
                     if code.strip().startswith('ls'):
                         # Handle ls with pipes
                         if '|' in code and 'head' in code:
                             # Convert "ls | head -n" to "Get-ChildItem | Select-Object -First n"
                             import re
                             match = re.search(r'head\s+-?(\d+)', code)
                             if match:
                                 num = match.group(1)
                                 code = f"Get-ChildItem | Select-Object -First {num}"
                             else:
                                 code = "Get-ChildItem | Select-Object -First 10"
                         else:
                             code = code.replace('ls', 'Get-ChildItem', 1)
                     elif code.strip().startswith('pwd'):
                         code = code.replace('pwd', 'Get-Location', 1)
                     elif code.strip().startswith('cat'):
                         code = code.replace('cat', 'Get-Content', 1)
                 
                 result = bash_command_tool(code)
                 # Phase 3.2: Detect error output pattern from tool and convert to structured error
                 if isinstance(result, str) and result.strip().lower().startswith("error:"):
                     return {
                         "success": False,
                         "output": "",
                         "errors": [result]
                     }
                 return {
                     "success": True,
                     "output": result,
                     "errors": []
                 }
            else:
                return {
                    "success": False,
                    "error": f"Language '{language}' not supported for execution"
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
        Generate a system prompt for this agent with relevant documentation context.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        str
            System prompt for the agent with injected documentation context
        """
        # Get base system prompt
        base_prompt = generate_system_prompt(self.config, learning_context)
        
        # Try to inject relevant documentation context
        try:
            framework_name = learning_context.get("framework_config", {}).get("name", "").lower()
            
            if framework_name and self._initialize_retriever_for_framework(framework_name):
                # Get some general documentation for the framework
                general_query = f"introduction overview getting started {framework_name}"
                relevant_docs = self.retriever.retrieve_docs(general_query, k=2) if self.retriever else []
                
                if relevant_docs:
                    # Format documentation context
                    context_parts = []
                    for i, doc in enumerate(relevant_docs, 1):
                        # Extract text content from document dict
                        doc_text = doc.get('text', str(doc)) if isinstance(doc, dict) else str(doc)
                        # Get metadata for source attribution
                        metadata = doc.get('metadata', {}) if isinstance(doc, dict) else {}
                        source = metadata.get('source', f'Doc {i}')
                        
                        # Truncate long documents to fit in prompt
                        doc_content = doc_text[:400] + "..." if len(doc_text) > 400 else doc_text
                        context_parts.append(f"**Source: {source}**\n{doc_content}")
                    
                    context_text = "\n\n".join(context_parts)
                    
                    # Inject context into system prompt
                    enhanced_prompt = f"""{base_prompt}

**📚 FRAMEWORK DOCUMENTATION CONTEXT:**
The following documentation snippets are available for reference when providing {framework_name} guidance:

{context_text}

**IMPORTANT:** When relevant, reference these documentation sources in your responses to provide accurate, up-to-date information. Always cite sources when using specific information from the documentation.
"""
                    
                    if self.is_logging:
                        logger.info(f"Enhanced system prompt with {len(relevant_docs)} documentation snippets for {framework_name}")
                    
                    return enhanced_prompt
            
        except Exception as e:
            if self.is_logging:
                logger.error(f"Error injecting documentation context into system prompt: {str(e)}")
        
        # Return base prompt if context injection fails
        return base_prompt
    
    def _initialize_retriever_for_framework(self, framework_name: str) -> bool:
        """
        Initialize Retriever for a specific framework.
        
        Parameters:
        ----------
        framework_name : str
            Name of the framework (e.g., 'langchain', 'langgraph')
            
        Returns:
        -------
        bool
            True if successfully initialized, False otherwise
        """
        try:
            if framework_name != self.current_framework:
                # Create VectorStore path for the framework
                vs_path = f"data/frameworks/vectordb/{framework_name}/{framework_name}"
                collection_name = f"framework_docs_{framework_name}"
                
                # Initialize VectorStore with consistent embedding model
                vector_store = VectorStore(
                    persistent_dir=vs_path, 
                    collection_name=collection_name,
                    embedding_model="gemini-embedding-001"
                )
                
                # Check if VectorStore has data
                if vector_store.count() > 0:
                    self.retriever = Retriever(vector_store)
                    self.current_framework = framework_name
                    if self.is_logging:
                        logger.info(f"Initialized Retriever for framework: {framework_name}")
                    return True
                else:
                    if self.is_logging:
                        logger.warning(f"VectorStore for {framework_name} is empty")
                    return False
            return True
            
        except Exception as e:
            if self.is_logging:
                logger.error(f"Failed to initialize Retriever for {framework_name}: {str(e)}")
            return False
    
    def _enhance_query_with_context(self, query: str, learning_context: Dict) -> str:
        """
        Enhance a user query with specific relevant documentation for the query.
        
        Parameters:
        ----------
        query : str
            Original user query
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        str
            Enhanced query with query-specific documentation context
        """
        try:
            # Get current framework from learning context
            framework_name = learning_context.get("framework_config", {}).get("name", "").lower()
            
            if not framework_name:
                return query
            
            # Initialize retriever for current framework
            if not self._initialize_retriever_for_framework(framework_name):
                return query
            
            # Retrieve documents specifically relevant to this query
            if self.retriever:
                relevant_docs = self.retriever.retrieve_docs(query, k=3)
                
                if relevant_docs:
                    # Format retrieved context with source attribution
                    context_parts = []
                    for i, doc in enumerate(relevant_docs, 1):
                        # Extract text content from document dict
                        doc_text = doc.get('text', str(doc)) if isinstance(doc, dict) else str(doc)
                        # Get metadata for source attribution
                        metadata = doc.get('metadata', {}) if isinstance(doc, dict) else {}
                        source = metadata.get('source', f'Reference {i}')
                        
                        # Truncate long documents
                        doc_content = doc_text[:350] + "..." if len(doc_text) > 350 else doc_text
                        context_parts.append(f"**[{source}]**\n{doc_content}")
                    
                    context_text = "\n\n".join(context_parts)
                    
                    # Enhance query with specific context
                    enhanced_query = f"""{query}

**📋 QUERY-SPECIFIC DOCUMENTATION:**
The following documentation is specifically relevant to your question:

{context_text}

**Please reference these sources when providing your response and cite them appropriately.**"""
                    
                    if self.is_logging:
                        logger.info(f"Enhanced query with {len(relevant_docs)} query-specific documents")
                    
                    return enhanced_query
            
            return query
            
        except Exception as e:
            if self.is_logging:
                logger.error(f"Error enhancing query with context: {str(e)}")
            return query
    
    async def _light_run_check(self, code: str, language: str) -> Dict:
        """Phase 3.3: Perform a lightweight run validation without executing business logic.
        For Python, attempt bytecode compilation using py_compile.
        """
        try:
            lang = (language or "").lower()
            if lang in ["python", "py"]:
                import tempfile
                import os
                bash_command_tool = None
                try:
                    bash_command_tool = self.tools_manager.get_tool("run_bash_command")
                except Exception:
                    bash_command_tool = None
                if not bash_command_tool:
                    return {"ok": False, "reason": "Execution tools not available"}
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                try:
                    result = bash_command_tool(f"python -m py_compile {temp_file}")
                    if isinstance(result, str) and result.strip().lower().startswith("error:"):
                        return {"ok": False, "reason": result}
                    return {"ok": True}
                finally:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception:
                        pass
            # For other languages: no-op light check
            return {"ok": True}
        except Exception as e:
            if self.is_logging:
                logger.error(f"Light run check failed: {e}")
            return {"ok": False, "reason": str(e)}

    async def _diagnose_error_and_suggest_fix(self, code: str, error_text: str, language: str, learning_context: Dict) -> Dict:
        """Phase 3.2: Use the LLM to analyze runtime error and propose concise fix suggestions."""
        try:
            prompt = (
                "You are a code troubleshooting assistant. Analyze the runtime error and suggest a minimal fix.\n"
                f"Language: {language}\n\n"
                "Error details:\n"
                f"{error_text}\n\n"
                "Code snippet:\n"
                f"{code}\n\n"
                "Respond with a short bullet list of concrete steps to fix the issue."
            )
            # Use async LLM call if available
            if hasattr(self.llm, "ainvoke"):
                resp = await self.llm.ainvoke(prompt)
                content = getattr(resp, "content", str(resp))
            else:
                resp = self.llm.invoke(prompt)
                content = getattr(resp, "content", str(resp))
            return {"suggestions": content.strip() if isinstance(content, str) else str(content)}
        except Exception as e:
            if self.is_logging:
                logger.error(f"Error diagnosis failed: {e}")
            return {"suggestions": "Unable to generate suggestions due to an internal error."}

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

        # Extract code blocks and detected language
        code_blocks = self._extract_code_from_response(processed.get("content", "") or "")
        detected_language = self._detect_language(processed.get("content", "") or "", learning_context)

        # If code exists in the response, automatically write to files and remove code from reply
        if code_blocks:
            try:
                written_files = self._write_code_files(code_blocks, learning_context)
                # Strip code blocks from the content to enforce "no code in response"
                import re
                content_no_code = re.sub(r"```[a-zA-Z0-9_+\-]*\s*[\s\S]*?```", "", processed.get("content", ""))
                content_no_code = content_no_code.strip()
                # Compose a concise English-only message listing created files
                if written_files:
                    files_list_text = "\n".join([f"- {p}" for p in written_files])
                    notice = (
                        "I created the code files for you (no code shown inline):\n" + files_list_text
                    )
                else:
                    notice = "Code files were generated."
                # Keep any non-code explanation if present, then append notice
                processed["content"] = (content_no_code + ("\n\n" if content_no_code else "") + notice).strip()
            except Exception as e:
                logger.error(f"Failed to write code files: {e}")
                # As a fallback, remove code blocks from content to honor the no-code policy
                import re
                processed["content"] = re.sub(r"```[a-zA-Z0-9_+\-]*\s*[\s\S]*?```", "", processed.get("content", "") or "").strip()

        # Add code assistant-specific metadata
        processed["code_example"] = code_blocks
        processed["language"] = detected_language
        # Lightweight signals for hub state updates (Phase 5)
        try:
            signals = dict(processed.get("signals") or {})
            # If there is any validation/execution information attached upstream, surface success signal
            # (We avoid running execution here; just propagate known flags.)
            exec_info = processed.get("execution") or {}
            if isinstance(exec_info, dict) and exec_info.get("success") is True:
                signals["code_success"] = True
            processed["signals"] = signals
        except Exception:
            pass

        return processed

    def _write_code_files(self, code_blocks: List[Dict], learning_context: Dict) -> List[str]:
        """Write extracted code blocks to disk using the write_file tool.
        Returns a list of created file paths.
        """
        # Resolve framework directory
        framework_id = (learning_context or {}).get("framework") or (learning_context or {}).get("framework_config", {}).get("name") or "general"
        framework_id = str(framework_id).lower()
        base_dir = Path("generated") / framework_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extensions by language
        def _ext_for_language(lang: str) -> str:
            mapping = {
                "python": "py",
                "py": "py",
                "javascript": "js",
                "js": "js",
                "typescript": "ts",
                "ts": "ts",
                "bash": "sh",
                "shell": "sh",
                "sh": "sh",
                "markdown": "md",
                "md": "md",
                "json": "json",
                "yaml": "yml",
                "yml": "yml",
                "html": "html",
                "css": "css",
                "java": "java",
                "go": "go",
            }
            lang = (lang or "").strip().lower()
            return mapping.get(lang, "txt")

        # Load write_file tool
        write_tool = None
        try:
            write_tool = self.tools_manager.get_tool("write_file")
            if not write_tool:
                # Try to register the module if not already loaded
                self.tools_manager.register_module_tool("computer_tools")
                write_tool = self.tools_manager.get_tool("write_file")
        except Exception as e:
            logger.error(f"Unable to load write_file tool: {e}")

        written_paths: List[str] = []
        import time as _time
        for idx, block in enumerate(code_blocks, start=1):
            lang = (block.get("language") or "").strip().lower()
            ext = _ext_for_language(lang)
            ts = int(_time.time())
            filename = f"snippet_{ts}_{idx}.{ext}"
            file_path = base_dir / filename
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content = block.get("code", "")
            try:
                if write_tool and callable(write_tool):
                    # Use the registered tool function
                    write_tool(path=str(file_path), content=content)
                else:
                    # Fallback: write directly
                    file_path.write_text(content, encoding="utf-8")
                written_paths.append(str(file_path))
            except Exception as e:
                logger.error(f"Failed writing file {file_path}: {e}")

        return written_paths
    
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