import sys
import os
import json
import inspect
import importlib
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable, Union, Literal
import ast
import uuid
from pathlib import Path
from ..mcp import load_mcp_tools
from ..mcp.client import DistributedMCPClient
from langchain_core.messages.tool import ToolMessage
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlobalToolRegistry:
    """
    Singleton class to manage global tool registration and caching.
    Prevents duplicate tool analysis and registration across multiple agents.
    """
    _instance = None
    _registered_modules = {}  # Cache for already analyzed modules
    _tools_cache = {}  # Cache for tool definitions
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalToolRegistry, cls).__new__(cls)
        return cls._instance
    
    def is_module_registered(self, module_name: str) -> bool:
        """Check if a module has already been registered and analyzed."""
        return module_name in self._registered_modules
    
    def get_module_tools(self, module_name: str) -> Optional[Dict]:
        """Get cached tools for a module."""
        return self._registered_modules.get(module_name)
    
    def register_module_tools(self, module_name: str, tools: Dict):
        """Cache the tools for a module."""
        self._registered_modules[module_name] = tools
        # Also cache individual tools
        for tool_name, tool_data in tools.items():
            self._tools_cache[tool_name] = tool_data
        logger.info(f"Cached {len(tools)} tools for module {module_name}")
    
    def get_all_cached_tools(self) -> Dict:
        """Get all cached tools from all modules."""
        return self._tools_cache.copy()
    
    def clear_cache(self):
        """Clear all cached tools (useful for testing or reset)."""
        self._registered_modules.clear()
        self._tools_cache.clear()
        logger.info("Cleared global tool registry cache")


class ToolManager:
    """Centralized tool management class with global caching"""
    def __init__(self, tools_path: Path = Path("templates/tools.json"), is_reset_tools: bool=False):
        self.tools_path = tools_path
        self.is_reset_tools = is_reset_tools
        self.tools_path = Path(tools_path) if isinstance(tools_path, str) else tools_path
        if not self.tools_path.exists():
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

        if self.is_reset_tools:
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")
            # Also clear global cache when resetting
            GlobalToolRegistry().clear_cache()

        self._registered_functions: Dict[str, Callable] = {}
        self.global_registry = GlobalToolRegistry()
        
    def load_tools(self) -> Dict[str, Any]:
        """Load existing tools from JSON"""
        if self.tools_path:
            with open(self.tools_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}

    def save_tools(self, tools: Dict[str, Any]) -> None:
        """Save tools to JSON"""
        with open(self.tools_path, "w", encoding="utf-8") as f:
            json.dump(tools, f, indent=4, ensure_ascii=False)

    def register_function_tool(self, func):
        """Decorator to register a function as a tool
        # Example usage:
        @function_tool
        def sample_function(x: int, y: str) -> str:
            '''Sample function for testing'''
            return f"{y}: {x}"
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Get function metadata
        signature = inspect.signature(func)

        # Try to get module path, fall back to None if not available
        module_path = "__runtime__"

        # Create metadata
        if module_path == "__runtime__":
            metadata = {
                "tool_name": func.__name__,
                "arguments": {
                    name: (
                        str(param.annotation)
                        if param.annotation != inspect.Parameter.empty
                        else "Any"
                    )
                    for name, param in signature.parameters.items()
                },
                "return": (
                    str(signature.return_annotation)
                    if signature.return_annotation != inspect.Signature.empty
                    else "Any"
                ),
                "docstring": (func.__doc__ or "").strip(),
                "module_path": module_path,
                "tool_type": "function",
                "tool_call_id": "tool_" + str(uuid.uuid4()),
                "is_runtime": module_path == "__runtime__",
            }

            # Register both the function and its metadata
            self._registered_functions[func.__name__] = func
            tools = self.load_tools()
            tools[func.__name__] = metadata
            self.save_tools(tools)
            logger.info(
                f"Registered tool: {func.__name__} "
                f"({'runtime' if module_path == '__runtime__' else 'file-based'})"
            )
        return wrapper

    async def register_mcp_tool(self, client: DistributedMCPClient, server_name: str = None) -> list[Dict[str, Any]]:
        # Load all tools
        logger.info(f"Registering MCP tools")
        all_tools = []
        if server_name:
            all_tools = await client.get_tools(server_name=server_name)
            logger.info(f"Loaded MCP tools of {server_name}: {len(all_tools)}")
        else:
            try:
                all_tools = await client.get_tools()
                logger.info(f"Loaded MCP tools: {len(all_tools)}")
            except Exception as e:
                logger.error(f"Error loading MCP tools: {e}")
                return []
        # Convert MCP tools to our format
        def convert_mcp_tool(mcp_tool: Dict[str, Any]):
            tool_name = mcp_tool['name']
            arguments = dict([(k, v['type']) for (k,v) in mcp_tool['args_schema']['properties'].items()])
            docstring = mcp_tool['description']
            return_value = mcp_tool['response_format']
            tool = {}
            tool['tool_name'] = tool_name
            tool['arguments'] = arguments
            tool['return'] = return_value
            tool['docstring'] = docstring
            tool['module_path'] = '__mcp__'
            tool['tool_type'] = 'mcp'
            # tool['mcp_client_connections'] = client.connections
            # tool['mcp_server_name'] = server_name
            tool['tool_call_id'] = "tool_" + str(uuid.uuid4())
            return tool
        
        new_tools = [convert_mcp_tool(mcp_tool.__dict__) for mcp_tool in all_tools]
        tools = self.load_tools()
        for tool in new_tools:
            tools[tool["tool_name"]] = tool
            tools[tool["tool_name"]]["tool_call_id"] = "tool_" + str(uuid.uuid4())
            logger.info(f"Registered {tool['tool_name']}:\n{tool}")
        self.save_tools(tools)
        logger.info(f"Completed registration for mcp module {server_name}")
        return new_tools

    def register_module_tool(self, module_name: str, llm=None) -> None:
        """Register tools from a module with global caching"""
        # Correctly form the full module path from the project root (src)
        full_module_path = f"src.GAAPF.core.tools.{module_name}"
        
        # Check if module is already registered in global cache
        if self.global_registry.is_module_registered(full_module_path):
            cached_tools = self.global_registry.get_module_tools(full_module_path)
            logger.info(f"Using cached tools for module {full_module_path} ({len(cached_tools)} tools)")
            
            # Add cached tools to local tools file
            tools = self.load_tools()
            for tool_name, tool_data in cached_tools.items():
                if tool_name not in tools:
                    tools[tool_name] = tool_data
            self.save_tools(tools)
            return
        
        # Module not cached, perform analysis
        logger.info(f"Analyzing new module: {full_module_path}")
        try:
            module = importlib.import_module(full_module_path)
            module_source = inspect.getsource(module)
        except (ImportError, ValueError, ModuleNotFoundError) as e:
            # Add a helpful error message for module not found
            if "No module named 'src'" in str(e):
                logger.error("Failed to import module. Ensure your project's root directory (containing 'src') is in PYTHONPATH.")
            raise ValueError(f"Failed to load module {full_module_path}: {str(e)}")

        if llm:
            try:
                # Use LLM to extract tools if available
                prompt = f"""Analyze the following Python module and extract all functions that can be used as tools.
Return a JSON object where keys are the function names and values are their metadata.
Metadata should include 'arguments' (a dictionary of argument names and their types), 'return' (the return type), and 'docstring'.

Do not include any other text, just the JSON object.

Module:
---
{module_source}
---
"""
                response = llm.invoke(prompt)
                
                # Clean the response content
                cleaned_content = response.content.strip()
                if cleaned_content.startswith("```json"):
                    cleaned_content = cleaned_content[7:]
                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-3]
                
                new_tools = json.loads(cleaned_content.strip())
            except (ValueError, SyntaxError, json.JSONDecodeError) as e:
                raise ValueError(f"Invalid tool format from LLM: {str(e)}")
        else:
            # Fallback to AST parsing if no LLM
            tree = ast.parse(module_source)
            new_tools = self._parse_ast_for_tools(tree)

        # Add module path and tool type to each tool
        for tool_name, tool_data in new_tools.items():
            tool_data["module_path"] = full_module_path
            tool_data["tool_type"] = "module"

        # Save the updated tools
        tools = self.load_tools()
        tools.update(new_tools)
        self.save_tools(tools)
        
        # Cache in global registry
        self.global_registry.register_module_tools(full_module_path, new_tools)

    def _parse_ast_for_tools(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Parse the AST of a module to extract tool functions.
        """
        tools: Dict[str, Any] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # argument types
                arguments: Dict[str, str] = {}
                for arg in node.args.args:
                    if arg.arg == "self":
                        continue
                    if arg.annotation:
                        arguments[arg.arg] = ast.unparse(arg.annotation)
                    else:
                        arguments[arg.arg] = "Any"

                # return type
                if node.returns:
                    return_type = ast.unparse(node.returns)
                else:
                    return_type = "Any"

                tools[node.name] = {
                    "tool_name": node.name,
                    "arguments": arguments,
                    "return": return_type,
                    "docstring": ast.get_docstring(node) or "",
                    "module_path": "__runtime__",   # filled in later
                    "tool_type": "function",
                    "tool_call_id": f"tool_{uuid.uuid4()}",
                    "is_runtime": True,
                }
        return tools

    def extract_tool(self, text: str) -> Optional[str]:
        """
        Return the first valid JSON object that *contains* the key "tool_name".
        Ignore brace pairs that are not valid JSON or are missing the key.
        """
        candidate = self._extract_json(text)
        while candidate:
            try:
                obj = json.loads(candidate)
                if "tool_name" in obj:
                    return candidate
            except json.JSONDecodeError:
                pass 
            next_start = text.find("{", text.find(candidate) + 1)
            if next_start == -1:
                break
            candidate = self._extract_json(text[next_start:])
        return None


    async def _execute_tool(self, 
                            tool_name: str, 
                            arguments: dict,
                            mcp_client: DistributedMCPClient,
                            mcp_server_name: str,
                            module_path: str,
                            tool_type: str = Literal['function', 'mcp', 'module']
                            ) -> Any:
        """Execute the specified tool with given arguments"""
        # FIX: Separate dangerous tools from safe tools
        dangerous_tools = {"run_command", "terminal_execute", "bash"}
        safe_tools = {"write_file"}  # Auto-approve safe tools
        
        if tool_name in dangerous_tools:
            # Truncate long content for display
            display_args = {}
            for key, value in arguments.items():
                if key == "content" and isinstance(value, str) and len(value) > 100:
                    display_args[key] = value[:100] + "... (truncated)"
                else:
                    display_args[key] = value
        
            # Only prompt for dangerous tools
            import asyncio
            answer = await asyncio.to_thread(
                input,
                f"\n[GAAPF] The agent wants to run '{tool_name}' "
                f"with arguments {display_args}. Execute it? (y/N): "
            )
            answer = answer.strip().lower()
            
            if answer not in {"y", "yes"}:
                logger.info(f"User declined execution of {tool_name}")
                from langchain_core.messages.tool import ToolMessage
                return ToolMessage(
                    content=f"Skipped execution of '{tool_name}' on user request.",
                    tool_call_id=f"tool_{uuid.uuid4()}"
                )
        elif tool_name in safe_tools:
            # Auto-approve safe tools like write_file
            logger.info(f"Auto-approving safe tool execution: {tool_name}")
        
        if tool_type == 'function':
            message = await FunctionTool.execute(self, tool_name, arguments)
        elif tool_type == 'mcp':
            message = await MCPTool.execute(self, tool_name, arguments, mcp_client, mcp_server_name)
        elif tool_type == 'module':
            message = await ModuleTool.execute(self, tool_name, arguments, module_path)
        return message
            
    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract first valid JSON object from text using stack-based parsing"""
        start = text.find("{")
        if start == -1:
            return None

        stack = []
        for i in range(start, len(text)):
            if text[i] == "{":
                stack.append("{")
            elif text[i] == "}":
                stack.pop()
                if not stack:
                    return text[start : i + 1]
        return None


class FunctionTool:
    @classmethod
    async def execute(cls,
            tool_manager: ToolManager,
            tool_name: str,
            arguments: Dict[str, Any]
            ):
        registered_functions = tool_manager.load_tools()

        if tool_name in tool_manager._registered_functions:
            try:
                func = tool_manager._registered_functions[tool_name]
                # artifact = await func(**arguments)
                artifact = await asyncio.to_thread(func, **arguments)
                content = f"Completed executing function tool {tool_name}({arguments})"
                logger.info(content)
                tool_call_id = registered_functions.get(tool_name, {}).get("tool_call_id")
                message = ToolMessage(
                    content=content, artifact=artifact, tool_call_id=tool_call_id
                )
                return message
            except Exception as e:
                content = f"Failed to execute function tool {tool_name}({arguments}): {str(e)}"
                logger.error(content)
                # raise {"error": content}
                return content


class MCPTool:
    @classmethod
    async def execute(cls,
            tool_manager: ToolManager,
            tool_name: str, 
            arguments: Dict[str, Any], 
            mcp_client: DistributedMCPClient,
            mcp_server_name: str):
        
        registered_functions = tool_manager.load_tools()
        """Call the MCP tool natively using the client session."""
        async with mcp_client.session(mcp_server_name) as session:
            payload = {
                "name": tool_name,
                "arguments": arguments
            }
            try:
                # Send the request to the MCP server
                # response = await session.call_tool(**payload)
                response = await session.call_tool(**payload)
                content = f"Completed executing mcp tool {tool_name}({arguments})"
                logger.info(content)
                tool_call_id = registered_functions.get(tool_name, {}).get("tool_call_id")
                artifact = response
                message = ToolMessage(
                    content=content, artifact=artifact, tool_call_id=tool_call_id
                )
                return message
            except Exception as e:
                content = f"Failed to execute mcp tool {tool_name}({arguments}): {str(e)}"
                logger.error(content)
                # raise {"error": content}
                return content


class ModuleTool:
    @classmethod
    async def execute(cls, 
            tool_manager: ToolManager,
            tool_name: str, 
            arguments: Dict[str, Any], 
            module_path: Union[str, Path], *arg, **kwargs):
        
        registered_functions = tool_manager.load_tools()
        try:
            if tool_name in globals():
                return globals()[tool_name](**arguments)

            module = importlib.import_module(module_path)
            func = getattr(module, tool_name)
            # artifact = await func(**arguments)
            artifact = await asyncio.to_thread(func, **arguments)
            content = f"Completed executing module tool {tool_name}({arguments})"
            logger.info(content)
            tool_call_id = registered_functions.get(tool_name, {}).get("tool_call_id")
            message = ToolMessage(
                content=content, artifact=artifact, tool_call_id=tool_call_id
            )
            return message
        except (ImportError, AttributeError) as e:
            content = f"Failed to execute module tool {tool_name}({arguments}): {str(e)}"
            logger.error(content)
            # raise {"error": content}
            return content
