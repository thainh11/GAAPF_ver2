"""
Tool registration and management system for GAAPF.

This module provides the ToolManager and GlobalToolRegistry classes
for loading, registering, and managing tools from JSON configurations.
"""

import json
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set
import re
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class GlobalToolRegistry:
    """Global registry for managing tool instances and metadata."""
    
    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict] = {}
        self._registered_modules: Set[str] = set()
    
    def register_tool(self, name: str, tool: Any, metadata: Dict = None):
        """Register a tool with optional metadata."""
        self._tools[name] = tool
        if metadata:
            self._metadata[name] = metadata
        logger.debug(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Any:
        """Get a registered tool by name."""
        return self._tools.get(name)
    
    def get_metadata(self, name: str) -> Dict:
        """Get metadata for a tool."""
        return self._metadata.get(name, {})
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def clear(self):
        """Clear all registered tools."""
        self._tools.clear()
        self._metadata.clear()
        self._registered_modules.clear()

    # Additional helpers used by callers
    def register_module(self, module_path: str) -> None:
        if module_path:
            self._registered_modules.add(module_path)

    def is_module_registered(self, module_path: str) -> bool:
        return module_path in self._registered_modules


class ToolManager:
    """Manages tool loading and registration from JSON configurations."""
    
    def __init__(self, tools_path: Union[Path, str] = None, is_reset_tools: bool = False):
        """
        Initialize the ToolManager.
        
        Args:
            tools_path: Path to the tools JSON configuration file
            is_reset_tools: Whether to reset/clear existing tools
        """
        self.tools_path = Path(tools_path) if tools_path else Path("templates/tools.json")
        self.registry = GlobalToolRegistry()
        self._loaded_tools: Dict[str, Any] = {}
        
        if is_reset_tools:
            self.registry.clear()
        
        # Load tools from JSON if file exists
        if self.tools_path.exists():
            try:
                self.load_tools_from_json()
                logger.info(f"✅ Loaded tools from {self.tools_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load tools from {self.tools_path}: {e}")
        else:
            logger.warning(f"⚠️ Tools file not found: {self.tools_path}")
    
    def load_tools_from_json(self):
        """Load tool configurations from JSON file."""
        try:
            with open(self.tools_path, 'r', encoding='utf-8') as f:
                tools_config = json.load(f)
            
            for tool_name, config in tools_config.items():
                try:
                    self._load_single_tool(tool_name, config)
                except Exception as e:
                    logger.error(f"❌ Failed to load tool {tool_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Failed to read tools configuration: {e}")
            raise
    
    def _load_single_tool(self, tool_name: str, config: Dict):
        """Load a single tool from its configuration."""
        tool_type = config.get('tool_type', 'module')
        module_path = config.get('module_path', '')
        # Respect runtime flag: skip tools not intended to load at runtime
        if config.get('is_runtime') is False:
            logger.debug(f"Skipping non-runtime tool: {tool_name}")
            return
        
        if tool_type == 'mcp':
            # Handle MCP tools (placeholder for now)
            self.registry.register_tool(tool_name, None, config)
            logger.debug(f"Registered MCP tool: {tool_name}")
            return
        
        if tool_type == 'module' and module_path:
            try:
                # Import the module and get the function
                if module_path.startswith('src.'):
                    # Handle src. prefixed imports
                    module_path = module_path[4:]  # Remove 'src.' prefix
                
                module = importlib.import_module(module_path)
                
                # Try to get the function by name
                tool_func = getattr(module, tool_name, None)
                if tool_func is None and '.' in tool_name:
                    # Support dotted names by traversing attributes
                    try:
                        obj = module
                        for part in tool_name.split('.'):  # e.g., FrameworkCollector.collect_framework_info
                            obj = getattr(obj, part)
                        tool_func = obj
                    except Exception:
                        tool_func = None
                if callable(tool_func):
                    self.registry.register_tool(tool_name, tool_func, config)
                    self._loaded_tools[tool_name] = tool_func
                    logger.debug(f"Loaded module tool: {tool_name} from {module_path}")
                else:
                    # Reduce noise: log at debug level if function isn't found; may be disabled or provided later
                    logger.debug(f"Function {tool_name} not found in module {module_path}")
                    
                # Mark module as registered for caching checks
                self.registry.register_module(module_path)
                
            except ImportError as e:
                logger.warning(f"⚠️ Could not import module {module_path} for tool {tool_name}: {e}")
            except Exception as e:
                logger.error(f"❌ Error loading tool {tool_name}: {e}")
    
    def get_tool(self, name: str) -> Any:
        """Get a tool by name."""
        return self.registry.get_tool(name)
    
    def get_loaded_tools(self) -> Dict[str, Any]:
        """Get all loaded tools."""
        return self._loaded_tools.copy()
    
    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return self.registry.list_tools()
    
    def register_tool(self, name: str, tool: Any, metadata: Dict = None):
        """Register a new tool."""
        self.registry.register_tool(name, tool, metadata)
        if callable(tool):
            self._loaded_tools[name] = tool
    
    def register_tools(self, tools: List[Union[str, BaseTool]]):
        """Register multiple tools."""
        for tool in tools:
            if isinstance(tool, str):
                # Tool name reference - try to load if not already loaded
                if tool not in self._loaded_tools:
                    logger.warning(f"⚠️ Tool '{tool}' not found in loaded tools")
            elif hasattr(tool, 'name'):
                # LangChain tool object
                self.register_tool(tool.name, tool)
            else:
                logger.warning(f"⚠️ Unknown tool type: {type(tool)}")

    # ------- Extended API expected by Agent/CLI -------
    def register_module_tool(self, tool_module_name: str, llm: Any = None) -> None:
        """
        Load and register all public callables from a module in src.GAAPF.core.tools.
        Accepts module short name like 'websearch_tools'.
        """
        # Prefer fully-qualified path from src, fallback without 'src.' for runtime imports
        candidate_modules = [
            f"src.GAAPF.core.tools.{tool_module_name}",
            f"GAAPF.core.tools.{tool_module_name}",
        ]
        last_error: Optional[Exception] = None
        for module_path in candidate_modules:
            try:
                import_path = module_path
                if module_path.startswith('src.'):
                    import_path = module_path[4:]
                module = importlib.import_module(import_path)
                # Register exported functions whose names do not start with underscore
                registered_any = False
                for attr_name in dir(module):
                    if attr_name.startswith('_'):
                        continue
                    attr = getattr(module, attr_name)
                    if callable(attr):
                        self.register_tool(attr_name, attr, metadata={
                            "tool_type": "module",
                            "module_path": module_path,
                        })
                        registered_any = True
                # Mark module registered regardless of which functions were found
                self.registry.register_module(import_path)
                if registered_any:
                    logger.info(f"🔧 Registered module tools from {module_path}")
                else:
                    logger.info(f"ℹ️ Module {module_path} imported but no public callables found to register")
                return
            except Exception as e:
                last_error = e
                continue
        raise AttributeError(f"Could not import tool module '{tool_module_name}': {last_error}")

    def register_function_tool(self, func: Any) -> None:
        """Register a single function as a tool."""
        if not callable(func):
            raise TypeError("register_function_tool expects a callable")
        name = getattr(func, "name", None) or getattr(func, "__name__", "anonymous_tool")
        self.register_tool(name, func, metadata={"tool_type": "function"})

    async def register_mcp_tool(self, mcp_client: Any, mcp_server_name: Optional[str] = None) -> List[Dict]:
        """Connect and load MCP tools (placeholder implementation)."""
        try:
            if hasattr(mcp_client, 'connect'):
                await mcp_client.connect(server_name=mcp_server_name)
            tools = []
            if hasattr(mcp_client, 'get_available_tools'):
                tools = await mcp_client.get_available_tools(server_name=mcp_server_name)
            # Just record that MCP integration is available
            self.register_tool("__mcp__", lambda **kwargs: None, metadata={"tool_type": "mcp"})
            return tools
        except Exception as e:
            logger.error(f"❌ Failed to register MCP tools: {e}")
            return []

    def extract_tool(self, content: Optional[str]) -> Optional[str]:
        """
        Try to extract a JSON object describing a tool call from an LLM response string.
        Returns a JSON string if found, else None.
        """
        if not content or not isinstance(content, str):
            return None
        # Quick path: content looks like a JSON object
        stripped = content.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            return stripped
        # Regex to find a JSON object that contains required keys
        try:
            pattern = re.compile(r"\{[\s\S]*?\}")
            for match in pattern.finditer(content):
                candidate = match.group(0)
                if '"tool_name"' in candidate and '"tool_type"' in candidate:
                    # Heuristic: ensure it parses
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    async def _execute_tool(
        self,
        tool_name: Optional[str],
        tool_type: Optional[str],
        arguments: Optional[Dict[str, Any]] = None,
        module_path: Optional[str] = None,
        mcp_client: Any = None,
        mcp_server_name: Optional[str] = None,
    ) -> Any:
        """
        Execute a tool call and return a message-like object with a 'content' field.
        """
        arguments = arguments or {}
        try:
            if tool_type == 'mcp':
                if mcp_client is None:
                    return AIMessage(content=f"MCP tool {tool_name} requested but no MCP client available.")
                result = await mcp_client.call_tool(tool_name, arguments, server_name=mcp_server_name)
                return AIMessage(content=json.dumps(result, ensure_ascii=False))
            elif tool_type == 'function':
                func = self.registry.get_tool(tool_name)
                if not callable(func):
                    return AIMessage(content=f"Function tool '{tool_name}' is not registered.")
                result = func(**arguments)
                return AIMessage(content=json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result)
            elif tool_type == 'module':
                # Import specified module to find the function
                import_path = module_path or ''
                if import_path.startswith('src.'):
                    import_path = import_path[4:]
                if not import_path:
                    # Fallback: try known tools namespace
                    import_path = f"GAAPF.core.tools.{tool_name}"
                module = importlib.import_module(import_path)
                if not hasattr(module, tool_name) and 'tool_name' in arguments:
                    # No-op; but we expect function by tool_name
                    pass
                if hasattr(module, tool_name):
                    func = getattr(module, tool_name)
                else:
                    # If 'tool_name' is something like 'search_web', use that
                    func = getattr(module, arguments.get('tool_name', ''), None)
                if func is None:
                    # Try direct attribute by provided tool_name parameter
                    func = getattr(module, tool_name or '', None)
                if not callable(func):
                    return AIMessage(content=f"Tool function '{tool_name}' not found in module '{module_path or import_path}'.")
                result = func(**arguments)
                return AIMessage(content=json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result)
            else:
                return AIMessage(content=f"Unknown tool type: {tool_type}")
        except Exception as e:
            logger.error(f"❌ Tool execution failed for {tool_name}: {e}")
            return AIMessage(content=f"Tool execution error for {tool_name}: {str(e)}")
    
    def save_tools_to_json(self, output_path: Union[Path, str] = None):
        """Save current tool configurations to JSON file."""
        output_path = Path(output_path) if output_path else self.tools_path
        
        tools_config = {}
        for tool_name in self.registry.list_tools():
            metadata = self.registry.get_metadata(tool_name)
            if metadata:
                tools_config[tool_name] = metadata
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tools_config, f, indent=4, ensure_ascii=False)
            logger.info(f"✅ Saved tools configuration to {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save tools configuration: {e}")
            raise


# Global instance for easy access
global_tool_registry = GlobalToolRegistry()