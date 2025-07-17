"""Agent Registry for GAAPF.

This module provides a centralized registry for agent types,
making it easier to add new agents without relying on strict naming conventions.
"""

import logging
from typing import Dict, Type, List, Optional, Any
from pathlib import Path
import importlib
import inspect

from . import SpecializedAgent
from ...utils.exceptions import AgentCreationError, AgentNotFoundError

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Registry for managing agent types and their creation."""
    
    def __init__(self):
        """Initialize the agent registry."""
        self._agents: Dict[str, Type[SpecializedAgent]] = {}
        self._agent_metadata: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
    
    def register_agent(
        self,
        agent_type: str,
        agent_class: Type[SpecializedAgent],
        description: str = None,
        capabilities: List[str] = None,
        priority: int = 0
    ):
        """Register an agent type.
        
        Args:
            agent_type: Unique identifier for the agent type
            agent_class: The agent class to register
            description: Human-readable description of the agent
            capabilities: List of agent capabilities
            priority: Priority for agent selection (higher = more preferred)
        """
        if not issubclass(agent_class, SpecializedAgent):
            raise ValueError(f"Agent class {agent_class} must inherit from SpecializedAgent")
        
        self._agents[agent_type] = agent_class
        self._agent_metadata[agent_type] = {
            "description": description or f"Specialized agent for {agent_type}",
            "capabilities": capabilities or [],
            "priority": priority,
            "class_name": agent_class.__name__,
            "module": agent_class.__module__
        }
        
        logger.info(f"Registered agent type '{agent_type}' with class {agent_class.__name__}")
    
    def unregister_agent(self, agent_type: str):
        """Unregister an agent type.
        
        Args:
            agent_type: The agent type to unregister
        """
        if agent_type in self._agents:
            del self._agents[agent_type]
            del self._agent_metadata[agent_type]
            logger.info(f"Unregistered agent type '{agent_type}'")
    
    def get_agent_class(self, agent_type: str) -> Type[SpecializedAgent]:
        """Get the agent class for a given type.
        
        Args:
            agent_type: The agent type to look up
            
        Returns:
            The agent class
            
        Raises:
            AgentNotFoundError: If the agent type is not registered
        """
        if agent_type not in self._agents:
            available_agents = list(self._agents.keys())
            raise AgentNotFoundError(agent_type, available_agents)
        
        return self._agents[agent_type]
    
    def create_agent(
        self,
        agent_type: str,
        llm,
        memory_path: Optional[Path] = None,
        is_logging: bool = False,
        **kwargs
    ) -> SpecializedAgent:
        """Create an agent instance.
        
        Args:
            agent_type: The type of agent to create
            llm: Language model instance
            memory_path: Path for agent memory
            is_logging: Whether to enable logging
            **kwargs: Additional arguments for agent initialization
            
        Returns:
            Created agent instance
            
        Raises:
            AgentCreationError: If agent creation fails
        """
        try:
            agent_class = self.get_agent_class(agent_type)
            
            # Create agent instance
            agent = agent_class(
                llm=llm,
                memory_path=memory_path,
                is_logging=is_logging,
                **kwargs
            )
            
            logger.info(f"Successfully created {agent_type} agent")
            return agent
            
        except AgentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to create {agent_type} agent: {e}")
            raise AgentCreationError(agent_type, str(e), e)
    
    def list_agents(self) -> List[str]:
        """List all registered agent types.
        
        Returns:
            List of registered agent type names
        """
        return list(self._agents.keys())
    
    def get_agent_metadata(self, agent_type: str) -> Dict[str, Any]:
        """Get metadata for an agent type.
        
        Args:
            agent_type: The agent type to get metadata for
            
        Returns:
            Agent metadata dictionary
            
        Raises:
            AgentNotFoundError: If the agent type is not registered
        """
        if agent_type not in self._agent_metadata:
            available_agents = list(self._agents.keys())
            raise AgentNotFoundError(agent_type, available_agents)
        
        return self._agent_metadata[agent_type].copy()
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """Get agent types that have a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of agent types with the specified capability
        """
        matching_agents = []
        for agent_type, metadata in self._agent_metadata.items():
            if capability in metadata.get("capabilities", []):
                matching_agents.append(agent_type)
        
        # Sort by priority (higher priority first)
        matching_agents.sort(
            key=lambda x: self._agent_metadata[x].get("priority", 0),
            reverse=True
        )
        
        return matching_agents
    
    def auto_discover_agents(self, agents_package: str = "src.GAAPF.core.agents"):
        """Automatically discover and register agents from a package.
        
        Args:
            agents_package: Package path to search for agents
        """
        try:
            # Import the agents package
            package = importlib.import_module(agents_package)
            package_path = Path(package.__file__).parent
            
            # Scan for Python files in the package
            for py_file in package_path.glob("*.py"):
                if py_file.name.startswith("__") or py_file.name == "registry.py":
                    continue
                
                module_name = py_file.stem
                try:
                    # Import the module
                    full_module_name = f"{agents_package}.{module_name}"
                    module = importlib.import_module(full_module_name)
                    
                    # Look for agent classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, SpecializedAgent) and 
                            obj != SpecializedAgent and
                            obj.__module__ == full_module_name):
                            
                            # Extract agent type from class name or module name
                            agent_type = self._extract_agent_type(name, module_name)
                            
                            # Get capabilities from class if available
                            capabilities = getattr(obj, 'CAPABILITIES', [])
                            description = getattr(obj, 'DESCRIPTION', None)
                            priority = getattr(obj, 'PRIORITY', 0)
                            
                            # Register the agent
                            if agent_type not in self._agents:
                                self.register_agent(
                                    agent_type=agent_type,
                                    agent_class=obj,
                                    description=description,
                                    capabilities=capabilities,
                                    priority=priority
                                )
                            
                except Exception as e:
                    logger.warning(f"Failed to import agent module {module_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to auto-discover agents: {e}")
    
    def _extract_agent_type(self, class_name: str, module_name: str) -> str:
        """Extract agent type from class name or module name.
        
        Args:
            class_name: Name of the agent class
            module_name: Name of the module
            
        Returns:
            Extracted agent type
        """
        # Try to extract from class name first
        if class_name.endswith("Agent"):
            # Remove "Agent" suffix and convert to snake_case
            base_name = class_name[:-5]  # Remove "Agent"
            agent_type = self._camel_to_snake(base_name)
        else:
            # Use module name as fallback
            agent_type = module_name
        
        return agent_type
    
    def _camel_to_snake(self, name: str) -> str:
        """Convert CamelCase to snake_case.
        
        Args:
            name: CamelCase string
            
        Returns:
            snake_case string
        """
        import re
        # Insert underscore before uppercase letters that follow lowercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        # Insert underscore before uppercase letters that follow lowercase letters or digits
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def initialize_default_agents(self):
        """Initialize registry with default agents."""
        if self._initialized:
            return
        
        # Auto-discover agents from the agents package
        self.auto_discover_agents()
        
        # Mark as initialized
        self._initialized = True
        
        logger.info(f"Initialized agent registry with {len(self._agents)} agents")
    
    def is_agent_available(self, agent_type: str) -> bool:
        """Check if an agent type is available.
        
        Args:
            agent_type: The agent type to check
            
        Returns:
            True if the agent type is registered, False otherwise
        """
        return agent_type in self._agents
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get information about the registry.
        
        Returns:
            Dictionary with registry information
        """
        return {
            "total_agents": len(self._agents),
            "agent_types": list(self._agents.keys()),
            "agent_metadata": self._agent_metadata.copy(),
            "initialized": self._initialized
        }

# Global registry instance
agent_registry = AgentRegistry()

# Decorator for easy agent registration
def register_agent(
    agent_type: str = None,
    description: str = None,
    capabilities: List[str] = None,
    priority: int = 0
):
    """Decorator to register an agent class.
    
    Args:
        agent_type: Agent type identifier (auto-generated if not provided)
        description: Agent description
        capabilities: List of agent capabilities
        priority: Agent priority
    """
    def decorator(cls):
        # Auto-generate agent type if not provided
        if agent_type is None:
            extracted_type = agent_registry._extract_agent_type(cls.__name__, cls.__module__.split('.')[-1])
        else:
            extracted_type = agent_type
        
        # Register the agent
        agent_registry.register_agent(
            agent_type=extracted_type,
            agent_class=cls,
            description=description,
            capabilities=capabilities,
            priority=priority
        )
        
        return cls
    
    return decorator