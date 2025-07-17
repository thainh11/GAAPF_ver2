"""GAAPF Adapter for RL System Integration

This module provides the adapter layer that enables seamless integration
between the RL system and the existing GAAPF framework.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import json
import inspect
from abc import ABC, abstractmethod

# RL System imports
from ..agents.rl_specialized_agent import RLSpecializedAgent
from ..managers.rl_constellation_manager import RLConstellationManager
from ..utils.reward_system import RewardCalculator
from ..utils.experience_buffer import ExperienceBuffer

logger = logging.getLogger(__name__)

class AdapterMode(Enum):
    """Adapter operation modes"""
    PASSTHROUGH = "passthrough"  # Direct passthrough to original GAAPF
    ENHANCED = "enhanced"        # RL-enhanced processing
    HYBRID = "hybrid"            # Mix of both approaches
    COMPARISON = "comparison"    # Run both for comparison

class AgentType(Enum):
    """Types of agents in GAAPF"""
    SPECIALIZED = "specialized"
    ASSESSMENT = "assessment"
    CODE_ASSISTANT = "code_assistant"
    INSTRUCTOR = "instructor"
    PROGRESS_TRACKER = "progress_tracker"
    MOTIVATIONAL_COACH = "motivational_coach"
    TROUBLESHOOTER = "troubleshooter"
    CUSTOM = "custom"

@dataclass
class AdapterConfig:
    """Configuration for GAAPF adapter"""
    # Operation mode
    mode: AdapterMode = AdapterMode.ENHANCED
    
    # Agent mapping
    enable_rl_agents: bool = True
    fallback_to_original: bool = True
    agent_timeout_seconds: float = 30.0
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    max_concurrent_requests: int = 10
    
    # Integration settings
    preserve_original_interface: bool = True
    enable_middleware: bool = True
    enable_hooks: bool = True
    
    # Monitoring
    enable_performance_tracking: bool = True
    enable_error_tracking: bool = True
    detailed_logging: bool = False
    
    # Safety settings
    enable_input_validation: bool = True
    enable_output_validation: bool = True
    max_retries: int = 3

@dataclass
class RequestContext:
    """Context for processing requests"""
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_type: Optional[AgentType] = None
    task_type: Optional[str] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'request_id': self.request_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'agent_type': self.agent_type.value if self.agent_type else None,
            'task_type': self.task_type,
            'priority': self.priority,
            'metadata': self.metadata,
            'start_time': self.start_time
        }

@dataclass
class ProcessingResult:
    """Result of request processing"""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    processing_time: float = 0.0
    system_used: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'result': self.result,
            'error': str(self.error) if self.error else None,
            'processing_time': self.processing_time,
            'system_used': self.system_used,
            'metadata': self.metadata
        }

class AgentWrapper(ABC):
    """Abstract base class for agent wrappers"""
    
    def __init__(self, original_agent: Any, rl_agent: Optional[RLSpecializedAgent] = None):
        self.original_agent = original_agent
        self.rl_agent = rl_agent
        self.agent_id = str(uuid.uuid4())
        self.created_at = time.time()
        self.request_count = 0
    
    @abstractmethod
    async def process(self, context: RequestContext, *args, **kwargs) -> ProcessingResult:
        """Process request using appropriate agent"""
        pass
    
    @abstractmethod
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        pass

class SpecializedAgentWrapper(AgentWrapper):
    """Wrapper for specialized agents"""
    
    async def process(self, context: RequestContext, *args, **kwargs) -> ProcessingResult:
        """Process request using specialized agent"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Determine which agent to use
            if self.rl_agent and context.metadata.get('use_rl', True):
                # Use RL-enhanced agent
                result = await self._process_with_rl_agent(context, *args, **kwargs)
                system_used = "rl_enhanced"
            else:
                # Use original agent
                result = await self._process_with_original_agent(context, *args, **kwargs)
                system_used = "traditional"
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                result=result,
                processing_time=processing_time,
                system_used=system_used,
                metadata={
                    'agent_id': self.agent_id,
                    'request_count': self.request_count,
                    'context': context.to_dict()
                }
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing request {context.request_id}: {e}")
            
            return ProcessingResult(
                success=False,
                error=e,
                processing_time=processing_time,
                system_used="error",
                metadata={
                    'agent_id': self.agent_id,
                    'request_count': self.request_count,
                    'context': context.to_dict()
                }
            )
    
    async def _process_with_rl_agent(self, context: RequestContext, *args, **kwargs) -> Any:
        """Process using RL-enhanced agent"""
        if not self.rl_agent:
            raise ValueError("RL agent not available")
        
        # Convert arguments to format expected by RL agent
        rl_kwargs = self._convert_args_for_rl_agent(context, *args, **kwargs)
        
        # Process with RL agent
        result = await self.rl_agent.ainvoke(**rl_kwargs)
        
        return result
    
    async def _process_with_original_agent(self, context: RequestContext, *args, **kwargs) -> Any:
        """Process using original agent"""
        if not self.original_agent:
            raise ValueError("Original agent not available")
        
        # Check if original agent has async invoke method
        if hasattr(self.original_agent, 'ainvoke'):
            result = await self.original_agent.ainvoke(*args, **kwargs)
        elif hasattr(self.original_agent, 'invoke'):
            # Run sync method in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.original_agent.invoke, *args, **kwargs)
        else:
            # Try calling the agent directly
            if asyncio.iscoroutinefunction(self.original_agent):
                result = await self.original_agent(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.original_agent, *args, **kwargs)
        
        return result
    
    def _convert_args_for_rl_agent(self, context: RequestContext, *args, **kwargs) -> Dict[str, Any]:
        """Convert arguments for RL agent processing"""
        # Extract common parameters
        rl_kwargs = {
            'user_id': context.user_id,
            'session_id': context.session_id,
            'task_type': context.task_type,
            'metadata': context.metadata
        }
        
        # Add original arguments
        rl_kwargs.update(kwargs)
        
        # Handle positional arguments
        if args:
            # Common patterns in GAAPF
            if len(args) >= 1:
                rl_kwargs['input'] = args[0]
            if len(args) >= 2:
                rl_kwargs['config'] = args[1]
        
        return rl_kwargs
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'agent_type': 'specialized',
            'has_rl_agent': self.rl_agent is not None,
            'has_original_agent': self.original_agent is not None,
            'created_at': self.created_at,
            'request_count': self.request_count,
            'original_agent_type': type(self.original_agent).__name__ if self.original_agent else None,
            'rl_agent_type': type(self.rl_agent).__name__ if self.rl_agent else None
        }

class ConstellationWrapper:
    """Wrapper for constellation management"""
    
    def __init__(self, original_manager: Any, rl_manager: Optional[RLConstellationManager] = None):
        self.original_manager = original_manager
        self.rl_manager = rl_manager
        self.manager_id = str(uuid.uuid4())
        self.created_at = time.time()
        self.formation_count = 0
    
    async def form_constellation(self, context: RequestContext, *args, **kwargs) -> ProcessingResult:
        """Form agent constellation"""
        start_time = time.time()
        self.formation_count += 1
        
        try:
            # Determine which manager to use
            if self.rl_manager and context.metadata.get('use_rl', True):
                # Use RL-enhanced constellation manager
                result = await self._form_with_rl_manager(context, *args, **kwargs)
                system_used = "rl_enhanced"
            else:
                # Use original constellation manager
                result = await self._form_with_original_manager(context, *args, **kwargs)
                system_used = "traditional"
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                result=result,
                processing_time=processing_time,
                system_used=system_used,
                metadata={
                    'manager_id': self.manager_id,
                    'formation_count': self.formation_count,
                    'context': context.to_dict()
                }
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error forming constellation for request {context.request_id}: {e}")
            
            return ProcessingResult(
                success=False,
                error=e,
                processing_time=processing_time,
                system_used="error",
                metadata={
                    'manager_id': self.manager_id,
                    'formation_count': self.formation_count,
                    'context': context.to_dict()
                }
            )
    
    async def _form_with_rl_manager(self, context: RequestContext, *args, **kwargs) -> Any:
        """Form constellation using RL manager"""
        if not self.rl_manager:
            raise ValueError("RL constellation manager not available")
        
        # Convert arguments for RL manager
        rl_kwargs = self._convert_args_for_rl_manager(context, *args, **kwargs)
        
        # Form constellation with RL manager
        result = await self.rl_manager.form_constellation(**rl_kwargs)
        
        return result
    
    async def _form_with_original_manager(self, context: RequestContext, *args, **kwargs) -> Any:
        """Form constellation using original manager"""
        if not self.original_manager:
            raise ValueError("Original constellation manager not available")
        
        # Check if original manager has async methods
        if hasattr(self.original_manager, 'form_constellation'):
            method = getattr(self.original_manager, 'form_constellation')
            if asyncio.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, method, *args, **kwargs)
        else:
            # Try calling the manager directly
            if asyncio.iscoroutinefunction(self.original_manager):
                result = await self.original_manager(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.original_manager, *args, **kwargs)
        
        return result
    
    def _convert_args_for_rl_manager(self, context: RequestContext, *args, **kwargs) -> Dict[str, Any]:
        """Convert arguments for RL constellation manager"""
        rl_kwargs = {
            'user_id': context.user_id,
            'session_id': context.session_id,
            'task_type': context.task_type,
            'metadata': context.metadata
        }
        
        # Add original arguments
        rl_kwargs.update(kwargs)
        
        # Handle positional arguments
        if args:
            if len(args) >= 1:
                rl_kwargs['task_requirements'] = args[0]
            if len(args) >= 2:
                rl_kwargs['agent_preferences'] = args[1]
        
        return rl_kwargs
    
    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'manager_id': self.manager_id,
            'manager_type': 'constellation',
            'has_rl_manager': self.rl_manager is not None,
            'has_original_manager': self.original_manager is not None,
            'created_at': self.created_at,
            'formation_count': self.formation_count,
            'original_manager_type': type(self.original_manager).__name__ if self.original_manager else None,
            'rl_manager_type': type(self.rl_manager).__name__ if self.rl_manager else None
        }

class GAAPFAdapter:
    """Main adapter for GAAPF integration"""
    
    def __init__(self, config: AdapterConfig):
        """
        Initialize GAAPF adapter.
        
        Parameters:
        ----------
        config : AdapterConfig
            Adapter configuration
        """
        self.config = config
        self.adapter_id = str(uuid.uuid4())
        self.created_at = time.time()
        
        # Component registries
        self.agent_wrappers: Dict[str, AgentWrapper] = {}
        self.constellation_wrappers: Dict[str, ConstellationWrapper] = {}
        self.middleware_stack: List[Callable] = []
        self.hooks: Dict[str, List[Callable]] = {
            'before_request': [],
            'after_request': [],
            'on_error': [],
            'on_success': []
        }
        
        # Performance tracking
        self.request_cache: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0
        }
        
        # RL components
        self.reward_calculator = None
        self.experience_buffer = None
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        logger.info(f"Initialized GAAPF adapter {self.adapter_id}")
    
    def register_agent(self, agent_type: AgentType, original_agent: Any, 
                      rl_agent: Optional[RLSpecializedAgent] = None) -> str:
        """Register an agent with the adapter"""
        wrapper_id = str(uuid.uuid4())
        
        # Create appropriate wrapper based on agent type
        if agent_type == AgentType.SPECIALIZED:
            wrapper = SpecializedAgentWrapper(original_agent, rl_agent)
        else:
            # For other agent types, use the base specialized wrapper for now
            wrapper = SpecializedAgentWrapper(original_agent, rl_agent)
        
        self.agent_wrappers[wrapper_id] = wrapper
        
        logger.info(f"Registered {agent_type.value} agent with ID {wrapper_id}")
        return wrapper_id
    
    def register_constellation_manager(self, original_manager: Any, 
                                     rl_manager: Optional[RLConstellationManager] = None) -> str:
        """Register a constellation manager with the adapter"""
        wrapper_id = str(uuid.uuid4())
        wrapper = ConstellationWrapper(original_manager, rl_manager)
        
        self.constellation_wrappers[wrapper_id] = wrapper
        
        logger.info(f"Registered constellation manager with ID {wrapper_id}")
        return wrapper_id
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware to the processing stack"""
        self.middleware_stack.append(middleware)
        logger.debug(f"Added middleware: {middleware.__name__}")
    
    def add_hook(self, hook_type: str, hook_function: Callable) -> None:
        """Add a hook function"""
        if hook_type in self.hooks:
            self.hooks[hook_type].append(hook_function)
            logger.debug(f"Added {hook_type} hook: {hook_function.__name__}")
        else:
            logger.warning(f"Unknown hook type: {hook_type}")
    
    async def process_agent_request(self, agent_id: str, context: RequestContext, 
                                  *args, **kwargs) -> ProcessingResult:
        """Process a request through an agent"""
        async with self.semaphore:
            return await self._process_agent_request_internal(agent_id, context, *args, **kwargs)
    
    async def _process_agent_request_internal(self, agent_id: str, context: RequestContext,
                                            *args, **kwargs) -> ProcessingResult:
        """Internal agent request processing"""
        start_time = time.time()
        self.performance_metrics['total_requests'] += 1
        
        try:
            # Execute before_request hooks
            if self.config.enable_hooks:
                await self._execute_hooks('before_request', context, *args, **kwargs)
            
            # Check cache
            cache_key = None
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(agent_id, context, *args, **kwargs)
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.performance_metrics['cache_hits'] += 1
                    return cached_result
                else:
                    self.performance_metrics['cache_misses'] += 1
            
            # Get agent wrapper
            if agent_id not in self.agent_wrappers:
                raise ValueError(f"Agent {agent_id} not found")
            
            wrapper = self.agent_wrappers[agent_id]
            
            # Apply middleware
            if self.config.enable_middleware:
                for middleware in self.middleware_stack:
                    context, args, kwargs = await self._apply_middleware(
                        middleware, context, *args, **kwargs
                    )
            
            # Process request
            result = await wrapper.process(context, *args, **kwargs)
            
            # Cache result if successful
            if result.success and cache_key and self.config.enable_caching:
                self._store_in_cache(cache_key, result)
            
            # Execute hooks
            if self.config.enable_hooks:
                if result.success:
                    await self._execute_hooks('on_success', context, result)
                    self.performance_metrics['successful_requests'] += 1
                else:
                    await self._execute_hooks('on_error', context, result)
                    self.performance_metrics['failed_requests'] += 1
                
                await self._execute_hooks('after_request', context, result)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time)
            
            return result
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.performance_metrics['failed_requests'] += 1
            
            error_result = ProcessingResult(
                success=False,
                error=e,
                processing_time=processing_time,
                system_used="adapter_error"
            )
            
            # Execute error hooks
            if self.config.enable_hooks:
                await self._execute_hooks('on_error', context, error_result)
                await self._execute_hooks('after_request', context, error_result)
            
            logger.error(f"Error processing agent request {context.request_id}: {e}")
            return error_result
    
    async def process_constellation_request(self, manager_id: str, context: RequestContext,
                                          *args, **kwargs) -> ProcessingResult:
        """Process a constellation formation request"""
        async with self.semaphore:
            return await self._process_constellation_request_internal(manager_id, context, *args, **kwargs)
    
    async def _process_constellation_request_internal(self, manager_id: str, context: RequestContext,
                                                    *args, **kwargs) -> ProcessingResult:
        """Internal constellation request processing"""
        start_time = time.time()
        self.performance_metrics['total_requests'] += 1
        
        try:
            # Execute before_request hooks
            if self.config.enable_hooks:
                await self._execute_hooks('before_request', context, *args, **kwargs)
            
            # Get constellation wrapper
            if manager_id not in self.constellation_wrappers:
                raise ValueError(f"Constellation manager {manager_id} not found")
            
            wrapper = self.constellation_wrappers[manager_id]
            
            # Apply middleware
            if self.config.enable_middleware:
                for middleware in self.middleware_stack:
                    context, args, kwargs = await self._apply_middleware(
                        middleware, context, *args, **kwargs
                    )
            
            # Process request
            result = await wrapper.form_constellation(context, *args, **kwargs)
            
            # Execute hooks
            if self.config.enable_hooks:
                if result.success:
                    await self._execute_hooks('on_success', context, result)
                    self.performance_metrics['successful_requests'] += 1
                else:
                    await self._execute_hooks('on_error', context, result)
                    self.performance_metrics['failed_requests'] += 1
                
                await self._execute_hooks('after_request', context, result)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time)
            
            return result
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.performance_metrics['failed_requests'] += 1
            
            error_result = ProcessingResult(
                success=False,
                error=e,
                processing_time=processing_time,
                system_used="adapter_error"
            )
            
            # Execute error hooks
            if self.config.enable_hooks:
                await self._execute_hooks('on_error', context, error_result)
                await self._execute_hooks('after_request', context, error_result)
            
            logger.error(f"Error processing constellation request {context.request_id}: {e}")
            return error_result
    
    async def _execute_hooks(self, hook_type: str, *args, **kwargs) -> None:
        """Execute hooks of specified type"""
        if hook_type in self.hooks:
            for hook in self.hooks[hook_type]:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(*args, **kwargs)
                    else:
                        hook(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Hook {hook.__name__} failed: {e}")
    
    async def _apply_middleware(self, middleware: Callable, context: RequestContext,
                              *args, **kwargs) -> Tuple[RequestContext, tuple, dict]:
        """Apply middleware to request"""
        try:
            if asyncio.iscoroutinefunction(middleware):
                result = await middleware(context, *args, **kwargs)
            else:
                result = middleware(context, *args, **kwargs)
            
            # Middleware can return modified context, args, kwargs
            if isinstance(result, tuple) and len(result) == 3:
                return result
            else:
                return context, args, kwargs
        
        except Exception as e:
            logger.warning(f"Middleware {middleware.__name__} failed: {e}")
            return context, args, kwargs
    
    def _generate_cache_key(self, component_id: str, context: RequestContext,
                           *args, **kwargs) -> str:
        """Generate cache key for request"""
        # Create a simple hash-based cache key
        import hashlib
        
        key_data = {
            'component_id': component_id,
            'user_id': context.user_id,
            'session_id': context.session_id,
            'task_type': context.task_type,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[ProcessingResult]:
        """Get result from cache"""
        if cache_key in self.request_cache:
            cached_item = self.request_cache[cache_key]
            
            # Check if cache item is still valid
            if time.time() - cached_item['timestamp'] < self.config.cache_ttl_seconds:
                return cached_item['result']
            else:
                # Remove expired item
                del self.request_cache[cache_key]
        
        return None
    
    def _store_in_cache(self, cache_key: str, result: ProcessingResult) -> None:
        """Store result in cache"""
        self.request_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Simple cache cleanup - remove oldest items if cache is too large
        if len(self.request_cache) > 1000:
            # Remove oldest 100 items
            oldest_keys = sorted(
                self.request_cache.keys(),
                key=lambda k: self.request_cache[k]['timestamp']
            )[:100]
            
            for key in oldest_keys:
                del self.request_cache[key]
    
    def _update_performance_metrics(self, processing_time: float) -> None:
        """Update performance metrics"""
        # Update average response time
        total_requests = self.performance_metrics['total_requests']
        current_avg = self.performance_metrics['avg_response_time']
        
        # Calculate new average
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.performance_metrics['avg_response_time'] = new_avg
    
    def get_adapter_status(self) -> Dict[str, Any]:
        """Get adapter status and metrics"""
        return {
            'adapter_id': self.adapter_id,
            'created_at': self.created_at,
            'config': {
                'mode': self.config.mode.value,
                'enable_rl_agents': self.config.enable_rl_agents,
                'enable_caching': self.config.enable_caching,
                'max_concurrent_requests': self.config.max_concurrent_requests
            },
            'components': {
                'agent_wrappers': len(self.agent_wrappers),
                'constellation_wrappers': len(self.constellation_wrappers),
                'middleware_count': len(self.middleware_stack),
                'hooks_count': sum(len(hooks) for hooks in self.hooks.values())
            },
            'performance_metrics': self.performance_metrics.copy(),
            'cache_stats': {
                'cache_size': len(self.request_cache),
                'cache_hit_rate': (
                    self.performance_metrics['cache_hits'] / 
                    max(1, self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
                ) * 100
            }
        }
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent"""
        if agent_id in self.agent_wrappers:
            return self.agent_wrappers[agent_id].get_agent_info()
        return None
    
    def get_constellation_info(self, manager_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific constellation manager"""
        if manager_id in self.constellation_wrappers:
            return self.constellation_wrappers[manager_id].get_manager_info()
        return None
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [
            {'id': agent_id, **wrapper.get_agent_info()}
            for agent_id, wrapper in self.agent_wrappers.items()
        ]
    
    def list_constellation_managers(self) -> List[Dict[str, Any]]:
        """List all registered constellation managers"""
        return [
            {'id': manager_id, **wrapper.get_manager_info()}
            for manager_id, wrapper in self.constellation_wrappers.items()
        ]
    
    def cleanup(self) -> None:
        """Cleanup adapter resources"""
        logger.info(f"Cleaning up GAAPF adapter {self.adapter_id}")
        
        # Clear caches
        self.request_cache.clear()
        
        # Clear registries
        self.agent_wrappers.clear()
        self.constellation_wrappers.clear()
        self.middleware_stack.clear()
        
        for hook_list in self.hooks.values():
            hook_list.clear()
        
        logger.info("GAAPF adapter cleanup complete")