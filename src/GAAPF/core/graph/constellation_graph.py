"""
Constellation Graph for GAAPF Architecture

This module provides the ConstellationGraph class that manages
agent handoffs and coordination within a constellation.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Annotated, TypedDict
import copy
import asyncio

from ..agents import SpecializedAgent
from ..utils.async_helpers import run_sync
from .graph import Graph
from .node import Node
from .operator import FlowStateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from ...utils.exceptions import (
    GaapfException, ConstellationException, HandoffException,
    ErrorSeverity, ErrorCategory, RecoveryStrategy,
    global_error_handler
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define reducer functions for state updates with memory management
def append_handoff_history(existing: List, update: Dict) -> List:
    """Append a handoff entry to the handoff history with size limit."""
    if existing is None:
        existing = []
    # Limit handoff history to prevent memory leak (max 10 entries)
    result = existing + [update]
    if len(result) > 10:
        result = result[-10:]  # Keep only last 10 entries
    return result

def update_agent_responses(existing: Dict, update: Dict) -> Dict:
    """Update the agent responses dictionary with memory optimization."""
    if existing is None:
        existing = {}
    result = copy.deepcopy(existing)
    result.update(update)
    # Limit agent responses to prevent memory leak (max 5 agents)
    if len(result) > 5:
        # Keep only the most recent 5 agent responses
        sorted_keys = sorted(result.keys())
        for key in sorted_keys[:-5]:
            del result[key]
    return result

# Define the state schema for the constellation graph with memory management
class ConstellationState(TypedDict):
    """State schema for the constellation graph with memory optimization."""
    user_id: str
    interaction_data: Dict
    learning_context: Dict
    primary_agent: str
    agent_responses: Annotated[Dict, update_agent_responses]
    current_agent: str
    handoff_history: Annotated[List, append_handoff_history]
    final_response: Optional[Dict]
    handoff_needed: bool
    handoff_to: Optional[str]
    handoff_reason: Optional[str]
    # Add memory management fields
    messages: Optional[List]  # Track messages with cleanup
    interaction_count: int  # Track interaction count for cleanup
    max_handoffs: int  # Dynamic handoff limit
    # Error handling and recovery
    errors: List[Dict[str, Any]]
    error_count: int
    recovery_attempts: int
    last_error_time: Optional[float]
    performance_metrics: Dict[str, Any]

# Optional config schema
class ConfigSchema(TypedDict):
    """Configuration schema for the constellation graph."""
    thread_id: str

class ConstellationGraph:
    """
    Specialized graph for agent constellation coordination.
    
    The ConstellationGraph manages agent handoffs and learning flows
    within a constellation of specialized agents.
    """
    
    def __init__(
        self,
        agents: Dict[str, SpecializedAgent],
        constellation_type: str,
        constellation_config: Dict,
        is_logging: bool = False,
        *args, **kwargs
    ):
        """
        Initialize a ConstellationGraph for agent coordination.
        
        Parameters:
        ----------
        agents : Dict[str, SpecializedAgent]
            Dictionary of agent instances by type
        constellation_type : str
            Type of constellation
        constellation_config : Dict
            Configuration for this constellation
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        self.agents = agents
        self.constellation_type = constellation_type
        self.constellation_config = constellation_config
        self.is_logging = is_logging
        
        # Initialize the graph components
        self.checkpoint = MemorySaver()
        self.graph = FlowStateGraph(ConstellationState, config_schema=ConfigSchema)
        
        # Create all nodes
        self.agent_nodes = {}
        self.router_node = self._create_router_node()
        self.aggregator_node = self._create_aggregator_node()
        self.handoff_node = self._create_handoff_node()
        self.termination_node = self._create_termination_node()
        
        # Create agent nodes
        for agent_type, agent in self.agents.items():
            self.agent_nodes[agent_type] = self._create_agent_node(agent_type, agent)
        
        # Build the flow
        self.flow = self._build_flow()
        
        # Compile the graph
        self.compiled_graph = self.graph.compile(
            checkpointer=self.checkpoint,
            flow=self.flow
        )
        
        if self.is_logging:
            logger.info(f"Initialized ConstellationGraph for {constellation_type} constellation with compiled LangGraph workflow")
    
    def process(
        self,
        user_id: str,
        interaction_data: Dict,
        learning_context: Dict,
        primary_agent: str
    ) -> Dict:
        """
        Process an interaction through the constellation graph with enhanced error handling.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        interaction_data : Dict
            Data about the interaction
        learning_context : Dict
            Current learning context
        primary_agent : str
            Type of the primary agent to start with
            
        Returns:
        -------
        Dict
            Processed response
        """
        start_time = time.time()
        
        # Validate primary agent
        if primary_agent not in self.agents:
            error = ConstellationException(
                f"Primary agent {primary_agent} not found in constellation",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.CONFIGURATION,
                recovery_strategy=RecoveryStrategy.FALLBACK
            )
            global_error_handler.handle_error(error)
            
            if self.agents:
                primary_agent = list(self.agents.keys())[0]
                if self.is_logging:
                    logger.info(f"Using first available agent: {primary_agent}")
            else:
                # No agents available, return a basic response
                if self.is_logging:
                    logger.error("No agents available in constellation")
                return {
                    "content": "I apologize, but no agents are currently available in this constellation. Please try again later.",
                    "agent_type": "system",
                    "constellation_type": self.constellation_type,
                    "handoff_history": [],
                    "handoff_to": None,
                    "handoff_reason": None,
                    "performance_metrics": {
                        "execution_time": time.time() - start_time,
                        "success": False,
                        "error_type": "no_agents_available"
                    }
                }
        
        # Create input state for graph with error tracking
        input_state = {
            "user_id": user_id,
            "interaction_data": interaction_data,
            "learning_context": copy.deepcopy(learning_context),
            "primary_agent": primary_agent,
            "agent_responses": {},
            "current_agent": primary_agent,
            "handoff_history": [],
            "final_response": None,
            "handoff_needed": False,
            "handoff_to": None,
            "handoff_reason": None,
            "messages": [],
            "interaction_count": 0,
            "max_handoffs": 3,
            "errors": [],
            "error_count": 0,
            "recovery_attempts": 0,
            "last_error_time": None,
            "performance_metrics": {
                "start_time": start_time,
                "handoff_count": 0,
                "agent_switches": 0
            }
        }
        
        # Create config for graph
        config = {
            "configurable": {},
            "thread_id": f"{user_id}_{primary_agent}"
        }
        
        # Process through compiled graph
        try:
            result = self.compiled_graph.invoke(input_state, config)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            result["performance_metrics"] = {
                **result.get("performance_metrics", {}),
                "execution_time": execution_time,
                "success": True
            }
            
            # Extract final response
            response = result.get("final_response", {})
            if not response:
                # If no final response, use primary agent response
                primary_response = result.get("agent_responses", {}).get(primary_agent, {})
                response = primary_response
            
            # Add metadata
            response["constellation_type"] = self.constellation_type
            response["performance_metrics"] = result.get("performance_metrics", {})
            
            # Clean up handoff history - filter out empty lists and other non-dict items
            handoff_history = result.get("handoff_history", [])
            cleaned_handoff_history = []
            for item in handoff_history:
                if isinstance(item, dict) and item:
                    cleaned_handoff_history.append(item)
            
            response["handoff_history"] = cleaned_handoff_history
            
            return response
            
        except Exception as e:
            # Create constellation exception with recovery strategy
            constellation_error = ConstellationException(
                f"Error processing through constellation graph: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.EXECUTION,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                original_exception=e
            )
            
            # Log error and attempt recovery
            global_error_handler.handle_error(constellation_error)
            
            # Attempt recovery based on error type
            recovery_result = self._attempt_constellation_recovery(
                constellation_error, input_state, config
            )
            
            if recovery_result:
                return recovery_result
            
            # Fallback to simplified execution if graph fails
            if self.is_logging:
                logger.info("Falling back to simplified execution")
            
            try:
                result = self._simulate_graph_execution(input_state)
                
                # Calculate performance metrics for fallback
                execution_time = time.time() - start_time
                result["performance_metrics"] = {
                    "execution_time": execution_time,
                    "success": False,
                    "fallback_used": True,
                    "error_count": 1
                }
                
                # Extract final response
                response = result.get("final_response", {})
                if not response:
                    # If no final response, use primary agent response
                    primary_response = result.get("agent_responses", {}).get(primary_agent, {})
                    response = primary_response
                
                # Add metadata
                response["constellation_type"] = self.constellation_type
                response["handoff_history"] = result.get("handoff_history", [])
                response["performance_metrics"] = result.get("performance_metrics", {})
                response["error_recovery"] = "fallback_execution"
                
                return response
                
            except Exception as fallback_error:
                # Critical error - both primary and fallback failed
                critical_error = ConstellationException(
                    f"Critical constellation failure: {str(fallback_error)}",
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.SYSTEM,
                    recovery_strategy=RecoveryStrategy.NONE,
                    original_exception=fallback_error
                )
                
                global_error_handler.handle_error(critical_error)
                
                # Return minimal error response
                return {
                    "content": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                    "agent_type": primary_agent,
                    "constellation_type": self.constellation_type,
                    "error": True,
                    "error_type": "critical_constellation_failure",
                    "performance_metrics": {
                        "execution_time": time.time() - start_time,
                        "success": False,
                        "critical_error": True
                    }
                }
    
    def _create_router_node(self) -> Node:
        """Create the router node."""
        router_node = Node(name="router_node")
        
        def router_exec(state: ConstellationState) -> Dict:
            """Route interaction to the current agent"""
            current_agent = state["current_agent"]
            
            if self.is_logging:
                logger.info(f"Routing to {current_agent} agent")
            
            return {}
        
        def router_branching(state: ConstellationState) -> str:
            """Determine which agent to route to."""
            return state["current_agent"]
        
        router_node.exec_fn = router_exec
        router_node.branching = router_branching
        
        return router_node
    
    def _create_agent_node(self, agent_type: str, agent: SpecializedAgent) -> Node:
        """Create a node for an agent."""
        agent_node = Node(name=f"{agent_type}_node")
        
        def agent_exec(state: ConstellationState) -> Dict:
            """Process interaction with the agent with enhanced learning context and error handling"""
            agent_start_time = time.time()
            
            if self.is_logging:
                logger.info(f"Processing with {agent_type} agent")
            
            # Get input data
            user_id = state["user_id"]
            interaction_data = state["interaction_data"]
            learning_context = state["learning_context"]
            
            try:
                # Check if agent is properly initialized
                if not agent or not hasattr(agent, 'ainvoke'):
                    raise ConstellationException(
                        f"Agent {agent_type} is not properly initialized",
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.CONFIGURATION,
                        recovery_strategy=RecoveryStrategy.FALLBACK
                    )
                
                # Use the agent to process the query
                query = interaction_data.get("query", "")
                
                # Check if memory should be disabled for this request
                save_memory = not learning_context.get("disable_memory", False)
                
                # Monitor agent performance
                response = run_sync(
                    agent.ainvoke(
                        query,
                        is_save_memory=save_memory,
                        user_id=user_id,
                        learning_context=learning_context,
                    )
                )
                
                # Calculate agent execution time
                agent_execution_time = time.time() - agent_start_time
                
                # Extract and enhance response content
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Enhance response based on learning context
                enhanced_content = self._enhance_response_with_learning_guidance(
                    content, learning_context, agent_type
                )
                
                # Update performance metrics
                state["performance_metrics"]["agent_switches"] = state["performance_metrics"].get("agent_switches", 0) + 1
                
                # Return enhanced agent response with performance data
                return {
                    "agent_responses": {
                        agent_type: {
                            "content": enhanced_content,
                            "agent_type": agent_type,
                            "handoff_to": response.handoff_to if hasattr(response, 'handoff_to') else None,
                            "handoff_reason": response.handoff_reason if hasattr(response, 'handoff_reason') else None,
                            "learning_context_aware": True,
                            "learning_stage": learning_context.get("learning_stage", "exploration"),
                            "execution_time": agent_execution_time,
                            "success": True
                        }
                    }
                }
                
            except ConstellationException as ce:
                # Handle constellation-specific errors
                global_error_handler.handle_error(ce)
                
                # Update error tracking in state
                error_info = {
                    "type": ce.category.value,
                    "message": str(ce),
                    "severity": ce.severity.value,
                    "timestamp": time.time(),
                    "agent": agent_type,
                    "recovery_strategy": ce.recovery_strategy.value
                }
                state["errors"].append(error_info)
                state["error_count"] += 1
                state["last_error_time"] = time.time()
                
                # Attempt agent-level recovery
                recovery_result = self._attempt_agent_recovery(
                    ce, agent_type, interaction_data, learning_context, state
                )
                
                if recovery_result:
                    return recovery_result
                
                # Fallback response
                fallback_response = self._create_fallback_response(
                    interaction_data, learning_context, agent_type, str(ce)
                )
                
                return {
                    "agent_responses": {
                        agent_type: {
                            "content": fallback_response,
                            "agent_type": agent_type,
                            "error_recovery": "fallback_response",
                            "execution_time": time.time() - agent_start_time,
                            "success": False
                        }
                    }
                }
                
            except Exception as e:
                # Handle unexpected errors
                agent_error = ConstellationException(
                    f"Unexpected error in agent {agent_type}: {str(e)}",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.EXECUTION,
                    recovery_strategy=RecoveryStrategy.FALLBACK,
                    original_exception=e
                )
                
                global_error_handler.handle_error(agent_error)
                
                # Update error tracking in state
                error_info = {
                    "type": "execution_error",
                    "message": str(e),
                    "severity": "medium",
                    "timestamp": time.time(),
                    "agent": agent_type,
                    "recovery_strategy": "fallback"
                }
                state["errors"].append(error_info)
                state["error_count"] += 1
                state["last_error_time"] = time.time()
                
                # Create a helpful fallback response
                fallback_response = self._create_fallback_response(
                    interaction_data, learning_context, agent_type, str(e)
                )
                
                return {
                    "agent_responses": {
                        agent_type: {
                            "content": fallback_response,
                            "agent_type": agent_type,
                            "error_recovery": "exception_fallback",
                            "execution_time": time.time() - agent_start_time,
                            "success": False
                        }
                    }
                }
        
        agent_node.exec_fn = agent_exec
        
        return agent_node
    
    def _create_aggregator_node(self) -> Node:
        """Create the aggregator node."""
        aggregator_node = Node(name="aggregator_node")
        
        def aggregator_exec(state: ConstellationState) -> Dict:
            """Aggregate agent responses"""
            current_agent = state["current_agent"]
            response = state["agent_responses"].get(current_agent, {})
            
            # Check for explicit handoff request in response
            handoff_to = response.get("handoff_to")
            if handoff_to and handoff_to in self.agents and handoff_to != current_agent:
                return {
                    "handoff_needed": True,
                    "handoff_to": handoff_to,
                    "handoff_reason": response.get("handoff_reason", "Agent requested handoff")
                }
            else:
                # Check if we should automatically hand off based on content
                handoff_info = self._check_automatic_handoff(response, state)
                if handoff_info:
                    return {
                        "handoff_needed": True,
                        "handoff_to": handoff_info["agent"],
                        "handoff_reason": handoff_info["reason"]
                    }
                else:
                    return {
                        "handoff_needed": False
                    }
        
        aggregator_node.exec_fn = aggregator_exec
        
        return aggregator_node
    
    def _create_handoff_node(self) -> Node:
        """Create the handoff decision node."""
        handoff_node = Node(name="handoff_node")
        
        def handoff_exec(state: ConstellationState) -> Dict:
            """Make handoff decisions with enhanced error handling"""
            try:
                if not state.get("handoff_needed", False):
                    # No handoff needed, prepare final response
                    current_agent = state["current_agent"]
                    final_response = state["agent_responses"].get(current_agent, {})
                    
                    # Add performance metrics to final response
                    if "performance_metrics" in state:
                        final_response["performance_metrics"] = state["performance_metrics"]
                        final_response["performance_metrics"]["total_execution_time"] = time.time() - state["performance_metrics"].get("start_time", time.time())
                    
                    return {
                        "final_response": final_response
                    }
                
                # Process handoff with validation
                handoff_to = state["handoff_to"]
                handoff_reason = state["handoff_reason"]
                current_agent = state["current_agent"]
                
                # Validate handoff target
                if handoff_to not in self.agents:
                    handoff_error = HandoffException(
                        f"Invalid handoff target: {handoff_to} not found in available agents",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.CONFIGURATION,
                        recovery_strategy=RecoveryStrategy.FALLBACK
                    )
                    global_error_handler.handle_error(handoff_error)
                    
                    # Fallback to current agent response
                    return {
                        "final_response": state["agent_responses"].get(current_agent, {})
                    }
                
                # Check handoff limits to prevent infinite loops
                handoff_count = state["performance_metrics"].get("handoff_count", 0)
                max_handoffs = state.get("max_handoffs", 3)
                
                if handoff_count >= max_handoffs:
                    handoff_limit_error = HandoffException(
                        f"Maximum handoff limit ({max_handoffs}) exceeded",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.RESOURCE,
                        recovery_strategy=RecoveryStrategy.FALLBACK
                    )
                    global_error_handler.handle_error(handoff_limit_error)
                    
                    # Return current agent response to prevent infinite handoffs
                    return {
                        "final_response": state["agent_responses"].get(current_agent, {})
                    }
                
                # Update handoff metrics
                state["performance_metrics"]["handoff_count"] = handoff_count + 1
                
                # Create handoff entry with enhanced tracking
                handoff_entry = {
                    "from_agent": current_agent,
                    "to_agent": handoff_to,
                    "reason": handoff_reason,
                    "timestamp": time.time(),
                    "handoff_number": handoff_count + 1
                }
                
                # Add to handoff history
                if "handoff_history" not in state:
                    state["handoff_history"] = []
                state["handoff_history"].append(handoff_entry)
                
                # Update current agent
                state["current_agent"] = handoff_to
                
                if self.is_logging:
                    logger.info(f"Handoff executed: {current_agent} -> {handoff_to} (reason: {handoff_reason})")
                
                # Return state update for continued processing
                return {
                    "current_agent": handoff_to,
                    "handoff_history": state["handoff_history"],
                    "performance_metrics": state["performance_metrics"]
                }
                
            except Exception as e:
                # Handle handoff execution errors
                handoff_exec_error = HandoffException(
                    f"Error during handoff execution: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.EXECUTION,
                    recovery_strategy=RecoveryStrategy.FALLBACK,
                    original_exception=e
                )
                
                global_error_handler.handle_error(handoff_exec_error)
                
                # Update error tracking
                error_info = {
                    "type": "handoff_execution_error",
                    "message": str(e),
                    "severity": "high",
                    "timestamp": time.time(),
                    "recovery_strategy": "fallback"
                }
                state["errors"].append(error_info)
                state["error_count"] += 1
                state["last_error_time"] = time.time()
                
                # Fallback to current agent response
                current_agent = state.get("current_agent", list(self.agents.keys())[0] if self.agents else "default")
                return {
                    "final_response": state["agent_responses"].get(current_agent, {
                        "content": "I apologize, but I encountered an issue during the handoff process. Let me help you with your request.",
                        "agent_type": current_agent,
                        "error_recovery": "handoff_fallback"
                    })
                }
            handoff_entry = {
                "from": current_agent,
                "to": handoff_to,
                "reason": handoff_reason
            }
            
            result = {
                "handoff_history": handoff_entry,
                "current_agent": handoff_to
            }
            
            # Check if we've exceeded maximum handoffs
            # Count only the valid handoffs (dictionaries)
            valid_handoffs = [h for h in state["handoff_history"] if isinstance(h, dict)]
            if len(valid_handoffs) >= 2:  # Already have 2, adding 1 more would be 3
                if self.is_logging:
                    logger.warning("Maximum handoffs reached, forcing termination")
                result["handoff_needed"] = False
                result["final_response"] = self._create_combined_response(state)
            
            return result
        
        def handoff_branching(state: ConstellationState) -> str:
            """Determine whether to continue or terminate."""
            return "continue" if state.get("handoff_needed", False) else "terminate"
        
        handoff_node.exec_fn = handoff_exec
        handoff_node.branching = handoff_branching
        
        return handoff_node
    
    def _create_termination_node(self) -> Node:
        """Create the termination node."""
        termination_node = Node(name="termination_node")
        
        def termination_exec(state: ConstellationState) -> Dict:
            """Make termination decisions"""
            # Ensure we have a final response
            if not state.get("final_response"):
                return {
                    "final_response": self._create_combined_response(state)
                }
            return {}
        
        termination_node.exec_fn = termination_exec
        
        return termination_node
    
    def _build_flow(self) -> List:
        """Build the flow for the graph."""
        # Create flow list
        flow = []
        
        # START -> Router
        flow.append(START >> self.router_node)
        
        # Router -> Agent nodes (conditional)
        agent_routes = {}
        for agent_type, node in self.agent_nodes.items():
            agent_routes[agent_type] = node
        
        flow.append(self.router_node >> agent_routes)
        
        # Agent nodes -> Aggregator
        for agent_type, node in self.agent_nodes.items():
            flow.append(node >> self.aggregator_node)
        
        # Aggregator -> Handoff decision
        flow.append(self.aggregator_node >> self.handoff_node)
        
        # Handoff decision -> Termination or Router (conditional)
        flow.append(self.handoff_node >> {
            "continue": self.router_node,
            "terminate": self.termination_node
        })
        
        # Termination -> END
        flow.append(self.termination_node >> END)
        
        return flow
    
    def _check_automatic_handoff(self, response: Dict, state: Dict) -> Optional[Dict]:
        """
        Intelligent handoff decision based on content analysis and context.
        
        Parameters:
        ----------
        response : Dict
            Agent response
        state : Dict
            Current state
            
        Returns:
        -------
        Optional[Dict]
            Handoff information if needed, None otherwise
        """
        current_agent = state["current_agent"]
        content = response.get("content", "")
        learning_context = state.get("learning_context", {})
        
        # Get dynamic handoff limit (default 5, can be configured)
        max_handoffs = state.get("max_handoffs", 5)
        
        # Count only the valid handoffs (dictionaries)
        valid_handoffs = [h for h in state["handoff_history"] if isinstance(h, dict)]
        if len(valid_handoffs) >= max_handoffs:
            return None
        
        # Prevent immediate handoff back to previous agent
        if valid_handoffs:
            last_handoff = valid_handoffs[-1]
            if last_handoff.get("from") == current_agent:
                return None
        
        # Enhanced handoff logic with context awareness
        handoff_decision = self._analyze_handoff_need(content, current_agent, learning_context, valid_handoffs)
        
        return handoff_decision
    
    def _analyze_handoff_need(self, content: str, current_agent: str, learning_context: Dict, handoff_history: List) -> Optional[Dict]:
        """
        Analyze content and context to determine handoff need with enhanced error handling.
        
        Parameters:
        ----------
        content : str
            Response content to analyze
        current_agent : str
            Current agent type
        learning_context : Dict
            Learning context information
        handoff_history : List
            Previous handoff history
            
        Returns:
        -------
        Optional[Dict]
            Handoff decision with agent and reason
        """
        try:
            # Validate inputs
            if not isinstance(content, str):
                content = str(content) if content else ""
            
            if not isinstance(current_agent, str):
                current_agent = str(current_agent) if current_agent else "default"
            
            if not isinstance(learning_context, dict):
                learning_context = {}
            
            if not isinstance(handoff_history, list):
                handoff_history = []
            
            content_lower = content.lower()
            learning_stage = learning_context.get("learning_stage", "exploration")
            
            # Multi-criteria handoff analysis with error protection
            handoff_scores = {}
            
            # Code-related handoff analysis
            if current_agent != "code_assistant" and "code_assistant" in self.agents:
                try:
                    code_indicators = ["code", "implementation", "function", "class", "method", "algorithm", "syntax", "programming"]
                    code_score = sum(1 for indicator in code_indicators if indicator in content_lower)
                    if code_score >= 2 or ("```" in content and code_score >= 1):
                        handoff_scores["code_assistant"] = {
                            "score": code_score + 2, 
                            "reason": "Code implementation and examples needed",
                            "confidence": min(0.9, 0.5 + (code_score * 0.1))
                        }
                except Exception as e:
                    if self.is_logging:
                        logger.warning(f"Error in code handoff analysis: {e}")
            
            # Documentation handoff analysis
            if current_agent != "documentation_expert" and "documentation_expert" in self.agents:
                try:
                    doc_indicators = ["documentation", "reference", "api", "manual", "guide", "specification", "docs"]
                    doc_score = sum(1 for indicator in doc_indicators if indicator in content_lower)
                    if doc_score >= 2:
                        handoff_scores["documentation_expert"] = {
                            "score": doc_score + 1, 
                            "reason": "Documentation and reference materials needed",
                            "confidence": min(0.8, 0.4 + (doc_score * 0.1))
                        }
                except Exception as e:
                    if self.is_logging:
                        logger.warning(f"Error in documentation handoff analysis: {e}")
            
            # Troubleshooting handoff analysis
            if current_agent != "troubleshooting_expert" and "troubleshooting_expert" in self.agents:
                try:
                    trouble_indicators = ["error", "issue", "problem", "bug", "troubleshoot", "debug", "fix", "resolve"]
                    trouble_score = sum(1 for indicator in trouble_indicators if indicator in content_lower)
                    if trouble_score >= 2:
                        handoff_scores["troubleshooting_expert"] = {
                            "score": trouble_score + 1, 
                            "reason": "Troubleshooting and problem resolution needed",
                            "confidence": min(0.85, 0.4 + (trouble_score * 0.1))
                        }
                except Exception as e:
                    if self.is_logging:
                        logger.warning(f"Error in troubleshooting handoff analysis: {e}")
            
            # Learning context-based handoff adjustments
            try:
                if learning_stage == "practice" and handoff_scores:
                    # Boost code assistant score during practice phase
                    if "code_assistant" in handoff_scores:
                        handoff_scores["code_assistant"]["score"] += 1
                        handoff_scores["code_assistant"]["confidence"] = min(0.95, handoff_scores["code_assistant"]["confidence"] + 0.1)
                
                elif learning_stage == "theory" and "documentation_expert" in handoff_scores:
                    # Boost documentation expert during theory phase
                    handoff_scores["documentation_expert"]["score"] += 1
                    handoff_scores["documentation_expert"]["confidence"] = min(0.9, handoff_scores["documentation_expert"]["confidence"] + 0.1)
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"Error in learning context handoff adjustment: {e}")
            
            # Prevent oscillation between agents
            try:
                if len(handoff_history) >= 2:
                    recent_agents = [h.get("to_agent", h.get("to", "")) for h in handoff_history[-2:]]
                    for agent_type in list(handoff_scores.keys()):
                        if agent_type in recent_agents:
                            # Reduce score for recently used agents
                            handoff_scores[agent_type]["score"] -= 2
                            handoff_scores[agent_type]["confidence"] *= 0.7
                            if handoff_scores[agent_type]["score"] <= 0:
                                del handoff_scores[agent_type]
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"Error in handoff oscillation prevention: {e}")
            
            # Select best handoff candidate
            if handoff_scores:
                try:
                    # Find agent with highest score and sufficient confidence
                    best_agent = max(handoff_scores.items(), 
                                   key=lambda x: x[1]["score"] * x[1].get("confidence", 0.5))
                    
                    agent_name, agent_info = best_agent
                    
                    # Only handoff if confidence is above threshold
                    if agent_info.get("confidence", 0) >= 0.6 and agent_info["score"] >= 3:
                        return {
                            "agent": agent_name,
                            "reason": agent_info["reason"],
                            "confidence": agent_info["confidence"],
                            "score": agent_info["score"]
                        }
                except Exception as e:
                    if self.is_logging:
                        logger.warning(f"Error in handoff candidate selection: {e}")
            
            return None
            
        except Exception as e:
            # Handle any unexpected errors in handoff analysis
            handoff_analysis_error = ConstellationException(
                f"Error in handoff analysis: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.ANALYSIS,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                original_exception=e
            )
            
            global_error_handler.handle_error(handoff_analysis_error)
            
            if self.is_logging:
                logger.error(f"Critical error in handoff analysis: {e}")
            
            return None
    
    def _create_combined_response(self, state: Dict) -> Dict:
        """
        Create a combined response from all agent responses.
        
        Parameters:
        ----------
        state : Dict
            Current state
            
        Returns:
        -------
        Dict
            Combined response
        """
        # Start with primary agent response
        primary_agent = state["primary_agent"]
        primary_response = state["agent_responses"].get(primary_agent, {})
        
        # If we have handoffs, use the last agent's response
        # Filter out non-dict items from handoff history
        valid_handoffs = [h for h in state["handoff_history"] if isinstance(h, dict)]
        if valid_handoffs:
            last_handoff = valid_handoffs[-1]
            last_agent = last_handoff.get("to")
            if last_agent:
                last_response = state["agent_responses"].get(last_agent, {})
                
                if last_response:
                    combined_response = copy.deepcopy(last_response)
                    combined_response["handoff_chain"] = valid_handoffs
                    return combined_response
        
        # Fallback to primary response
        return primary_response
    
    # Keep the simplified execution as a fallback
    def _simulate_graph_execution(self, input_data: Dict) -> Dict:
        """
        Simulate graph execution as a fallback.
        This is used if the compiled graph execution fails.
        """
        # Get the current agent
        current_agent = input_data["current_agent"]
        
        # Check if agent exists
        if current_agent not in self.agents:
            if self.is_logging:
                logger.error(f"Agent {current_agent} not found")
            input_data["final_response"] = {
                "content": f"Agent {current_agent} is not available.",
                "agent_type": "system",
                "handoff_to": None,
                "handoff_reason": None
            }
            return input_data
        
        # Get agent
        agent = self.agents[current_agent]
        
        # Process with agent
        try:
            # Create interaction data for the agent
            user_id = input_data["user_id"]
            interaction_data = input_data["interaction_data"]
            learning_context = input_data["learning_context"]
            
            # Use the agent to process the query
            query = interaction_data.get("query", "")
            
            # Simple agent processing (invoke with query)
            response = run_sync(
                agent.ainvoke(
                    query,
                    user_id=user_id,
                    learning_context=learning_context,
                )
            )
            
            # Store agent response
            input_data["agent_responses"][current_agent] = {
                "content": response.content if hasattr(response, 'content') else str(response),
                "agent_type": current_agent
            }
            
            # Set as final response
            input_data["final_response"] = input_data["agent_responses"][current_agent]
            
        except Exception as e:
            if self.is_logging:
                logger.error(f"Error processing with agent {current_agent}: {e}")
            input_data["final_response"] = {
                "content": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "agent_type": current_agent
            }
        
        return input_data
    
    def _create_fallback_response(self, interaction_data: Dict, learning_context: Dict, agent_type: str, error: str) -> str:
        """
        Create a fallback response when an agent fails.
        
        Parameters:
        ----------
        interaction_data : Dict
            Data about the interaction
        learning_context : Dict
            Current learning context
        agent_type : str
            Type of agent that failed
        error : str
            Error message
            
        Returns:
        -------
        str
            Fallback response
        """
        query = interaction_data.get("query", "")
        
        # Create a helpful response based on agent type and query
        if agent_type == "code_assistant":
            return f"""I understand you're asking about code related to: "{query}"

I can help you with programming questions, code examples, and implementation guidance. Please feel free to ask specific questions about:

- Syntax and language features
- Algorithm implementation
- Code structure and design patterns
- Debugging and troubleshooting
- Best practices for coding

What specific aspect of coding would you like to explore?"""
        
        elif agent_type == "documentation_expert":
            return f"""I understand you're looking for documentation on: "{query}"

I can help you find and understand documentation for various frameworks, libraries, and programming languages. Please feel free to ask about:

- API references and usage
- Framework features and components
- Configuration options and settings
- Best practices and conventions
- Implementation examples

What specific documentation are you looking for?"""
        
        elif agent_type == "instructor":
            return f"""I understand you want to learn about: "{query}"

I can help explain concepts, theories, and principles related to programming and software development. Please feel free to ask about:

- Programming concepts and paradigms
- Computer science fundamentals
- Software architecture principles
- Design patterns and best practices
- Learning resources and tutorials

What specific concept would you like me to explain?"""
        
        else:
            # Generic fallback for other agent types
            return f"""I understand you're asking about: "{query}"

I'm here to help with your learning journey. Please feel free to ask specific questions about programming, software development, or any technical topics you're interested in.

What would you like to learn about today?"""
    
    def _enhance_response_with_learning_guidance(self, content: str, learning_context: Dict, agent_type: str) -> str:
        """Enhance agent response with learning guidance based on context."""
        if not content or not learning_context:
            return content
    
    def _attempt_constellation_recovery(self, error: ConstellationException, 
                                      input_state: Dict, config: Dict) -> Optional[Dict]:
        """
        Attempt to recover from constellation errors.
        
        Parameters:
        ----------
        error : ConstellationException
            The error that occurred
        input_state : Dict
            Current state
        config : Dict
            Graph configuration
            
        Returns:
        -------
        Optional[Dict]
            Recovery result if successful, None otherwise
        """
        try:
            # Track recovery attempt
            input_state["recovery_attempts"] = input_state.get("recovery_attempts", 0) + 1
            input_state["last_error_time"] = time.time()
            
            # Add error to state
            error_info = {
                "type": error.category.value,
                "message": str(error),
                "severity": error.severity.value,
                "timestamp": time.time(),
                "recovery_strategy": error.recovery_strategy.value
            }
            input_state["errors"].append(error_info)
            input_state["error_count"] += 1
            
            # Attempt recovery based on strategy
            if error.recovery_strategy == RecoveryStrategy.RETRY:
                if input_state["recovery_attempts"] <= 2:
                    if self.is_logging:
                        logger.info(f"Attempting retry recovery (attempt {input_state['recovery_attempts']})")
                    
                    # Wait briefly before retry
                    time.sleep(0.1 * input_state["recovery_attempts"])
                    
                    # Retry with modified state
                    modified_state = copy.deepcopy(input_state)
                    modified_state["max_handoffs"] = 1  # Reduce complexity
                    
                    result = self.compiled_graph.invoke(modified_state, config)
                    result["error_recovery"] = "retry_successful"
                    return result
            
            elif error.recovery_strategy == RecoveryStrategy.FALLBACK:
                if self.is_logging:
                    logger.info("Attempting fallback recovery")
                
                # Use simplified execution
                result = self._simulate_graph_execution(input_state)
                result["error_recovery"] = "fallback_recovery"
                return result
            
            elif error.recovery_strategy == RecoveryStrategy.RESET:
                if self.is_logging:
                    logger.info("Attempting reset recovery")
                
                # Reset to minimal state
                reset_state = {
                    "user_id": input_state["user_id"],
                    "interaction_data": input_state["interaction_data"],
                    "learning_context": input_state["learning_context"],
                    "primary_agent": input_state["primary_agent"],
                    "agent_responses": {},
                    "current_agent": input_state["primary_agent"],
                    "handoff_history": [],
                    "final_response": None,
                    "handoff_needed": False,
                    "handoff_to": None,
                    "handoff_reason": None,
                    "messages": [],
                    "interaction_count": 0,
                    "max_handoffs": 1,  # Minimal handoffs
                    "errors": input_state["errors"],
                    "error_count": input_state["error_count"],
                    "recovery_attempts": input_state["recovery_attempts"],
                    "last_error_time": input_state["last_error_time"],
                    "performance_metrics": input_state["performance_metrics"]
                }
                
                result = self._simulate_graph_execution(reset_state)
                result["error_recovery"] = "reset_recovery"
                return result
            
        except Exception as recovery_error:
            if self.is_logging:
                logger.error(f"Recovery attempt failed: {recovery_error}")
            
            # Log recovery failure
            recovery_failure = ConstellationException(
                f"Recovery failed: {str(recovery_error)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.RECOVERY,
                recovery_strategy=RecoveryStrategy.NONE,
                original_exception=recovery_error
            )
            global_error_handler.handle_error(recovery_failure)
        
        return None
    
    def _attempt_agent_recovery(self, error: ConstellationException, agent_type: str,
                              interaction_data: Dict, learning_context: Dict, 
                              state: ConstellationState) -> Optional[Dict]:
        """
        Attempt to recover from agent-specific errors.
        
        Parameters:
        ----------
        error : ConstellationException
            The error that occurred
        agent_type : str
            Type of agent that failed
        interaction_data : Dict
            Original interaction data
        learning_context : Dict
            Learning context
        state : ConstellationState
            Current state
            
        Returns:
        -------
        Optional[Dict]
            Recovery result if successful, None otherwise
        """
        try:
            # Track recovery attempt
            state["recovery_attempts"] = state.get("recovery_attempts", 0) + 1
            
            if error.recovery_strategy == RecoveryStrategy.RETRY:
                if state["recovery_attempts"] <= 2:
                    if self.is_logging:
                        logger.info(f"Attempting agent retry for {agent_type} (attempt {state['recovery_attempts']})")
                    
                    # Get the agent again
                    agent = self.agents.get(agent_type)
                    if agent and hasattr(agent, 'ainvoke'):
                        try:
                            # Simplified retry with reduced complexity
                            query = interaction_data.get("query", "")
                            save_memory = False  # Disable memory for retry
                            
                            response = run_sync(
                                agent.ainvoke(
                                    query,
                                    is_save_memory=save_memory,
                                    user_id=state["user_id"],
                                    learning_context=learning_context,
                                )
                            )
                            
                            content = response.content if hasattr(response, 'content') else str(response)
                            
                            return {
                                "agent_responses": {
                                    agent_type: {
                                        "content": content,
                                        "agent_type": agent_type,
                                        "error_recovery": "retry_successful",
                                        "success": True
                                    }
                                }
                            }
                        except Exception:
                            pass  # Fall through to other recovery strategies
            
            elif error.recovery_strategy == RecoveryStrategy.FALLBACK:
                # Try alternative agent if available
                alternative_agents = [name for name in self.agents.keys() if name != agent_type]
                
                if alternative_agents:
                    fallback_agent_type = alternative_agents[0]
                    fallback_agent = self.agents[fallback_agent_type]
                    
                    if self.is_logging:
                        logger.info(f"Attempting fallback from {agent_type} to {fallback_agent_type}")
                    
                    try:
                        query = interaction_data.get("query", "")
                        save_memory = not learning_context.get("disable_memory", False)
                        
                        response = run_sync(
                            fallback_agent.ainvoke(
                                query,
                                is_save_memory=save_memory,
                                user_id=state["user_id"],
                                learning_context=learning_context,
                            )
                        )
                        
                        content = response.content if hasattr(response, 'content') else str(response)
                        
                        # Update current agent in state
                        state["current_agent"] = fallback_agent_type
                        
                        return {
                            "agent_responses": {
                                fallback_agent_type: {
                                    "content": content,
                                    "agent_type": fallback_agent_type,
                                    "error_recovery": "agent_fallback",
                                    "original_agent": agent_type,
                                    "success": True
                                }
                            }
                        }
                    except Exception:
                        pass  # Fall through to degraded response
            
            # If all recovery attempts fail, return None for degraded response
            return None
            
        except Exception as recovery_error:
            if self.is_logging:
                logger.error(f"Agent recovery failed for {agent_type}: {recovery_error}")
            
            # Log recovery failure
            recovery_failure = ConstellationException(
                f"Agent recovery failed for {agent_type}: {str(recovery_error)}",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.RECOVERY,
                recovery_strategy=RecoveryStrategy.NONE,
                original_exception=recovery_error
            )
            global_error_handler.handle_error(recovery_failure)
            
            return None
    
    def _enhance_response_with_learning_guidance(self, content: str, learning_context: Dict, agent_type: str) -> str:
        """
        Enhance agent response with learning guidance based on context.
        
        Parameters:
        ----------
        content : str
            Original response content
        learning_context : Dict
            Learning context information
        agent_type : str
            Type of agent that generated the response
            
        Returns:
        -------
        str
            Enhanced content with learning guidance
        """
        try:
            if not content or not learning_context:
                return content
            
            # Extract context for guidance
            learning_stage = learning_context.get("learning_stage", "exploration")
            interaction_count = learning_context.get("interaction_count", 0)
            current_module = learning_context.get("current_module", "")
            
            # Don't add guidance if response is very short (likely an error or simple answer)
            if len(content) < 50:
                return content
            
            # Add learning guidance based on stage and agent type
            guidance_notes = []
            
            # Stage-based guidance
            if learning_stage == "exploration" and interaction_count < 5:
                guidance_notes.append("💡 **Tip**: Feel free to ask for more examples or clarification on any concept!")
            elif learning_stage == "concept" and agent_type in ["instructor", "knowledge_synthesizer"]:
                guidance_notes.append("🎯 **Next**: Try asking for practice exercises to apply these concepts.")
            elif learning_stage == "practice" and agent_type == "practice_facilitator":
                guidance_notes.append("🚀 **Challenge**: Once comfortable, ask for more advanced exercises or real-world applications.")
            
            # Agent-specific guidance
            if agent_type == "instructor" and "example" not in content.lower():
                guidance_notes.append("📚 **Suggestion**: Ask for specific examples to see these concepts in action.")
            elif agent_type == "code_assistant" and "```" in content:
                guidance_notes.append("💻 **Practice**: Try running this code and experimenting with modifications.")
            elif agent_type == "practice_facilitator":
                guidance_notes.append("🎯 **Remember**: Learning happens through practice - don't hesitate to try and make mistakes!")
            
            # Add guidance if we have any
            if guidance_notes:
                enhanced_content = content + "\n\n---\n\n" + "\n\n".join(guidance_notes)
                return enhanced_content
            
            return content
            
        except Exception as e:
            if self.is_logging:
                logger.warning(f"Error enhancing response with learning guidance: {e}")
            return content  # Return original content if enhancement fails
