"""
Constellation Graph for GAAPF Architecture

This module provides the ConstellationGraph class that manages
agent handoffs and coordination within a constellation.
"""

import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated
import copy

from ..agents import SpecializedAgent
from .operator import FlowStateGraph
from .node import Node
from .constants import END, START
from langgraph.checkpoint.memory import MemorySaver

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define state schema for constellation graph
def merge_agent_responses(existing: Dict, update: Dict) -> Dict:
    """Merge agent responses"""
    if isinstance(existing, dict) and isinstance(update, dict):
        merged = existing.copy()
        merged.update(update)
        return merged
    return update or existing

def append_handoff_history(existing: List, update: List) -> List:
    """Append to handoff history"""
    if isinstance(existing, list) and isinstance(update, list):
        return existing + update
    elif isinstance(update, list):
        return (existing or []) + update
    elif update:
        return (existing or []) + [update]
    return existing or []

class ConstellationState(TypedDict):
    """State schema for constellation graph execution"""
    user_id: str
    interaction_data: Dict
    learning_context: Dict
    primary_agent: str
    current_agent: str
    agent_responses: Annotated[Dict, merge_agent_responses]
    handoff_history: Annotated[List, append_handoff_history]
    handoff_needed: bool
    handoff_to: Optional[str]
    handoff_reason: Optional[str]
    final_response: Optional[Dict]

# Node implementations for constellation workflow
class RouterNode(Node):
    """Routes interaction to the appropriate agent"""
    
    def __init__(self, is_logging: bool = False):
        super().__init__(name="router_node")
        self.is_logging = is_logging
    
    def exec(self, state: ConstellationState) -> Dict:
        """Route interaction to current agent"""
        current_agent = state["current_agent"]
        
        if self.is_logging:
            logger.info(f"Routing to {current_agent} agent")
        
        return {"current_agent": current_agent}

class AgentNode(Node):
    """Executes a specific agent"""
    
    def __init__(self, agent_type: str, agent: SpecializedAgent, is_logging: bool = False):
        super().__init__(name=f"{agent_type}_node")
        self.agent_type = agent_type
        self.agent = agent
        self.is_logging = is_logging
    
    def exec(self, state: ConstellationState) -> Dict:
        """Execute the agent"""
        if self.is_logging:
            logger.info(f"Processing with {self.agent_type} agent")
        
        try:
            # Get input data
            user_id = state["user_id"]
            interaction_data = state["interaction_data"]
            learning_context = state["learning_context"]
            
            # Use the agent to process the query
            query = interaction_data.get("query", "")
            
            # Process with agent
            response = self.agent.invoke(query, user_id=user_id)
            
            # Create response data
            response_data = {
                "content": response.content if hasattr(response, 'content') else str(response),
                "agent_type": self.agent_type
            }
            
            # Return agent response
            return {
                "agent_responses": {self.agent_type: response_data}
            }
            
        except Exception as e:
            if self.is_logging:
                logger.error(f"Error processing with agent {self.agent_type}: {e}")
            
            error_response = {
                "content": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "agent_type": self.agent_type
            }
            
            return {
                "agent_responses": {self.agent_type: error_response}
            }

class AggregatorNode(Node):
    """Aggregates agent responses and checks for handoff conditions"""
    
    def __init__(self, agents: Dict[str, SpecializedAgent], is_logging: bool = False):
        super().__init__(name="aggregator_node")
        self.agents = agents
        self.is_logging = is_logging
    
    def exec(self, state: ConstellationState) -> Dict:
        """Aggregate agent responses and check handoff conditions"""
        current_agent = state["current_agent"]
        agent_responses = state.get("agent_responses", {})
        response = agent_responses.get(current_agent, {})
        
        # Check for explicit handoff request in response
        handoff_to = response.get("handoff_to")
        if handoff_to and handoff_to in self.agents and handoff_to != current_agent:
            return {
                "handoff_needed": True,
                "handoff_to": handoff_to,
                "handoff_reason": response.get("handoff_reason", "Agent requested handoff")
            }
        
        # Check if we should automatically hand off based on content
        handoff_info = self._check_automatic_handoff(response, state)
        if handoff_info:
            return {
                "handoff_needed": True,
                "handoff_to": handoff_info["agent"],
                "handoff_reason": handoff_info["reason"]
            }
        
        return {"handoff_needed": False}
    
    def _check_automatic_handoff(self, response: Dict, state: ConstellationState) -> Optional[Dict]:
        """Check if we should automatically hand off based on response content."""
        current_agent = state["current_agent"]
        content = response.get("content", "")
        handoff_history = state.get("handoff_history", [])
        
        # Skip if we've already had multiple handoffs
        if len(handoff_history) >= 2:
            return None
        
        # Simple keyword-based handoff rules
        handoff_rules = [
            {
                "keywords": ["code", "implementation", "example", "syntax"],
                "agent": "code_assistant",
                "reason": "Code implementation needed"
            },
            {
                "keywords": ["error", "bug", "fix", "issue", "problem"],
                "agent": "troubleshooter",
                "reason": "Error resolution needed"
            },
            {
                "keywords": ["document", "reference", "api", "specification"],
                "agent": "documentation_expert",
                "reason": "Documentation reference needed"
            },
            {
                "keywords": ["practice", "exercise", "challenge", "try"],
                "agent": "practice_facilitator",
                "reason": "Practice exercises needed"
            },
            {
                "keywords": ["assess", "evaluate", "test", "quiz"],
                "agent": "assessment",
                "reason": "Assessment needed"
            },
            {
                "keywords": ["connect", "integrate", "synthesize", "relate"],
                "agent": "knowledge_synthesizer",
                "reason": "Knowledge synthesis needed"
            }
        ]
        
        # Check each rule
        for rule in handoff_rules:
            agent_type = rule["agent"]
            # Skip if this is the current agent or not available
            if agent_type == current_agent or agent_type not in self.agents:
                continue
            
            # Check if any keywords match
            if any(keyword in content.lower() for keyword in rule["keywords"]):
                return {"agent": agent_type, "reason": rule["reason"]}
        
        return None

class HandoffDecisionNode(Node):
    """Makes handoff decisions and manages agent transitions"""
    
    def __init__(self, is_logging: bool = False):
        super().__init__(name="handoff_node")
        self.is_logging = is_logging
    
    def exec(self, state: ConstellationState) -> Dict:
        """Make handoff decisions"""
        handoff_needed = state.get("handoff_needed", False)
        
        if not handoff_needed:
            # No handoff needed, prepare final response
            current_agent = state["current_agent"]
            agent_responses = state.get("agent_responses", {})
            final_response = agent_responses.get(current_agent, {})
            return {
                "final_response": final_response,
                "handoff_needed": False
            }
        
        # Process handoff
        handoff_to = state["handoff_to"]
        handoff_reason = state["handoff_reason"]
        current_agent = state["current_agent"]
        handoff_history = state.get("handoff_history", [])
        
        # Record handoff in history
        handoff_entry = {
            "from": current_agent,
            "to": handoff_to,
            "reason": handoff_reason
        }
        
        # Check if we've exceeded maximum handoffs
        if len(handoff_history) >= 3:
            if self.is_logging:
                logger.warning("Maximum handoffs reached, forcing termination")
            # Create combined response
            agent_responses = state.get("agent_responses", {})
            final_response = self._create_combined_response(agent_responses, handoff_history)
            return {
                "handoff_needed": False,
                "final_response": final_response
            }
        
        # Continue with handoff
        return {
            "current_agent": handoff_to,
            "handoff_history": [handoff_entry],
            "handoff_needed": True
        }
    
    def branching(self, state: ConstellationState) -> str:
        """Determine next step based on handoff decision"""
        return "continue" if state.get("handoff_needed", False) else "terminate"
    
    def _create_combined_response(self, agent_responses: Dict, handoff_history: List) -> Dict:
        """Create a combined response from all agent responses."""
        if handoff_history:
            last_handoff = handoff_history[-1]
            last_agent = last_handoff["to"]
            last_response = agent_responses.get(last_agent, {})
            
            if last_response:
                combined_response = copy.deepcopy(last_response)
                combined_response["handoff_chain"] = handoff_history
                return combined_response
        
        # Fallback to first available response
        if agent_responses:
            return list(agent_responses.values())[0]
        
        return {
            "content": "No response available.",
            "agent_type": "system"
        }

class TerminationNode(Node):
    """Handles final response preparation"""
    
    def __init__(self, is_logging: bool = False):
        super().__init__(name="termination_node")
        self.is_logging = is_logging
    
    def exec(self, state: ConstellationState) -> Dict:
        """Prepare final response"""
        final_response = state.get("final_response")
        
        if not final_response:
            # Create final response from current agent
            current_agent = state["current_agent"]
            agent_responses = state.get("agent_responses", {})
            final_response = agent_responses.get(current_agent, {})
        
        return {"final_response": final_response}

class ConstellationGraph:
    """
    Specialized graph for agent constellation coordination using proper LangGraph compilation.
    
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
        
        # Build and compile the graph
        self.compiled_graph = self._build_and_compile_graph()
        
        if self.is_logging:
            logger.info(f"Initialized ConstellationGraph for {constellation_type} constellation with {len(self.agents)} agents")
    
    def process(
        self,
        user_id: str,
        interaction_data: Dict,
        learning_context: Dict,
        primary_agent: str
    ) -> Dict:
        """
        Process an interaction through the constellation graph using proper LangGraph execution.
        
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
        # Validate primary agent
        if primary_agent not in self.agents:
            if self.is_logging:
                logger.warning(f"Primary agent {primary_agent} not found in constellation")
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
                    "handoff_history": []
                }
        
        # Create initial state for graph execution
        initial_state = {
            "user_id": user_id,
            "interaction_data": interaction_data,
            "learning_context": copy.deepcopy(learning_context),
            "primary_agent": primary_agent,
            "current_agent": primary_agent,
            "agent_responses": {},
            "handoff_history": [],
            "handoff_needed": False,
            "handoff_to": None,
            "handoff_reason": None,
            "final_response": None
        }
        
        # Execute through compiled graph
        result = self.compiled_graph.invoke(initial_state)
        
        # Extract final response
        response = result.get("final_response", {})
        if not response:
            # Fallback to primary agent response
            primary_response = result.get("agent_responses", {}).get(primary_agent, {})
            response = primary_response or {
                "content": "No response generated.",
                "agent_type": primary_agent
            }
        
        # Add metadata
        response["constellation_type"] = self.constellation_type
        response["handoff_history"] = result.get("handoff_history", [])
        
        return response
    
    def _build_and_compile_graph(self) -> Any:
        """
        Build and compile the constellation graph using proper LangGraph compilation.
        
        Returns:
        -------
        CompiledStateGraph
            Compiled graph ready for execution
        """
        # Create FlowStateGraph with proper state schema
        graph = FlowStateGraph(ConstellationState)
        
        # Create node instances
        router_node = RouterNode(is_logging=self.is_logging)
        aggregator_node = AggregatorNode(agents=self.agents, is_logging=self.is_logging)
        handoff_node = HandoffDecisionNode(is_logging=self.is_logging)
        termination_node = TerminationNode(is_logging=self.is_logging)
        
        # Create agent nodes
        agent_nodes = {}
        for agent_type, agent in self.agents.items():
            agent_node = AgentNode(agent_type, agent, is_logging=self.is_logging)
            agent_nodes[agent_type] = agent_node
        
        # Define the flow using >> operator pattern
        flow = []
        
        # Router to agent nodes (conditional)
        agent_path_map = {}
        for agent_type, agent_node in agent_nodes.items():
            agent_path_map[agent_type] = agent_node
        
        # Set up router branching
        def router_condition(state: ConstellationState) -> str:
            return state["current_agent"]
        
        router_node.branching = router_condition
        router_node_route = router_node >> agent_path_map
        flow.append(router_node_route)
        
        # Agent nodes to aggregator
        for agent_node in agent_nodes.values():
            flow.append(agent_node >> aggregator_node)
        
        # Aggregator to handoff decision
        flow.append(aggregator_node >> handoff_node)
        
        # Handoff decision with conditional routing
        handoff_route = handoff_node >> {
            "continue": router_node,
            "terminate": termination_node
        }
        flow.append(handoff_route)
        
        # Termination to end
        flow.append(termination_node >> END)
        
        # Compile the graph
        checkpointer = MemorySaver()
        compiled_graph = graph.compile(checkpointer=checkpointer, flow=flow)
        
        return compiled_graph 