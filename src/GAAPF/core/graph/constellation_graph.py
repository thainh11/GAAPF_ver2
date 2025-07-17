"""
Constellation Graph for GAAPF Architecture

This module provides the ConstellationGraph class that manages
agent handoffs and coordination within a constellation.
"""

import logging
from typing import Dict, List, Optional, Any, Annotated, TypedDict
import copy
import asyncio

from ..agents import SpecializedAgent
from ..utils.async_helpers import run_sync
from .graph import Graph
from .node import Node
from .operator import FlowStateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define reducer functions for state updates
def append_handoff_history(existing: List, update: Dict) -> List:
    """Append a handoff entry to the handoff history."""
    if existing is None:
        existing = []
    return existing + [update]

def update_agent_responses(existing: Dict, update: Dict) -> Dict:
    """Update the agent responses dictionary."""
    result = copy.deepcopy(existing)
    result.update(update)
    return result

# Define the state schema for the constellation graph
class ConstellationState(TypedDict):
    """State schema for the constellation graph."""
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
        Process an interaction through the constellation graph.
        
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
                    "handoff_history": [],
                    "handoff_to": None,
                    "handoff_reason": None
                }
        
        # Create input state for graph
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
            "handoff_reason": None
        }
        
        # Create config for graph
        config = {
            "configurable": {},
            "thread_id": f"{user_id}_{primary_agent}"
        }
        
        # Process through compiled graph
        try:
            result = self.compiled_graph.invoke(input_state, config)
            
            # Extract final response
            response = result.get("final_response", {})
            if not response:
                # If no final response, use primary agent response
                primary_response = result.get("agent_responses", {}).get(primary_agent, {})
                response = primary_response
            
            # Add metadata
            response["constellation_type"] = self.constellation_type
            
            # Clean up handoff history - filter out empty lists and other non-dict items
            handoff_history = result.get("handoff_history", [])
            cleaned_handoff_history = []
            for item in handoff_history:
                if isinstance(item, dict) and item:
                    cleaned_handoff_history.append(item)
            
            response["handoff_history"] = cleaned_handoff_history
            
            return response
        except Exception as e:
            if self.is_logging:
                logger.error(f"Error processing through graph: {e}")
            
            # Fallback to simplified execution if graph fails
            if self.is_logging:
                logger.info("Falling back to simplified execution")
            
            result = self._simulate_graph_execution(input_state)
            
            # Extract final response
            response = result.get("final_response", {})
            if not response:
                # If no final response, use primary agent response
                primary_response = result.get("agent_responses", {}).get(primary_agent, {})
                response = primary_response
            
            # Add metadata
            response["constellation_type"] = self.constellation_type
            response["handoff_history"] = result.get("handoff_history", [])
            
            return response
    
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
            """Process interaction with the agent with enhanced learning context"""
            if self.is_logging:
                logger.info(f"Processing with {agent_type} agent")
            
            # Get input data
            user_id = state["user_id"]
            interaction_data = state["interaction_data"]
            learning_context = state["learning_context"]
            
            try:
                # Check if agent is properly initialized
                if not agent or not hasattr(agent, 'ainvoke'):
                    raise AttributeError(f"Agent {agent_type} is not properly initialized")
                
                # Use the agent to process the query
                query = interaction_data.get("query", "")
                
                # ---- pass the ORIGINAL user query; the agent will build its own context ----
                # Check if memory should be disabled for this request
                save_memory = not learning_context.get("disable_memory", False)
                
                response = run_sync(
                    agent.ainvoke(
                        query,
                        is_save_memory=save_memory,
                        user_id=user_id,
                        learning_context=learning_context,
                    )
                )
                
                # Extract and enhance response content
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Enhance response based on learning context
                enhanced_content = self._enhance_response_with_learning_guidance(
                    content, learning_context, agent_type
                )
                
                # Return enhanced agent response
                return {
                    "agent_responses": {
                        agent_type: {
                            "content": enhanced_content,
                            "agent_type": agent_type,
                            "handoff_to": response.handoff_to if hasattr(response, 'handoff_to') else None,
                            "handoff_reason": response.handoff_reason if hasattr(response, 'handoff_reason') else None,
                            "learning_context_aware": True,
                            "learning_stage": learning_context.get("learning_stage", "exploration")
                        }
                    }
                }
            except Exception as e:
                if self.is_logging:
                    logger.error(f"Error processing with agent {agent_type}: {e}")
                
                # Create a helpful fallback response
                fallback_response = self._create_fallback_response(interaction_data, learning_context, agent_type, str(e))
                
                # Return fallback response
                return {
                    "agent_responses": {
                        agent_type: {
                            "content": fallback_response,
                            "agent_type": agent_type
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
            """Make handoff decisions"""
            if not state.get("handoff_needed", False):
                # No handoff needed, prepare final response
                current_agent = state["current_agent"]
                return {
                    "final_response": state["agent_responses"].get(current_agent, {})
                }
            
            # Process handoff
            handoff_to = state["handoff_to"]
            handoff_reason = state["handoff_reason"]
            current_agent = state["current_agent"]
            
            # Create handoff entry
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
        Check if we should automatically hand off based on response content.
        
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
        
        # Count only the valid handoffs (dictionaries)
        valid_handoffs = [h for h in state["handoff_history"] if isinstance(h, dict)]
        if len(valid_handoffs) >= 2:
            return None
        
        # Simple keyword-based handoff rules
        if current_agent != "code_assistant" and ("code" in content.lower() or "implementation" in content.lower()):
            return {"agent": "code_assistant", "reason": "Code implementation needed"}
        
        if current_agent != "documentation_expert" and ("documentation" in content.lower() or "reference" in content.lower()):
            return {"agent": "documentation_expert", "reason": "Documentation reference needed"}
        
        if current_agent != "troubleshooter" and ("error" in content.lower() or "bug" in content.lower() or "fix" in content.lower()):
            return {"agent": "troubleshooter", "reason": "Error resolution needed"}
        
        if current_agent != "practice_facilitator" and ("practice" in content.lower() or "exercise" in content.lower()):
            return {"agent": "practice_facilitator", "reason": "Practice exercises needed"}
        
        if current_agent != "knowledge_synthesizer" and ("connect" in content.lower() or "integrate" in content.lower() or "synthesize" in content.lower()):
            return {"agent": "knowledge_synthesizer", "reason": "Knowledge synthesis needed"}
        
        # No automatic handoff needed
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