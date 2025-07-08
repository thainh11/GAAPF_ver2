from typing import Optional, Sequence, Dict, Any
from typing_extensions import Self
from .constants import END, START
from langgraph.utils.runnable import coerce_to_runnable
from .state_graph import StateGraph
from .node import Node

class Operator:
    """
    Base class for operators in the GAAPF architecture.
    
    Operators are processing functions that can be applied to states
    in the learning workflow.
    """
    
    def __init__(self, name: str = None, description: str = None):
        """
        Initialize an operator.
        
        Parameters:
        ----------
        name : str, optional
            Name of the operator
        description : str, optional
            Description of what the operator does
        """
        self.name = name or self.__class__.__name__
        self.description = description or "Base operator"
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the operator to a state.
        
        Parameters:
        ----------
        state : Dict[str, Any]
            Current state to process
            
        Returns:
        -------
        Dict[str, Any]
            Processed state
        """
        # Base implementation returns state unchanged
        return state
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

# Modified StateGraph class
class FlowStateGraph(StateGraph):
    def __init__(self, state_schema=None, config_schema=None, *, input=None, output=None):
        super().__init__(state_schema, config_schema, input=input, output=output)
        self.node_instances = {}
        self.flow_provided = False

    def add_node(self, name: str, node: Node) -> Self:
        if not isinstance(node, Node):
            raise ValueError(f"Node {name} must be an instance of Node class")
        if name in self.channels:
            raise ValueError(f"'{name}' is already being used as a state key")
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        # if name in (START.name, END.name):
        #     raise ValueError(f"Node '{name}' is reserved")
        self.node_instances[name] = node
        if hasattr(node, 'config') and (node.config is not None):
            runnable = coerce_to_runnable(lambda state, config: node.exec(state, config), name=name, trace=True)
        else:
            runnable = coerce_to_runnable(lambda state: node.exec(state), name=name, trace=True)
        super().add_node(name, runnable)
        return self

    def process_flow(self, flow: Sequence[Node]) -> Self:
        if not flow:
            raise ValueError("Flow cannot be empty")
        self.flow_provided = True
        has_start_edge = any(
            isinstance(route.target, (str, Node)) and route.name == START.name
            for route in flow
        )
        if not has_start_edge:
            first_node = flow[0]
            if first_node.name not in self.node_instances:
                self.add_node(first_node.name, first_node)
            self.add_edge(START.name, first_node.name)
        for route in flow:
            if (route.name not in self.node_instances) and (route.name not in [START.name, END.name]):
                self.add_node(route.name, route)
            if isinstance(route.target, str):
                self.add_edge(route.name, route.target)
            elif isinstance(route.target, dict):
                path_map = {key: target.name if isinstance(target, Node) else target for key, target in route.target.items()}
                self.add_conditional_edges(source=route.name, path=self.node_instances[route.name].branching, path_map=path_map)
            elif isinstance(route.target, Node):
                if route.target.name not in self.node_instances and (route.name != START.name and route.target.name != END.name):
                    self.add_node(route.target.name, route.target)
                self.add_edge(route.name, route.target.name)
            else:
                raise ValueError(f"Invalid target type {type(route.target)} in route")
        return self

    def validate(self, interrupt: Optional[Sequence[str]] = None) -> None:
        if self.flow_provided:
            all_sources = {start for start, _ in self._all_edges}
            all_targets = {end for _, end in self._all_edges}
            for source in all_sources:
                if source not in self.nodes and source != START.name:
                    raise ValueError(f"Found edge starting at unknown node '{source}'")
            for target in all_targets:
                if target not in self.nodes and target != END.name:
                    raise ValueError(f"Found edge ending at unknown node '{target}'")
        else:
            super().validate(interrupt)

    def compile(self, checkpointer=None, *, flow=None, **kwargs):
        if flow:
            self.process_flow(flow)
        self.validate()
        compiled = super().compile(checkpointer=checkpointer, **kwargs)
        return compiled
