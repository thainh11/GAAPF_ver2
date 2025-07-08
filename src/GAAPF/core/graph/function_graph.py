from typing import Optional, Sequence
from typing_extensions import Self
from .node import Node
from .constants import END, START
from .state_graph import StateGraph
from langgraph.utils.runnable import coerce_to_runnable
import inspect
from typing import Callable, Union, Dict, Any

class NodeWrapper(Node):
    def __init__(self, func: Callable, branching: Callable = None, config: Any = None, name: Optional[str] = None):
        self.func = func
        self.branching = branching
        self.config = config
        self.name = name or func.__name__
        self.target = None
        self.accepts_config = self._check_accepts_config()

    def _check_accepts_config(self) -> bool:
        """Check if self.func accepts a 'config' parameter."""
        sig = inspect.signature(self.func)
        return 'config' in sig.parameters
    
    def __call__(self, state: Any, config: Any = None, **kargs) -> Union[dict, str]:
        if config:
            if self.accepts_config:
                return self.func(state, config, **kargs)
            else:
                raise ValueError(f"function {self.func} should have parameter config")
        else:
            return self.func(state, **kargs)
        
    def __rshift__(self, other: Union['NodeWrapper', Dict[str, 'NodeWrapper'], str]) -> 'NodeWrapper':
        self.target = other
        return self

def node(*, branching: Callable = None, **kargs) -> Callable:
    def decorator(func: Callable) -> NodeWrapper:
        return NodeWrapper(func, branching=branching, **kargs)
    return decorator

# Modified StateGraph class
class FunctionStateGraph(StateGraph):
    def __init__(self, state_schema=None, config_schema=None, *, input=None, output=None):
        super().__init__(state_schema, config_schema, input=input, output=output)
        self.node_instances = {}
        self.flow_provided = False

    def add_node(self, name: str, node: any) -> Self:
        # Validate node has required attributes instead of checking isinstance(node, Node)
        if not isinstance(node, Node):
            raise ValueError(f"Node {name} must be an instance of Node class, but type was {type(node)}")
        if not hasattr(node, 'name') or not callable(node):
            raise ValueError(f"Node {name} must have 'name' and be callable")
        if name in self.channels:
            raise ValueError(f"'{name}' is already being used as a state key")
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        self.node_instances[name] = node
        # Check if node expects config in its execution
        if hasattr(node, 'config') and (not node.config is None):
            runnable = coerce_to_runnable(lambda state, config: node(state, config), name=name, trace=True)
        else:
            runnable = coerce_to_runnable(lambda state: node(state), name=name, trace=True)
        super().add_node(name, runnable)
        return self

    def process_flow(self, flow: Sequence[any]) -> Self:
        if not flow:
            raise ValueError("Flow cannot be empty")
        self.flow_provided = True
        has_start_edge = any(
            isinstance(route.target, (str, dict)) or (hasattr(route.target, 'name') and callable(route.target))
            for route in flow
            if hasattr(route, 'name') and route.name == START.name
        )
        if not has_start_edge:
            first_node = flow[0]
            if first_node.name not in self.node_instances:
                self.add_node(first_node.name, first_node)
            self.add_edge(START.name, first_node.name)
        for route in flow:
            if not hasattr(route, 'name') or not hasattr(route, 'target'):
                raise ValueError(f"Route {route} must have 'name' and 'target' attributes")
            if (route.name not in self.node_instances) and (route.name not in [START.name, END.name]):
                self.add_node(route.name, route)
            if isinstance(route.target, str):
                self.add_edge(route.name, route.target)
            elif isinstance(route.target, dict):
                path_map = {key: target.name if isinstance(target, NodeWrapper) else target for key, target in route.target.items()}
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
