from typing import Union, Dict, Optional, Any, Callable
from langchain_core.runnables import RunnableConfig

# Modified Node class with >> operator support
class Node():
    def __init__(self, name: Optional[str] = None, config: Any = None):
        self.name = name or self.__class__.__name__
        self.config = config
        self.exec_fn = None  # Function to execute for this node
    
    def exec(self, state: Any, config: Optional[RunnableConfig] = None) -> Union[dict, str]:
        """Execute the node's function"""
        if self.exec_fn:
            return self.exec_fn(state)
        return state

    def branching(self, state: Any, config: Optional[RunnableConfig] = None) -> str:
        pass

    def __rshift__(self, other: Union['Node', Dict[str, 'Node'], str]) -> 'Node':
        self.target = other
        return self
