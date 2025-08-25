import asyncio
import sys
from pathlib import Path
from typing import Any

# Ensure src/ on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from GAAPF.core.agents.code_assistant import CodeAssistantAgent


class MockLLM:
    """Minimal LLM stub that returns a response containing a Python code block.
    The CodeAssistantAgent will extract and write this code to generated/langchain/.
    """

    def invoke(self, messages: Any) -> Any:
        # Return only a Python fenced block; agent will strip and write to file
        content = (
            """
```python
from langchain_core.tools import tool

@tool
def hello(name: str) -> str:
    "Say hello"
    return f"Hello, {name}!"

if __name__ == "__main__":
    # Tool can be invoked directly
    print(hello.invoke({"name": "LangChain"}))
```
"""
        ).strip()
        class R:
            def __init__(self, c: str):
                self.content = c
        return R(content)


async def main() -> None:
    llm = MockLLM()
    agent = CodeAssistantAgent(llm, is_logging=True)

    # Prepare directories and baseline for diff
    out_dir = PROJECT_ROOT / "generated" / "langchain"
    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(p.name for p in out_dir.glob("*"))

    # Query crafted to avoid generator delegation triggers
    query = (
        "Return a Python code block only that defines a simple LangChain Tool and invokes it. "
        "No explanations."
    )

    res = await agent.ainvoke(query, learning_context={"framework": "langchain"})

    after = set(p.name for p in out_dir.glob("*"))
    created = sorted(after - before)

    print({
        "framework": "langchain",
        "written_files": created,
        "message": res.get("content", "")[:200]
    })


if __name__ == "__main__":
    asyncio.run(main())


