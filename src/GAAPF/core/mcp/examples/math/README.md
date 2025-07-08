# Setup MCP server

- Step 1: Install uv to manage packages and setup environments

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Step 2: Install math project

```
# Create a new directory for our project
uv init add
cd add

# Create virtual environment and activate it
uv venv
source .venv/bin/activate

# Install dependencies
uv add "mcp[cli]" httpx

# Create our server file
touch math_server.py
```

- Step 3: Create Math MCP server by adjusting math.py

```
# main.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math", inspector_proxy_port=6278, inspector_ui_port=6275)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- Step 4: Start Math MCP server
To run the server, run the following command in the terminal:
```
mcp dev main.py
```


# Reference

1. [MCP Server quickstart](https://modelcontextprotocol.io/quickstart/server)

2. [MCP python-sdk](https://github.com/modelcontextprotocol/python-sdk)
