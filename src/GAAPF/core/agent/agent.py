import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Awaitable, List, AsyncGenerator, Optional, Union, Dict
from typing_extensions import is_typeddict
from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
import logging
from pathlib import Path
from typing import Union
from typing_extensions import is_typeddict
import nest_asyncio
import traceback
import time

from ..register.tool import ToolManager
from ..memory.memory import Memory
from ..mcp.client import DistributedMCPClient
from ..graph.function_graph import FunctionStateGraph

nest_asyncio.apply()

# Setup enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentMeta(ABC):
    """Abstract base class for agents"""

    @abstractmethod
    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List[Union[str, BaseTool]] = [],
        *args,
        **kwargs,
    ):
        """Initialize a new Agent with LLM and tools"""
        pass

    @abstractmethod
    def invoke(self, query: str, *args, **kwargs) -> Any:
        """Synchronously invoke the agent's main function"""
        pass

    @abstractmethod
    async def ainvoke(self, query: str, *args, **kwargs) -> Awaitable[Any]:
        """Asynchronously invoke the agent's main function"""
        pass

class Agent(AgentMeta):
    """Enhanced AI agent with comprehensive error handling, logging, and tool-calling capabilities"""
    
    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List[Union[str, BaseTool]] = [],
        tools_path: Path = Path("templates/tools.json"),
        is_reset_tools = False,
        description: str = "You are a helpful assistant who can use the following tools to complete a task.",
        skills: list[str] = ["You can answer the user question with tools"],
        flow: list[str] = [],
        state_schema: type[Any] = None,
        config_schema: type[Any] = None,
        memory_path: Path = None,
        is_reset_memory = False,
        use_long_term_memory: bool = False,
        chroma_path: Optional[Union[Path, str]] = Path('memory/chroma_db'),
        collection_name: str = "long_term_memory",
        embedding_model: str = "text-embedding-004",
        api_key: Optional[str] = None,
        mcp_client: DistributedMCPClient = None,
        mcp_server_name: str = None,
        is_pii: bool = False,
        is_logging: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the enhanced agent with comprehensive error handling and logging.
        """
        start_time = time.time()
        
        self.llm = llm
        self.tools = tools
        self.description = description
        self.skills = skills
        self.flow = flow
        self.is_logging = is_logging
        
        if self.is_logging:
            logger.info(f"🚀 Initializing Agent with {len(tools)} tools and flow: {bool(flow)}")
        
        if self.flow:
            try:
                self.initialize_flow(state_schema=state_schema, config_schema=config_schema)
                if self.is_logging:
                    logger.info("✅ Agent flow initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize agent flow: {e}")
                if self.is_logging:
                    logger.error(f"Flow initialization error details: {traceback.format_exc()}")
                raise
        
        self.tools_path = None
        if tools_path:
            self.tools_path = Path(tools_path) if isinstance(tools_path, str) else tools_path
        else:
            self.tools_path = Path("templates/tools.json")
        
        if self.is_logging:
            logger.info(f"📁 Tools path set to: {self.tools_path}")
        
        self.is_reset_tools = is_reset_tools
        
        try:
            self.tools_manager = ToolManager(tools_path=self.tools_path, is_reset_tools=self.is_reset_tools)
            if self.is_logging:
                logger.info("✅ ToolManager initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ToolManager: {e}")
            raise

        # Use lazy tool registration to improve initialization performance
        self._pending_tools = self.tools
        self._tools_registered = False
        
        # Only register tools if explicitly needed or if there are very few
        if len(self.tools) <= 2:  # Register immediately for small tool sets
            try:
                self.register_tools(self.tools)
                self._tools_registered = True
                if self.is_logging:
                    logger.info(f"✅ Registered {len(self.tools)} tools immediately")
            except Exception as e:
                logger.error(f"❌ Failed to register tools: {e}")
                
        self.mcp_client = mcp_client
        self.mcp_server_name = mcp_server_name
        
        if memory_path and (not str(memory_path).endswith(".json")):
            raise ValueError("memory_path must be json format ending with .json. For example, 'templates/memory.json'")
            
        self.memory_path = Path(memory_path) if isinstance(memory_path, str) else memory_path
        self.is_reset_memory = is_reset_memory
        self.memory = None
        
        if memory_path:
            try:
                if use_long_term_memory:
                    from ..memory import LongTermMemory
                    self.memory = LongTermMemory(
                        memory_path=memory_path,
                        is_reset_memory=is_reset_memory,
                        chroma_path=chroma_path,
                        collection_name=collection_name,
                        embedding_model=embedding_model,
                        api_key=api_key,
                        is_logging=True
                    )
                    if self.is_logging:
                        logger.info(f"🧠 Long-term memory initialized with ChromaDB at {chroma_path}")
                else:
                    from ..memory import Memory
                    self.memory = Memory(memory_path=memory_path, is_reset_memory=is_reset_memory)
                    if self.is_logging:
                        logger.info("🧠 Standard memory initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize memory: {e}")
                if self.is_logging:
                    logger.error(f"Memory initialization error: {traceback.format_exc()}")
                    
        self.is_pii = is_pii
        self._user_id = None
        self._agent_type = "agent"  # Default agent type
        
        if not self.is_pii:
            self._user_id = 'unknown_user'
            
        end_time = time.time()
        if self.is_logging:
            logger.info(f"✅ Agent initialization completed in {end_time - start_time:.3f}s")
    
    async def connect_mcp_tool(self):
        """Connect to MCP tools with comprehensive error handling"""
        if self.is_logging:
            logger.info(f"🔌 Connecting to MCP tools: client={bool(self.mcp_client)}, server={self.mcp_server_name}")
            
        try:
            if self.mcp_client and self.mcp_server_name:
                mcp_tools = await self.tools_manager.register_mcp_tool(self.mcp_client, self.mcp_server_name)
                if self.is_logging:
                    logger.info(f"✅ Connected to MCP server {self.mcp_server_name} with {len(mcp_tools)} tools!")
            elif self.mcp_client:
                mcp_tools = await self.tools_manager.register_mcp_tool(self.mcp_client)
                if self.is_logging:
                    logger.info(f"✅ Connected to MCP server with {len(mcp_tools)} tools!")
            return "Successfully connected to mcp server!"
        except Exception as e:
            logger.error(f"❌ Failed to connect MCP tools: {e}")
            if self.is_logging:
                logger.error(f"MCP connection error: {traceback.format_exc()}")
            return f"Failed to connect to MCP server: {str(e)}"

    def initialize_flow(self, 
                        state_schema: type[Any],
                        config_schema: type[Any]):
        """Initialize agent flow with enhanced error handling"""
        if self.is_logging:
            logger.info("🔧 Initializing agent flow with state and config schemas")
            
        # Validate state_schema if provided
        if state_schema is not None and not is_typeddict(state_schema):
            raise TypeError("state_schema must be a TypedDict subclass")
        
        # Validate config_schema if provided
        if config_schema is not None and not is_typeddict(config_schema):
            raise TypeError("config_schema must be a TypedDict subclass")
        
        if self.flow:
            try:
                self.graph = FunctionStateGraph(state_schema=state_schema, config_schema=config_schema) 
                self.checkpoint = MemorySaver()
                self.compiled_graph = self.graph.compile(checkpointer=self.checkpoint, flow=self.flow)
                if self.is_logging:
                    logger.info("✅ Agent flow compiled successfully")
            except Exception as e:
                logger.error(f"❌ Failed to compile agent flow: {e}")
                raise
            
    def register_tools(self, tools: List[str]) -> Any:
        """Register tools with comprehensive error handling and logging"""
        if not tools:
            return
            
        if self.is_logging:
            logger.info(f"🔧 Registering {len(tools)} tools: {tools}")
            
        success_count = 0
        for tool in tools:
            try:
                self.tools_manager.register_module_tool(tool, llm=self.llm)
                success_count += 1
                if self.is_logging:
                    logger.info(f"✅ Registered tool: {tool}")
            except Exception as e:
                logger.error(f"❌ Failed to register tool {tool}: {e}")
                if self.is_logging:
                    logger.error(f"Tool registration error for {tool}: {traceback.format_exc()}")
                    
        if self.is_logging:
            logger.info(f"📊 Tool registration complete: {success_count}/{len(tools)} successful")
            
    def _ensure_tools_registered(self):
        """Ensure tools are registered before use (lazy loading) with logging"""
        if not self._tools_registered and self._pending_tools:
            if self.is_logging:
                logger.info(f"⏳ Lazy loading {len(self._pending_tools)} tools for agent: {self._agent_type}")
            self.register_tools(self._pending_tools)
            self._tools_registered = True

    @property
    def user_id(self):
        return self._user_id

    @user_id.setter
    def user_id(self, new_user_id):
        if self.is_logging:
            logger.info(f"👤 User ID changed from {self._user_id} to {new_user_id}")
        self._user_id = new_user_id

    def prompt_template(self, query: str, user_id: str = "unknown_user", framework_id: str = None, *args, **kwargs) -> str:
        """Generate prompt template with enhanced memory context and error handling"""
        if self.is_logging:
            logger.info(f"📝 Generating prompt template for user: {user_id}, framework: {framework_id}")
            
        try:
            # Try to load tools with error handling
            try:
                with open(self.tools_path, "r", encoding="utf-8") as f:
                    tools = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"⚠️ Could not load tools from {self.tools_path}: {e}")
                tools = {}
                # Create empty tools file
                try:
                    self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")
                    if self.is_logging:
                        logger.info("📄 Created empty tools file")
                except Exception as create_error:
                    logger.error(f"❌ Failed to create tools file: {create_error}")
            
            memory_content = ""
            if self.memory:
                try:
                    if self.is_logging:
                        logger.info("🧠 Loading memory context...")
                        
                    # Check if we're using LongTermMemory and get relevant context
                    if hasattr(self.memory, 'get_relevant_context'):
                        memory_context = self.memory.get_relevant_context(query, user_id, framework_id=framework_id)
                        if memory_context and memory_context != "No relevant memories found.":
                            memory_content += f"{memory_context}\n"
                            if self.is_logging:
                                logger.info("✅ Long-term memory context loaded")
                    
                    # Get standard memory as string
                    memory_string = self.memory.load_memory(load_type='string', user_id=user_id)
                    if memory_string:
                        memory_content += f"- Memory: {memory_string}\n"
                        if self.is_logging:
                            logger.info("✅ Short-term memory context loaded")
                            
                except Exception as e:
                    logger.error(f"❌ Failed to load memory context: {e}")
                    if self.is_logging:
                        logger.error(f"Memory loading error: {traceback.format_exc()}")

            prompt = (
                "You are given a task, a list of available tools, and the memory about user to have precise information.\n"
                f"- Task: {query}\n"
                f"- Tools list: {json.dumps(tools)}\n"
                f"{memory_content}\n"
                f"- User: {user_id}\n"
                "------------------------\n"    
                "Instructions:\n"
                "- Let's answer in a natural, clear, and detailed way without providing reasoning or explanation.\n"
                f"- If user used I in Memory, let's replace by name {user_id} in User part.\n"
                "- You need to think about whether the question need to use Tools?\n"
                "- If it was daily normal conversation. Let's directly answer as a human with memory.\n"
                "- If the task requires a tool, select the appropriate tool with its relevant arguments from Tools list according to following format (no explanations, no markdown):\n"
                "{\n"
                '    "tool_name": "Function name",\n'
                '    "tool_type": "Type of tool. Only get one of three values [\\"function\\", \\"module\\", \\"mcp\\"]",\n'
                '    "arguments": {"arg_name": "arg_value"},\n'
                '    "module_path": "Path to import the tool"\n'
                "}\n"
                "Let's say I don't know and suggest where to search if you are unsure the answer.\n"
                "Not make up anything.\n"
            )
            
            if self.is_logging:
                logger.info("✅ Prompt template generated successfully")
                
            return prompt
            
        except Exception as e:
            logger.error(f"❌ Failed to generate prompt template: {e}")
            if self.is_logging:
                logger.error(f"Prompt generation error: {traceback.format_exc()}")
            # Return a safe fallback prompt
            return f"Task: {query}\nUser: {user_id}\nPlease respond to this query in a helpful way."

    def invoke(self, query: str, is_save_memory: bool = False, user_id: str = "unknown_user", learning_context: Dict = None, **kwargs) -> Any:
        """
        Enhanced synchronous wrapper with comprehensive error handling.
        """
        if self.is_logging:
            logger.info(f"🔄 Synchronous invoke called for user: {user_id}")
            
        try:
            # Get the current running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If no event loop is running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            if self.is_logging:
                logger.info("🔄 Created new event loop for synchronous call")

        try:
            # Schedule the async method and wait for its result
            task = loop.create_task(self.ainvoke(query, is_save_memory, user_id, learning_context, **kwargs))
            result = loop.run_until_complete(task)
            if self.is_logging:
                logger.info("✅ Synchronous invoke completed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ Synchronous invoke failed: {e}")
            if self.is_logging:
                logger.error(f"Sync invoke error: {traceback.format_exc()}")
            # Return safe fallback response
            return AIMessage(content="I apologize, but I encountered an error processing your request. Please try again.")

    def stream(self, query: str, is_save_memory: bool = False, user_id: str = "unknown_user", learning_context: Dict = None, **kwargs) -> AsyncGenerator[Any, None]:
        """
        Enhanced streaming with comprehensive error handling and logging.
        """
        if self.is_logging:
            logger.info(f"🌊 Starting stream for user: {user_id}")
            
        # Enhanced user ID handling with error protection
        try:
            if learning_context and "user_profile" in learning_context:
                user_profile = learning_context["user_profile"]
                if user_id and user_id != "unknown_user":
                    self._user_id = user_id
                elif user_profile and user_profile.get("user_id"):
                    self._user_id = user_profile.get("user_id")
                else:
                    self._user_id = user_id or "unknown_user"
                
                display_name = user_profile.get("name", self._user_id) if user_profile else self._user_id
            else:
                self._user_id = user_id or "unknown_user"
                display_name = self._user_id

            if self.is_logging:
                logger.info(f"👤 Stream user context: {display_name} (ID: {self._user_id})")

            prompt = self.prompt_template(query=query, user_id=self._user_id)
            skills = "- ".join(self.skills)
            messages = [
                SystemMessage(content=f"{self.description}\nHere is your skills: {skills}"),
                HumanMessage(content=prompt),
            ]

            if self.memory and is_save_memory:
                try:
                    self.save_memory(query, user_id=self._user_id)
                    if self.is_logging:
                        logger.info("💾 Saved user query to memory")
                except Exception as e:
                    logger.error(f"❌ Failed to save user query to memory: {e}")

            try:
                if hasattr(self, 'compiled_graph'):
                    if self.is_logging:
                        logger.info("🔄 Using compiled graph for streaming")
                        
                    result = []
                    config = kwargs.get('config', {"configurable": {"user_id": user_id, "thread_id": f"{self._agent_type}-{user_id}"}})
                    
                    for chunk in self.compiled_graph.stream(input=query, config=config):
                        for v in chunk.values():
                            if v:
                                result += v 
                                yield v
                                
                    if self.memory and is_save_memory:
                        try:
                            self.save_memory(tool_message=result, user_id=self._user_id)
                            if self.is_logging:
                                logger.info("💾 Saved graph result to memory")
                        except Exception as e:
                            logger.error(f"❌ Failed to save graph result: {e}")
                    yield result
                else:
                    if self.is_logging:
                        logger.info("🔄 Using standard LLM streaming")
                        
                    # Accumulate streamed content
                    full_content = AIMessageChunk(content="")
                    for chunk in self.llm.stream(messages):
                        full_content += chunk
                        yield chunk

                    # After streaming is complete, process tool data
                    tool_data_str = self.tools_manager.extract_tool(full_content.content)
                    if (tool_data_str is None) or (tool_data_str == "{}"):
                        if self.is_logging:
                            logger.info("✅ No tool call detected, returning direct response")
                        return full_content
                    
                    # Parse and execute tool with enhanced error handling
                    try:
                        tool_call = json.loads(tool_data_str)
                        if self.is_logging:
                            logger.info(f"🔧 Executing tool call: {tool_call.get('tool_name', 'unknown')}")
                            
                        tool_message = asyncio.run(
                            self.tools_manager._execute_tool(
                                tool_name=tool_call.get("tool_name"),
                                tool_type=tool_call.get("tool_type"),
                                arguments=tool_call.get("arguments", {}),
                                module_path=tool_call.get("module_path"),
                                mcp_client=self.mcp_client,
                                mcp_server_name=self.mcp_server_name
                            )
                        )

                        # Enhanced tool result processing
                        tool_result = getattr(tool_message, 'artifact', getattr(tool_message, 'content', str(tool_message)))
                        
                        prompt_tool = (
                            "You are a professional reporter. Your task is to deliver a clear and factual report that directly addresses the given question. "
                            "Use the tool's name and result only to support your explanation. Do not fabricate any information or over-interpret the result."
                            f"- Question: {query}"
                            f"- Tool Used: {tool_call}"
                            f"- Result: {tool_result}"
                            "Report"
                        )
                        
                        message = self.llm.invoke(prompt_tool)
                        
                        # Enhanced tool message handling
                        if hasattr(tool_message, 'content'):
                            tool_message.content = "\n" + message.content
                        else:
                            tool_message = ToolMessage(content=message.content, tool_call_id="stream_tool_call")

                        if self.memory and is_save_memory:
                            try:
                                self.save_memory(tool_message=tool_message, user_id=self._user_id)
                                if self.is_logging:
                                    logger.info("💾 Saved tool result to memory")
                            except Exception as e:
                                logger.error(f"❌ Failed to save tool result: {e}")
                                
                        yield tool_message
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Tool call JSON decode error: {e}")
                        yield AIMessage(content="I encountered an error processing the tool call, but I'm still here to help.")
                    except Exception as e:
                        logger.error(f"❌ Tool execution error: {e}")
                        if self.is_logging:
                            logger.error(f"Tool execution error details: {traceback.format_exc()}")
                        yield AIMessage(content="I encountered an error with tool execution, but I can still assist you.")

            except Exception as e:
                logger.error(f"❌ Streaming error: {e}")
                if self.is_logging:
                    logger.error(f"Streaming error details: {traceback.format_exc()}")
                yield AIMessage(content="I encountered an error during streaming, but I'm still available to help.")

        except Exception as e:
            logger.error(f"❌ Critical streaming error: {e}")
            if self.is_logging:
                logger.error(f"Critical streaming error: {traceback.format_exc()}")
            yield AIMessage(content="I apologize for the technical difficulties. Please try again.")

    async def ainvoke(self, query: str, is_save_memory: bool = False, user_id: str = "unknown_user", learning_context: Dict = None, **kwargs) -> Any:
        """
        Enhanced asynchronous invoke with comprehensive error handling and logging.
        """
        start_time = time.time()
        
        if self.is_logging:
            logger.info(f"🚀 Async invoke started for user: {user_id}, query length: {len(query)}")
            
        try:
            self._ensure_tools_registered()

            # Enhanced user context handling
            if learning_context and "user_profile" in learning_context:
                user_profile = learning_context["user_profile"]
                self._user_id = user_profile.get("user_id", user_id or "unknown_user")
                display_name = user_profile.get("name", self._user_id)
            else:
                self._user_id = user_id or "unknown_user"
                display_name = self._user_id

            if self.is_logging:
                logger.info(f"👤 Processing request for {display_name} (ID: {self._user_id})")

            # Enhanced message preparation
            messages = []
            if learning_context:
                try:
                    # Check if the subclass implements the context-aware method
                    try:
                        system_prompt = self._generate_system_prompt(learning_context)
                        if self.is_logging:
                            logger.info("✅ Generated context-aware system prompt")
                    except TypeError:
                        system_prompt = self._generate_system_prompt()
                        if self.is_logging:
                            logger.info("✅ Generated standard system prompt")
                        
                    messages.append(SystemMessage(content=system_prompt))
                    
                    # Enhanced memory loading
                    if self.memory:
                        try:
                            if self.is_logging:
                                logger.info("🧠 Loading conversation history from memory...")
                            history_messages = self.memory.get_messages(user_id=self._user_id, agent_type=self._agent_type)
                            messages.extend(history_messages)
                            if self.is_logging:
                                logger.info(f"✅ Loaded {len(history_messages)} messages from memory")
                        except Exception as e:
                            logger.error(f"❌ Failed to load memory messages: {e}")

                    enhanced_query = self._enhance_query_with_context(query, learning_context)
                    messages.append(HumanMessage(content=enhanced_query))
                    
                except Exception as e:
                    logger.error(f"❌ Error preparing learning context: {e}")
                    # Fallback to basic prompt
                    prompt = self.prompt_template(query=query, user_id=self._user_id)
                    skills = "- ".join(self.skills)
                    messages = [
                        SystemMessage(content=f"{self.description}\nHere is your skills: {skills}"),
                        HumanMessage(content=prompt),
                    ]
            else:
                prompt = self.prompt_template(query=query, user_id=self._user_id)
                skills = "- ".join(self.skills)
                messages = [
                    SystemMessage(content=f"{self.description}\nHere is your skills: {skills}"),
                    HumanMessage(content=prompt),
                ]

            # Save user message to memory with error handling
            if self.memory and is_save_memory:
                try:
                    self.save_memory(HumanMessage(content=query), user_id=self._user_id)
                    if self.is_logging:
                        logger.info("💾 Saved user message to memory")
                except Exception as e:
                    logger.error(f"❌ Failed to save user message: {e}")

            # Enhanced execution with comprehensive error handling
            try:
                if hasattr(self, 'compiled_graph'):
                    if self.is_logging:
                        logger.info("🔄 Executing with compiled graph")
                        
                    config = kwargs.get('config', {"configurable": {"user_id": user_id, "thread_id": f"{self._agent_type}-{user_id}"}})
                    
                    # Prepare input for the graph
                    graph_input = {"messages": messages}
                    if hasattr(self, 'graph') and hasattr(self.graph, 'schema') and 'query' in getattr(self.graph.schema, '__annotations__', {}):
                        graph_input['query'] = query

                    result = await self.compiled_graph.ainvoke(input=graph_input, config=config)
                    
                    if self.memory and is_save_memory:
                        try:
                            self.save_memory(AIMessage(content=str(result)), user_id=self._user_id)
                            if self.is_logging:
                                logger.info("💾 Saved graph result to memory")
                        except Exception as e:
                            logger.error(f"❌ Failed to save graph result: {e}")
                    
                    # Enhanced result extraction
                    if isinstance(result, dict):
                        if 'response' in result:
                            final_result = AIMessage(content=result['response'])
                        elif 'messages' in result and isinstance(result['messages'], list) and result['messages']:
                            last_message = result['messages'][-1]
                            if isinstance(last_message, AIMessage):
                                final_result = last_message
                            else:
                                final_result = AIMessage(content=str(last_message))
                        else:
                            final_result = AIMessage(content=str(result))
                    else:
                        final_result = AIMessage(content=str(result))
                        
                    end_time = time.time()
                    if self.is_logging:
                        logger.info(f"✅ Graph execution completed in {end_time - start_time:.3f}s")
                    return final_result
                    
                else:
                    if self.is_logging:
                        logger.info("🔄 Executing with standard LLM call")
                        
                    # Enhanced LLM call with tool processing
                    response = await self.llm.ainvoke(messages)
                    
                    # Enhanced tool extraction and execution
                    tool_data = self.tools_manager.extract_tool(response.content)
                    
                    if not tool_data or ("None" in tool_data) or (tool_data == "{}"):
                        if self.memory and is_save_memory:
                            try:
                                self.save_memory(response, user_id=self._user_id)
                                if self.is_logging:
                                    logger.info("💾 Saved direct response to memory")
                            except Exception as e:
                                logger.error(f"❌ Failed to save response: {e}")
                        
                        end_time = time.time()
                        if self.is_logging:
                            logger.info(f"✅ Direct response completed in {end_time - start_time:.3f}s")
                        return response
                    
                    # Enhanced tool execution
                    try:
                        tool_call = json.loads(tool_data)
                        if self.is_logging:
                            logger.info(f"🔧 Executing tool: {tool_call.get('tool_name', 'unknown')}")
                            
                        tool_message = await self.tools_manager._execute_tool(
                            tool_name=tool_call.get("tool_name"),
                            tool_type=tool_call.get("tool_type"), 
                            arguments=tool_call.get("arguments", {}),
                            module_path=tool_call.get("module_path"),
                            mcp_client=self.mcp_client,
                            mcp_server_name=self.mcp_server_name
                        )

                        # Enhanced tool result processing
                        tool_result = getattr(tool_message, 'artifact', getattr(tool_message, 'content', str(tool_message)))
                        
                        prompt_tool = (
                            "You are a professional reporter. Your task is to deliver a clear and factual report that directly addresses the given question. "
                            "Use the tool's name and result only to support your explanation. Do not fabricate any information or over-interpret the result."
                            f"- Question: {query}"
                            f"- Tool Used: {tool_call}"
                            f"- Result: {tool_result}"
                            "Report"
                        )
                        
                        message = await self.llm.ainvoke(prompt_tool)
                        
                        # Enhanced response message creation
                        if hasattr(tool_message, 'content'):
                            tool_message.content = message.content
                        else:
                            tool_message = AIMessage(content=message.content)
                        
                        if self.memory and is_save_memory:
                            try:
                                self.save_memory(tool_message, user_id=self._user_id)
                                if self.is_logging:
                                    logger.info("💾 Saved tool response to memory")
                            except Exception as e:
                                logger.error(f"❌ Failed to save tool response: {e}")
                                
                        end_time = time.time()
                        if self.is_logging:
                            logger.info(f"✅ Tool execution completed in {end_time - start_time:.3f}s")
                        return tool_message
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Tool call JSON decode error: {e}")
                        return AIMessage(content="I encountered an error parsing the tool call, but I'm here to help with your question.")
                    except Exception as e:
                        logger.error(f"❌ Tool execution error: {e}")
                        if self.is_logging:
                            logger.error(f"Tool execution error details: {traceback.format_exc()}")
                        return AIMessage(content="I encountered an error with tool execution, but I can still assist you with your request.")
                
            except Exception as e:
                logger.error(f"❌ Execution error: {e}")
                if self.is_logging:
                    logger.error(f"Execution error details: {traceback.format_exc()}")
                return AIMessage(content="I apologize, but I encountered an error processing your request. Please try rephrasing your question or try again.")
            
        except Exception as e:
            logger.error(f"❌ Critical ainvoke error: {e}")
            if self.is_logging:
                logger.error(f"Critical error details: {traceback.format_exc()}")
            return AIMessage(content="I apologize for the technical difficulties. Please try again in a moment.")

    def _enhance_query_with_context(self, query: str, learning_context: Dict) -> str:
        """
        Enhanced query context enrichment with error handling.
        Subclasses can override this for specialized behavior.
        """
        try:
            if self.is_logging:
                logger.info("🔗 Enhancing query with learning context")
            return query
        except Exception as e:
            logger.error(f"❌ Error enhancing query context: {e}")
            return query

    def _generate_system_prompt(self, learning_context: Dict = None) -> str:
        """
        Enhanced system prompt generation with error handling.
        Subclasses should override this for specialized behavior.
        """
        try:
            if self.is_logging:
                logger.info("📝 Generating system prompt")
            skills = "- ".join(self.skills)
            return f"{self.description}\nHere are your skills: {skills}"
        except Exception as e:
            logger.error(f"❌ Error generating system prompt: {e}")
            return self.description

    def save_memory(self, message: Union[ToolMessage, AIMessage, str], user_id: str = "unknown_user") -> None:
        """
        Enhanced memory saving with comprehensive error handling and logging.
        """
        if not self.memory:
            return
            
        try:
            if self.is_logging:
                logger.info(f"💾 Saving message to memory for user: {user_id}")
                
            if isinstance(message, str):
                self.memory.save_short_term_memory(self.llm, message, user_id=user_id, agent_type=self._agent_type)
                if self.is_logging:
                    logger.info(f"✅ Saved string message to memory: {message[:100]}...")
            elif isinstance(message, AIMessage):
                self.memory.save_short_term_memory(self.llm, message.content, user_id=user_id, agent_type=self._agent_type)
                if self.is_logging:
                    logger.info(f"✅ Saved AI message to memory: {message.content[:100]}...")
            elif hasattr(message, 'artifact') and isinstance(message.artifact, str):
                self.memory.save_short_term_memory(self.llm, message.artifact, user_id=user_id, agent_type=self._agent_type)
                if self.is_logging:
                    logger.info(f"✅ Saved tool artifact to memory: {str(message.artifact)[:100]}...")
            elif hasattr(message, 'content'):
                self.memory.save_short_term_memory(self.llm, message.content, user_id=user_id, agent_type=self._agent_type)
                if self.is_logging:
                    logger.info(f"✅ Saved message content to memory: {message.content[:100]}...")
            else:
                # Fallback for other types - convert to string
                self.memory.save_short_term_memory(self.llm, str(message), user_id=user_id, agent_type=self._agent_type)
                if self.is_logging:
                    logger.info(f"✅ Saved converted message to memory: {str(message)[:100]}...")
                    
        except Exception as e:
            logger.error(f"❌ Failed to save message to memory: {e}")
            if self.is_logging:
                logger.error(f"Memory save error details: {traceback.format_exc()}")
                                               
    def function_tool(self, func: Any):
        """Register function tool with error handling"""
        try:
            if self.is_logging:
                logger.info(f"🔧 Registering function tool: {func.__name__ if hasattr(func, '__name__') else 'unknown'}")
            return self.tools_manager.register_function_tool(func)
        except Exception as e:
            logger.error(f"❌ Failed to register function tool: {e}")
            raise
