#!/usr/bin/env python3
"""
Agent Coordination and Tool Execution Test

This test focuses on:
1. Agent initialization and setup
2. Constellation graph coordination
3. Tool execution (not just descriptions)
4. Agent handoffs and collaboration
5. LangGraph state management
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.GAAPF.core.graph.constellation_graph import ConstellationGraph
from src.GAAPF.core.agents.instructor import InstructorAgent
from src.GAAPF.core.agents.mentor import MentorAgent
from src.GAAPF.core.agents.research_assistant import ResearchAssistantAgent
from src.GAAPF.core.agents.code_assistant import CodeAssistantAgent
from src.GAAPF.core.tools.deepsearch import DeepSearchTool
from src.GAAPF.core.tools.websearch_tools import WebSearchTool

# Import LLM
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI
    print("✅ LLM imports successful")
except ImportError as e:
    print(f"❌ LLM import error: {e}")
    sys.exit(1)


class AgentCoordinationTester:
    """Tester for agent coordination and tool execution"""
    
    def __init__(self):
        self.test_user_id = "test_agent_coordination"
        self.llm = self._initialize_llm()
        self.agents = {}
        self.constellation_graph = None
        print("✅ Agent coordination tester initialized")
    
    def _initialize_llm(self):
        """Initialize LLM with available providers"""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    google_api_key=google_api_key,
                    temperature=0.7
                )
                print("✅ Using Google Gemini LLM")
                return llm
            except Exception as e:
                print(f"❌ Google Gemini failed: {e}")
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=openai_api_key,
                    temperature=0.7
                )
                print("✅ Using OpenAI LLM")
                return llm
            except Exception as e:
                print(f"❌ OpenAI failed: {e}")
        
        raise ValueError("No LLM available. Please set GOOGLE_API_KEY or OPENAI_API_KEY")
    
    async def test_agent_initialization(self):
        """Test initialization of all specialized agents"""
        print(f"\n{'='*50}")
        print("TESTING AGENT INITIALIZATION")
        print(f"{'='*50}")
        
        try:
            # Create memory directory
            memory_dir = Path("memory")
            memory_dir.mkdir(exist_ok=True)
            
            # Initialize tools
            tools = []
            
            # Add web search tool if API key is available
            if os.getenv("TAVILY_API_KEY"):
                try:
                    web_search_tool = WebSearchTool(api_key=os.getenv("TAVILY_API_KEY"))
                    tools.append(web_search_tool)
                    print("✅ Web search tool initialized")
                except Exception as e:
                    print(f"⚠️  Web search tool failed: {e}")
            
            # Initialize agents with tools
            self.agents = {
                'instructor': InstructorAgent(
                    llm=self.llm,
                    tools=tools,
                    memory_path=memory_dir / f"instructor_{self.test_user_id}_memory.json",
                    config={"explanation_depth": "balanced"},
                    is_logging=True
                ),
                'mentor': MentorAgent(
                    llm=self.llm,
                    tools=tools,
                    memory_path=memory_dir / f"mentor_{self.test_user_id}_memory.json",
                    config={"guidance_style": "supportive"},
                    is_logging=True
                ),
                'research_assistant': ResearchAssistantAgent(
                    llm=self.llm,
                    tools=tools,
                    memory_path=memory_dir / f"research_assistant_{self.test_user_id}_memory.json",
                    config={"research_depth": "comprehensive"},
                    is_logging=True
                ),
                'code_assistant': CodeAssistantAgent(
                    llm=self.llm,
                    tools=tools,
                    memory_path=memory_dir / f"code_assistant_{self.test_user_id}_memory.json",
                    config={"code_style": "production"},
                    is_logging=True
                )
            }
            
            print(f"✅ Initialized {len(self.agents)} specialized agents:")
            for agent_name, agent in self.agents.items():
                print(f"   - {agent_name}: {agent.agent_type}")
                print(f"     Tools: {len(agent.tools)}")
                print(f"     Config: {agent.config}")
            
            return {"status": "success", "agents_count": len(self.agents)}
            
        except Exception as e:
            print(f"❌ Agent initialization failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def test_constellation_graph_setup(self):
        """Test constellation graph setup and compilation"""
        print(f"\n{'='*50}")
        print("TESTING CONSTELLATION GRAPH SETUP")
        print(f"{'='*50}")
        
        try:
            # Constellation configuration
            constellation_config = {
                "name": "learning_constellation",
                "type": "educational",
                "coordination_mode": "collaborative",
                "max_handoffs": 3,
                "timeout_seconds": 30
            }
            
            # Initialize constellation graph
            self.constellation_graph = ConstellationGraph(
                agents=self.agents,
                constellation_type="learning",
                constellation_config=constellation_config,
                is_logging=True
            )
            
            # Verify graph compilation
            assert self.constellation_graph.compiled_graph is not None, "Graph should be compiled"
            assert hasattr(self.constellation_graph, 'agents'), "Graph should have agents"
            assert len(self.constellation_graph.agents) > 0, "Graph should have at least one agent"
            
            print("✅ Constellation graph setup successful")
            print(f"   - Agents: {len(self.constellation_graph.agents)}")
            print(f"   - Type: {self.constellation_graph.constellation_type}")
            print(f"   - Config: {constellation_config}")
            
            return {"status": "success", "graph_compiled": True}
            
        except Exception as e:
            print(f"❌ Constellation graph setup failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def test_basic_agent_interaction(self):
        """Test basic agent interaction without handoffs"""
        print(f"\n{'='*50}")
        print("TESTING BASIC AGENT INTERACTION")
        print(f"{'='*50}")
        
        try:
            # Test data
            test_scenarios = [
                {
                    "name": "Instructor - Concept Explanation",
                    "agent": "instructor",
                    "interaction_data": {
                        "query": "Explain the concept of chains in LangChain",
                        "context": "learning_session",
                        "user_input": "I want to understand how chains work in LangChain"
                    }
                },
                {
                    "name": "Code Assistant - Code Example",
                    "agent": "code_assistant",
                    "interaction_data": {
                        "query": "Show me a simple LangChain chain example",
                        "context": "code_example_needed",
                        "user_input": "I need to see actual code for a LangChain chain"
                    }
                },
                {
                    "name": "Mentor - Learning Guidance",
                    "agent": "mentor",
                    "interaction_data": {
                        "query": "How should I approach learning LangChain?",
                        "context": "learning_guidance",
                        "user_input": "I'm new to LangChain and need guidance"
                    }
                }
            ]
            
            learning_context = {
                "framework_name": "LangChain",
                "current_module": "introduction",
                "user_profile": {
                    "name": self.test_user_id,
                    "experience_level": "intermediate",
                    "goals": ["learn_concepts", "build_production_app"]
                },
                "session_id": f"test_session_{int(datetime.now().timestamp())}"
            }
            
            results = []
            
            for scenario in test_scenarios:
                print(f"\n🧪 Testing: {scenario['name']}")
                
                # Process interaction through constellation graph
                response = self.constellation_graph.process(
                    user_id=self.test_user_id,
                    interaction_data=scenario["interaction_data"],
                    learning_context=learning_context,
                    primary_agent=scenario["agent"]
                )
                
                # Validate response
                assert isinstance(response, dict), "Response should be a dictionary"
                assert "content" in response, "Response should have content"
                assert len(response["content"]) > 50, "Response should be substantial"
                
                result = {
                    "scenario": scenario["name"],
                    "agent": scenario["agent"],
                    "response_length": len(response["content"]),
                    "has_handoffs": len(response.get("handoff_history", [])) > 0,
                    "status": "success"
                }
                
                results.append(result)
                
                print(f"   ✅ {scenario['name']} completed")
                print(f"      - Response length: {result['response_length']} characters")
                print(f"      - Handoffs: {len(response.get('handoff_history', []))}")
            
            print(f"\n✅ All basic interactions completed successfully")
            return {"status": "success", "results": results}
            
        except Exception as e:
            print(f"❌ Basic agent interaction test failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def test_agent_handoff_scenarios(self):
        """Test agent handoff scenarios"""
        print(f"\n{'='*50}")
        print("TESTING AGENT HANDOFF SCENARIOS")
        print(f"{'='*50}")
        
        try:
            # Handoff scenarios
            handoff_scenarios = [
                {
                    "name": "Instructor to Code Assistant",
                    "primary_agent": "instructor",
                    "interaction_data": {
                        "query": "Explain LangChain chains and show me code examples",
                        "context": "explanation_with_code",
                        "user_input": "I need both theory and practical examples"
                    }
                },
                {
                    "name": "Research to Instructor",
                    "primary_agent": "research_assistant",
                    "interaction_data": {
                        "query": "Find latest LangChain features and explain them",
                        "context": "research_and_explain",
                        "user_input": "What are the newest features in LangChain?"
                    }
                },
                {
                    "name": "Mentor to Code Assistant",
                    "primary_agent": "mentor",
                    "interaction_data": {
                        "query": "Help me learn LangChain with practical exercises",
                        "context": "guided_learning",
                        "user_input": "I learn best with hands-on coding"
                    }
                }
            ]
            
            learning_context = {
                "framework_name": "LangChain",
                "current_module": "advanced_concepts",
                "user_profile": {
                    "name": self.test_user_id,
                    "experience_level": "intermediate",
                    "goals": ["learn_concepts", "build_production_app"],
                    "learning_style": "hands_on"
                },
                "session_id": f"handoff_session_{int(datetime.now().timestamp())}"
            }
            
            handoff_results = []
            
            for scenario in handoff_scenarios:
                print(f"\n🔄 Testing: {scenario['name']}")
                
                # Process with potential handoffs
                response = self.constellation_graph.process(
                    user_id=self.test_user_id,
                    interaction_data=scenario["interaction_data"],
                    learning_context=learning_context,
                    primary_agent=scenario["primary_agent"]
                )
                
                # Analyze handoff behavior
                handoff_history = response.get("handoff_history", [])
                
                result = {
                    "scenario": scenario["name"],
                    "primary_agent": scenario["primary_agent"],
                    "handoffs_occurred": len(handoff_history),
                    "handoff_chain": [h.get("to_agent", "unknown") for h in handoff_history],
                    "final_agent": response.get("agent_type", "unknown"),
                    "response_length": len(response.get("content", "")),
                    "status": "success"
                }
                
                handoff_results.append(result)
                
                print(f"   ✅ {scenario['name']} completed")
                print(f"      - Handoffs: {result['handoffs_occurred']}")
                print(f"      - Chain: {' -> '.join([scenario['primary_agent']] + result['handoff_chain'])}")
                print(f"      - Final agent: {result['final_agent']}")
            
            print(f"\n✅ All handoff scenarios completed")
            return {"status": "success", "handoff_results": handoff_results}
            
        except Exception as e:
            print(f"❌ Agent handoff test failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def test_tool_execution_verification(self):
        """Test that agents actually execute tools instead of just describing actions"""
        print(f"\n{'='*50}")
        print("TESTING TOOL EXECUTION VERIFICATION")
        print(f"{'='*50}")
        
        try:
            # Test scenarios that should trigger tool usage
            tool_scenarios = [
                {
                    "name": "Research with Web Search",
                    "agent": "research_assistant",
                    "interaction_data": {
                        "query": "Find the latest LangChain documentation and tutorials",
                        "context": "research_task",
                        "user_input": "I need current information about LangChain"
                    },
                    "expected_tool": "web_search"
                }
            ]
            
            learning_context = {
                "framework_name": "LangChain",
                "current_module": "research",
                "user_profile": {
                    "name": self.test_user_id,
                    "experience_level": "intermediate"
                },
                "session_id": f"tool_test_{int(datetime.now().timestamp())}"
            }
            
            tool_results = []
            
            for scenario in tool_scenarios:
                print(f"\n🔧 Testing: {scenario['name']}")
                
                # Check if required tools are available
                agent = self.agents.get(scenario["agent"])
                if not agent or len(agent.tools) == 0:
                    print(f"   ⚠️  Skipping - no tools available for {scenario['agent']}")
                    continue
                
                # Process interaction
                response = self.constellation_graph.process(
                    user_id=self.test_user_id,
                    interaction_data=scenario["interaction_data"],
                    learning_context=learning_context,
                    primary_agent=scenario["agent"]
                )
                
                # Check if tools were actually executed
                # This would require checking agent execution logs or response metadata
                content = response.get("content", "")
                
                # Look for indicators of actual tool execution vs just descriptions
                execution_indicators = [
                    "search results",
                    "found information",
                    "according to",
                    "based on search",
                    "retrieved data"
                ]
                
                description_indicators = [
                    "I would search",
                    "I could look up",
                    "I should find",
                    "I need to search"
                ]
                
                has_execution_indicators = any(indicator in content.lower() for indicator in execution_indicators)
                has_description_indicators = any(indicator in content.lower() for indicator in description_indicators)
                
                result = {
                    "scenario": scenario["name"],
                    "agent": scenario["agent"],
                    "tools_available": len(agent.tools),
                    "response_length": len(content),
                    "has_execution_indicators": has_execution_indicators,
                    "has_description_indicators": has_description_indicators,
                    "likely_executed_tools": has_execution_indicators and not has_description_indicators,
                    "status": "success"
                }
                
                tool_results.append(result)
                
                print(f"   ✅ {scenario['name']} completed")
                print(f"      - Tools available: {result['tools_available']}")
                print(f"      - Execution indicators: {result['has_execution_indicators']}")
                print(f"      - Description indicators: {result['has_description_indicators']}")
                print(f"      - Likely executed tools: {result['likely_executed_tools']}")
            
            return {"status": "success", "tool_results": tool_results}
            
        except Exception as e:
            print(f"❌ Tool execution verification failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def test_langgraph_state_management(self):
        """Test LangGraph state management in constellation"""
        print(f"\n{'='*50}")
        print("TESTING LANGGRAPH STATE MANAGEMENT")
        print(f"{'='*50}")
        
        try:
            # Test state persistence across interactions
            session_id = f"state_test_{int(datetime.now().timestamp())}"
            
            # First interaction
            print("\n1️⃣ First interaction...")
            response1 = self.constellation_graph.process(
                user_id=self.test_user_id,
                interaction_data={
                    "query": "Start learning LangChain basics",
                    "context": "session_start",
                    "user_input": "I want to begin learning LangChain"
                },
                learning_context={
                    "framework_name": "LangChain",
                    "current_module": "introduction",
                    "session_id": session_id,
                    "user_profile": {"name": self.test_user_id, "experience_level": "beginner"}
                },
                primary_agent="instructor"
            )
            
            # Second interaction (should maintain context)
            print("\n2️⃣ Second interaction...")
            response2 = self.constellation_graph.process(
                user_id=self.test_user_id,
                interaction_data={
                    "query": "Continue with the next topic",
                    "context": "session_continue",
                    "user_input": "What's next in my learning path?"
                },
                learning_context={
                    "framework_name": "LangChain",
                    "current_module": "introduction",
                    "session_id": session_id,
                    "user_profile": {"name": self.test_user_id, "experience_level": "beginner"}
                },
                primary_agent="instructor"
            )
            
            # Validate state management
            assert isinstance(response1, dict), "First response should be a dict"
            assert isinstance(response2, dict), "Second response should be a dict"
            
            # Check for context continuity indicators
            content2 = response2.get("content", "").lower()
            continuity_indicators = [
                "continue", "next", "following", "as we discussed",
                "building on", "moving forward", "now that"
            ]
            
            has_continuity = any(indicator in content2 for indicator in continuity_indicators)
            
            result = {
                "session_id": session_id,
                "interactions": 2,
                "response1_length": len(response1.get("content", "")),
                "response2_length": len(response2.get("content", "")),
                "has_continuity": has_continuity,
                "status": "success"
            }
            
            print(f"   ✅ State management test completed")
            print(f"      - Session ID: {session_id}")
            print(f"      - Interactions: {result['interactions']}")
            print(f"      - Context continuity: {result['has_continuity']}")
            
            return result
            
        except Exception as e:
            print(f"❌ LangGraph state management test failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def run_all_tests(self):
        """Run all agent coordination tests"""
        print(f"\n{'='*60}")
        print("STARTING AGENT COORDINATION TESTING")
        print(f"{'='*60}")
        
        test_results = {}
        
        try:
            # Test 1: Agent initialization
            agent_init_result = await self.test_agent_initialization()
            test_results["agent_initialization"] = agent_init_result
            
            if agent_init_result["status"] != "success":
                raise Exception("Agent initialization failed")
            
            # Test 2: Constellation graph setup
            graph_setup_result = await self.test_constellation_graph_setup()
            test_results["constellation_graph_setup"] = graph_setup_result
            
            if graph_setup_result["status"] != "success":
                raise Exception("Constellation graph setup failed")
            
            # Test 3: Basic agent interaction
            basic_interaction_result = await self.test_basic_agent_interaction()
            test_results["basic_agent_interaction"] = basic_interaction_result
            
            # Test 4: Agent handoff scenarios
            handoff_result = await self.test_agent_handoff_scenarios()
            test_results["agent_handoff_scenarios"] = handoff_result
            
            # Test 5: Tool execution verification
            tool_execution_result = await self.test_tool_execution_verification()
            test_results["tool_execution_verification"] = tool_execution_result
            
            # Test 6: LangGraph state management
            state_management_result = await self.test_langgraph_state_management()
            test_results["langgraph_state_management"] = state_management_result
            
            # Summary
            print(f"\n{'='*60}")
            print("AGENT COORDINATION TEST SUMMARY")
            print(f"{'='*60}")
            
            for test_name, result in test_results.items():
                status_icon = "✅" if result.get("status") == "success" else "❌"
                print(f"{status_icon} {test_name}: {result.get('status', 'unknown')}")
                if result.get("status") == "failed":
                    print(f"   Error: {result.get('error', 'Unknown error')}")
            
            # Save results
            result_file = Path("test_project") / f"agent_coordination_results_{int(datetime.now().timestamp())}.json"
            with open(result_file, 'w') as f:
                import json
                json.dump(test_results, f, indent=2)
            
            print(f"\n📁 Results saved to: {result_file}")
            
            return test_results
            
        except Exception as e:
            print(f"💥 Critical error in agent coordination testing: {str(e)}")
            test_results["critical_error"] = {"status": "failed", "error": str(e)}
            return test_results


async def main():
    """Main test execution"""
    print("🚀 Starting Agent Coordination Testing...")
    
    # Check required environment variables
    required_vars = ["GOOGLE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   At least one LLM API key is required")
        return
    
    tester = AgentCoordinationTester()
    results = await tester.run_all_tests()
    
    print(f"\n🏁 Agent coordination testing completed!")
    return results


if __name__ == "__main__":
    asyncio.run(main()) 