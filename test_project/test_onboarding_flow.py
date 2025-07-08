#!/usr/bin/env python3
"""
Focused Test for Onboarding Flow

This test specifically focuses on:
1. Framework selection from supported list
2. User configuration setup
3. Tavily tools integration for information collection
4. Initial curriculum generation
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.GAAPF.core.core.framework_onboarding import FrameworkOnboarding, SUPPORTED_FRAMEWORKS
from src.GAAPF.core.core.knowledge_graph import KnowledgeGraph
from src.GAAPF.core.memory.long_term_memory import LongTermMemory
from src.GAAPF.core.tools.framework_collector import FrameworkCollector

# Import LLM
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI
    print("✅ LLM imports successful")
except ImportError as e:
    print(f"❌ LLM import error: {e}")
    sys.exit(1)


class OnboardingFlowTester:
    """Focused tester for onboarding flow"""
    
    def __init__(self):
        self.test_user_id = "test_onboarding_user"
        self.llm = self._initialize_llm()
        print("✅ Onboarding flow tester initialized")
    
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
    
    async def test_framework_selection(self):
        """Test framework selection functionality"""
        print(f"\n{'='*50}")
        print("TESTING FRAMEWORK SELECTION")
        print(f"{'='*50}")
        
        # Test 1: Get supported frameworks
        print("\n📋 Testing supported frameworks list...")
        supported_frameworks = SUPPORTED_FRAMEWORKS
        
        print(f"✅ Found {len(supported_frameworks)} supported frameworks:")
        for i, framework in enumerate(supported_frameworks, 1):
            print(f"   {i}. {framework}")
        
        # Verify all required frameworks are present
        required_frameworks = [
            "LangChain",
            "LangGraph", 
            "Microsoft AutoGen",
            "CrewAI",
            "Haystack by Deepset",
            "Hugging Face SmolAgents",
            "OpenAI Agents Python"
        ]
        
        for framework in required_frameworks:
            assert framework in supported_frameworks, f"Missing required framework: {framework}"
        
        print("✅ All required frameworks are supported")
        return supported_frameworks
    
    async def test_user_configuration(self):
        """Test user configuration setup"""
        print(f"\n{'='*50}")
        print("TESTING USER CONFIGURATION")
        print(f"{'='*50}")
        
        # Test different user configurations
        test_configs = [
            {
                "name": "Beginner Configuration",
                "config": {
                    "experience_level": "beginner",
                    "goals": ["learn_basics"],
                    "learning_style": "visual",
                    "time_commitment": "light"
                }
            },
            {
                "name": "Intermediate Configuration",
                "config": {
                    "experience_level": "intermediate",
                    "goals": ["learn_concepts", "build_production_app"],
                    "learning_style": "hands_on",
                    "time_commitment": "moderate"
                }
            },
            {
                "name": "Advanced Configuration",
                "config": {
                    "experience_level": "advanced",
                    "goals": ["research", "teach_others"],
                    "learning_style": "theoretical",
                    "time_commitment": "intensive"
                }
            }
        ]
        
        for test_config in test_configs:
            print(f"\n🔧 Testing {test_config['name']}...")
            config = test_config["config"]
            
            # Validate configuration structure
            assert "experience_level" in config, "Missing experience_level"
            assert "goals" in config, "Missing goals"
            assert isinstance(config["goals"], list), "Goals should be a list"
            assert len(config["goals"]) > 0, "Goals list should not be empty"
            
            print(f"   ✅ {test_config['name']} validated")
            print(f"      - Experience: {config['experience_level']}")
            print(f"      - Goals: {', '.join(config['goals'])}")
            print(f"      - Style: {config.get('learning_style', 'default')}")
        
        return test_configs
    
    async def test_tavily_integration(self):
        """Test Tavily tools integration"""
        print(f"\n{'='*50}")
        print("TESTING TAVILY INTEGRATION")
        print(f"{'='*50}")
        
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            print("⚠️  TAVILY_API_KEY not found - skipping Tavily integration test")
            return {"status": "skipped", "reason": "missing_api_key"}
        
        try:
            # Initialize memory and knowledge graph
            memory = LongTermMemory(
                user_id=self.test_user_id,
                collection_name="tavily_test",
                embedding_model="google",
                is_logging=True
            )
            
            knowledge_graph = KnowledgeGraph()
            
            # Initialize framework collector
            collector = FrameworkCollector(
                memory=memory,
                is_logging=True,
                tavily_api_key=tavily_api_key
            )
            
            print("✅ Framework collector initialized with Tavily")
            
            # Test 1: Search framework information
            print("\n🔍 Testing framework information search...")
            search_results = await collector.search_framework_information(
                framework_name="LangChain",
                search_queries=["LangChain tutorial basics", "LangChain documentation"]
            )
            
            assert isinstance(search_results, list), "Search results should be a list"
            print(f"✅ Search completed - found {len(search_results)} results")
            
            # Test 2: Extract concepts (if we have results)
            if search_results and len(search_results) > 0:
                print("\n📊 Testing concept extraction...")
                extracted_concepts = await collector.extract_framework_concepts(
                    search_results=search_results[:2],  # Use first 2 results
                    framework_name="LangChain"
                )
                
                assert isinstance(extracted_concepts, dict), "Extracted concepts should be a dict"
                print(f"✅ Concept extraction completed")
                print(f"   - Concepts found: {len(extracted_concepts.get('concepts', []))}")
            
            return {"status": "success", "results_count": len(search_results)}
            
        except Exception as e:
            print(f"❌ Tavily integration test failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def test_curriculum_generation(self):
        """Test initial curriculum generation"""
        print(f"\n{'='*50}")
        print("TESTING CURRICULUM GENERATION")
        print(f"{'='*50}")
        
        try:
            # Initialize components
            memory = LongTermMemory(
                user_id=self.test_user_id,
                collection_name="curriculum_test",
                embedding_model="google",
                is_logging=True
            )
            
            knowledge_graph = KnowledgeGraph()
            
            # Initialize framework onboarding
            onboarding = FrameworkOnboarding(
                memory=memory,
                knowledge_graph=knowledge_graph,
                is_logging=True,
                tavily_api_key=os.getenv("TAVILY_API_KEY")
            )
            
            # Test curriculum generation for different frameworks
            test_frameworks = ["LangChain", "CrewAI", "LangGraph"]
            
            for framework in test_frameworks:
                print(f"\n🎓 Testing curriculum generation for {framework}...")
                
                user_config = {
                    "experience_level": "intermediate",
                    "goals": ["learn_concepts", "build_production_app"],
                    "learning_style": "hands_on",
                    "time_commitment": "moderate"
                }
                
                # Generate curriculum
                curriculum = await onboarding.initialize_framework(
                    framework_name=framework,
                    user_id=self.test_user_id,
                    user_config=user_config,
                    initialization_mode="quick"
                )
                
                # Validate curriculum structure
                assert isinstance(curriculum, dict), "Curriculum should be a dictionary"
                assert "framework_name" in curriculum, "Missing framework_name"
                assert "modules" in curriculum, "Missing modules"
                assert curriculum["framework_name"] == framework, "Framework name mismatch"
                
                print(f"   ✅ {framework} curriculum generated successfully")
                print(f"      - Modules: {len(curriculum.get('modules', []))}")
                print(f"      - Difficulty: {curriculum.get('difficulty_level', 'Unknown')}")
            
            return {"status": "success", "frameworks_tested": len(test_frameworks)}
            
        except Exception as e:
            print(f"❌ Curriculum generation test failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def test_complete_onboarding_flow(self):
        """Test the complete onboarding flow end-to-end"""
        print(f"\n{'='*50}")
        print("TESTING COMPLETE ONBOARDING FLOW")
        print(f"{'='*50}")
        
        try:
            # Step 1: Framework selection
            print("\n1️⃣ Framework Selection...")
            frameworks = await self.test_framework_selection()
            selected_framework = "LangChain"  # Choose LangChain for testing
            print(f"   Selected framework: {selected_framework}")
            
            # Step 2: User configuration
            print("\n2️⃣ User Configuration...")
            configs = await self.test_user_configuration()
            user_config = configs[1]["config"]  # Use intermediate config
            print(f"   Selected config: {configs[1]['name']}")
            
            # Step 3: Initialize components
            print("\n3️⃣ Component Initialization...")
            memory = LongTermMemory(
                user_id=self.test_user_id,
                collection_name="complete_onboarding_test",
                embedding_model="google",
                is_logging=True
            )
            
            knowledge_graph = KnowledgeGraph()
            
            onboarding = FrameworkOnboarding(
                memory=memory,
                knowledge_graph=knowledge_graph,
                is_logging=True,
                tavily_api_key=os.getenv("TAVILY_API_KEY")
            )
            print("   ✅ Components initialized")
            
            # Step 4: Framework initialization with information collection
            print("\n4️⃣ Framework Initialization with Information Collection...")
            curriculum = await onboarding.initialize_framework(
                framework_name=selected_framework,
                user_id=self.test_user_id,
                user_config=user_config,
                initialization_mode="quick",
                is_background_collection=True
            )
            
            # Validate final result
            assert isinstance(curriculum, dict), "Final curriculum should be a dictionary"
            assert curriculum["framework_name"] == selected_framework, "Framework name mismatch"
            assert len(curriculum.get("modules", [])) > 0, "Curriculum should have modules"
            
            print("   ✅ Framework initialization completed")
            print(f"      - Framework: {curriculum['framework_name']}")
            print(f"      - Modules: {len(curriculum.get('modules', []))}")
            print(f"      - Difficulty: {curriculum.get('difficulty_level', 'Unknown')}")
            
            # Step 5: Save results
            print("\n5️⃣ Saving Results...")
            result_file = Path("test_project") / f"onboarding_result_{int(datetime.now().timestamp())}.json"
            
            result_data = {
                "test_timestamp": datetime.now().isoformat(),
                "selected_framework": selected_framework,
                "user_config": user_config,
                "curriculum": curriculum,
                "test_status": "success"
            }
            
            with open(result_file, 'w') as f:
                import json
                json.dump(result_data, f, indent=2)
            
            print(f"   ✅ Results saved to: {result_file}")
            
            return {"status": "success", "curriculum": curriculum, "result_file": str(result_file)}
            
        except Exception as e:
            print(f"❌ Complete onboarding flow test failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def run_all_tests(self):
        """Run all onboarding flow tests"""
        print(f"\n{'='*60}")
        print("STARTING ONBOARDING FLOW TESTING")
        print(f"{'='*60}")
        
        test_results = {}
        
        try:
            # Test 1: Framework selection
            await self.test_framework_selection()
            test_results["framework_selection"] = {"status": "success"}
            
            # Test 2: User configuration
            await self.test_user_configuration()
            test_results["user_configuration"] = {"status": "success"}
            
            # Test 3: Tavily integration
            tavily_result = await self.test_tavily_integration()
            test_results["tavily_integration"] = tavily_result
            
            # Test 4: Curriculum generation
            curriculum_result = await self.test_curriculum_generation()
            test_results["curriculum_generation"] = curriculum_result
            
            # Test 5: Complete flow
            complete_result = await self.test_complete_onboarding_flow()
            test_results["complete_flow"] = complete_result
            
            # Summary
            print(f"\n{'='*60}")
            print("ONBOARDING FLOW TEST SUMMARY")
            print(f"{'='*60}")
            
            for test_name, result in test_results.items():
                status_icon = "✅" if result.get("status") == "success" else "⚠️" if result.get("status") == "skipped" else "❌"
                print(f"{status_icon} {test_name}: {result.get('status', 'unknown')}")
                if result.get("status") == "failed":
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                elif result.get("status") == "skipped":
                    print(f"   Reason: {result.get('reason', 'Unknown reason')}")
            
            return test_results
            
        except Exception as e:
            print(f"💥 Critical error in onboarding flow testing: {str(e)}")
            test_results["critical_error"] = {"status": "failed", "error": str(e)}
            return test_results


async def main():
    """Main test execution"""
    print("🚀 Starting Onboarding Flow Testing...")
    
    # Check required environment variables
    required_vars = ["GOOGLE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   At least one LLM API key is required")
        return
    
    tester = OnboardingFlowTester()
    results = await tester.run_all_tests()
    
    print(f"\n🏁 Onboarding flow testing completed!")
    return results


if __name__ == "__main__":
    asyncio.run(main()) 