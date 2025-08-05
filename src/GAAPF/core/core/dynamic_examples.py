#!/usr/bin/env python3
"""
Dynamic Capabilities Examples

This module provides comprehensive examples of how to use the new LLM-driven
dynamic capabilities in the GAAPF framework. It demonstrates integration
patterns, migration strategies, and best practices.

Key Examples:
- Basic dynamic constellation generation
- Learning parameter calibration
- Intelligent recommendations
- Progressive migration from fixed to dynamic
- Performance monitoring and optimization
- Error handling and fallback strategies

Author: AI Assistant
Date: 2024
"""

import logging
import time
from typing import Dict, List, Optional, Any

try:
    from langchain.schema import BaseLanguageModel
    from langchain.llms import OpenAI
except ImportError:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_openai import OpenAI

from .dynamic_integration import (
    DynamicIntegrationManager,
    IntegrationMode,
    DynamicCapabilities,
    create_dynamic_integration_manager,
    get_constellation_with_dynamic_capabilities,
    get_enhanced_learning_experience
)
from .enhanced_constellation_manager import (
    EnhancedConstellationManager,
    create_enhanced_constellation_manager,
    get_optimal_constellation_for_context
)
from .constellation_types import get_constellation_type, CONSTELLATION_TYPES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicCapabilitiesDemo:
    """
    Demonstration class showing how to use dynamic capabilities.
    """
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.integration_manager = None
        self.examples_run = []
    
    def example_1_basic_dynamic_constellation(self):
        """
        Example 1: Basic dynamic constellation generation.
        
        This example shows how to generate a custom constellation
        based on learning context using LLM intelligence.
        """
        print("\n" + "="*60)
        print("EXAMPLE 1: Basic Dynamic Constellation Generation")
        print("="*60)
        
        # Sample learning context
        learning_context = {
            "subject": "Python Programming",
            "topic": "Object-Oriented Programming",
            "learning_stage": "intermediate",
            "difficulty_level": "medium",
            "learning_style": "hands-on",
            "current_activity": "learning new concepts",
            "time_available": 45,  # minutes
            "user_goals": ["understand classes", "practice inheritance"]
        }
        
        user_query = "I want to learn about Python classes and inheritance with practical examples"
        
        try:
            # Get dynamic constellation
            result = get_constellation_with_dynamic_capabilities(
                llm=self.llm,
                learning_context=learning_context,
                user_query=user_query,
                integration_mode=IntegrationMode.DYNAMIC_PREFERRED
            )
            
            print(f"✅ Constellation Type: {result.get('constellation_type', 'unknown')}")
            print(f"✅ Source: {result.get('source', 'unknown')}")
            print(f"✅ Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"✅ Integration Mode: {result.get('integration_mode', 'unknown')}")
            
            if 'learning_parameters' in result:
                params = result['learning_parameters']
                print("\n📊 Calibrated Learning Parameters:")
                for param, value in params.items():
                    print(f"   {param}: {value:.2f}")
            
            if 'constellation' in result:
                constellation = result['constellation']
                print(f"\n🎯 Constellation Details:")
                print(f"   Name: {constellation.get('name', 'Unknown')}")
                print(f"   Description: {constellation.get('description', 'No description')[:100]}...")
                
                if 'agents' in constellation:
                    print(f"   Agents: {len(constellation['agents'])} configured")
                    for i, agent in enumerate(constellation['agents'][:3]):  # Show first 3
                        print(f"     {i+1}. {agent.get('role', 'Unknown Role')} - {agent.get('type', 'Unknown Type')}")
            
            self.examples_run.append("basic_dynamic_constellation")
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def example_2_learning_parameter_calibration(self):
        """
        Example 2: Dynamic learning parameter calibration.
        
        This example shows how to calibrate learning parameters
        based on user performance and context.
        """
        print("\n" + "="*60)
        print("EXAMPLE 2: Learning Parameter Calibration")
        print("="*60)
        
        # Create integration manager
        manager = create_dynamic_integration_manager(
            llm=self.llm,
            integration_mode=IntegrationMode.HYBRID,
            is_logging=True
        )
        
        # Sample learning context with performance data
        learning_context = {
            "subject": "Data Science",
            "topic": "Machine Learning Algorithms",
            "learning_stage": "advanced",
            "difficulty_level": "high",
            "learning_style": "analytical",
            "previous_topics": ["statistics", "python", "pandas"]
        }
        
        # Sample performance data
        performance_data = {
            "recent_scores": [0.85, 0.78, 0.92, 0.88],
            "completion_rate": 0.87,
            "time_per_task": 12.5,  # minutes
            "difficulty_preference": "challenging",
            "learning_velocity": 0.75,
            "areas_of_struggle": ["complex algorithms", "mathematical concepts"]
        }
        
        # Sample user feedback
        user_feedback = {
            "pace_feedback": "too_slow",
            "difficulty_feedback": "appropriate",
            "content_depth_feedback": "need_more_detail",
            "preferred_explanation_style": "step_by_step"
        }
        
        try:
            # Calibrate learning parameters
            calibrated_params = manager.calibrate_learning_parameters(
                learning_context=learning_context,
                user_id="demo_user",
                performance_data=performance_data,
                user_feedback=user_feedback
            )
            
            print("📊 Calibrated Learning Parameters:")
            for param, value in calibrated_params.items():
                print(f"   {param}: {value:.3f}")
            
            # Show how parameters adapt to performance
            print("\n🎯 Parameter Adaptations:")
            if calibrated_params.get('pacing', 0) > 0.7:
                print("   ⚡ Increased pacing due to high performance")
            if calibrated_params.get('difficulty', 0) > 0.6:
                print("   📈 Increased difficulty for challenge")
            if calibrated_params.get('content_depth', 0) > 0.7:
                print("   🔍 Increased content depth based on feedback")
            
            self.examples_run.append("parameter_calibration")
            return calibrated_params
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def example_3_intelligent_recommendations(self):
        """
        Example 3: Intelligent learning recommendations.
        
        This example shows how to get personalized learning
        activity and path recommendations.
        """
        print("\n" + "="*60)
        print("EXAMPLE 3: Intelligent Learning Recommendations")
        print("="*60)
        
        # Create integration manager
        manager = create_dynamic_integration_manager(
            llm=self.llm,
            integration_mode=IntegrationMode.DYNAMIC_PREFERRED
        )
        
        # Sample learning context
        learning_context = {
            "subject": "Web Development",
            "topic": "React.js",
            "learning_stage": "beginner",
            "difficulty_level": "easy",
            "learning_style": "visual",
            "time_available": 60,  # minutes
            "preferred_activities": ["interactive_coding", "video_tutorials"]
        }
        
        try:
            # Get activity recommendations
            activity_recommendations = manager.get_learning_recommendations(
                learning_context=learning_context,
                constellation_type="learning",
                user_query="I want to start learning React with hands-on practice",
                session_goals=["understand components", "create first app"]
            )
            
            print("🎯 Activity Recommendations:")
            if 'recommended_activities' in activity_recommendations:
                for i, activity in enumerate(activity_recommendations['recommended_activities'], 1):
                    print(f"   {i}. {activity.get('name', 'Unknown Activity')}")
                    print(f"      Type: {activity.get('type', 'unknown')}")
                    print(f"      Duration: {activity.get('duration', 0)} minutes")
                    if 'description' in activity:
                        print(f"      Description: {activity['description'][:80]}...")
                    print()
            
            # Get learning path recommendations
            learning_objectives = [
                "Understand React components",
                "Learn JSX syntax",
                "Master state management",
                "Build interactive applications"
            ]
            
            learning_path = manager.get_personalized_learning_path(
                learning_context=learning_context,
                learning_objectives=learning_objectives,
                time_constraints={"total_time": "2 weeks", "daily_time": "1 hour"}
            )
            
            print("🗺️ Personalized Learning Path:")
            if 'learning_path' in learning_path:
                path = learning_path['learning_path']
                print(f"   Path Name: {path.get('path_name', 'Unknown')}")
                print(f"   Total Duration: {path.get('total_duration', 'Unknown')}")
                
                if 'learning_modules' in path:
                    print("\n   📚 Learning Modules:")
                    for i, module in enumerate(path['learning_modules'], 1):
                        print(f"      {i}. {module.get('module_name', 'Unknown Module')}")
                        print(f"         Duration: {module.get('estimated_duration', 'Unknown')}")
                        print(f"         Constellation: {module.get('recommended_constellation', 'learning')}")
                        if 'learning_objectives' in module:
                            print(f"         Objectives: {', '.join(module['learning_objectives'])}")
                        print()
            
            self.examples_run.append("intelligent_recommendations")
            return {
                "activities": activity_recommendations,
                "learning_path": learning_path
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def example_4_progressive_migration(self):
        """
        Example 4: Progressive migration from fixed to dynamic.
        
        This example shows how to gradually migrate from fixed
        constellation types to dynamic generation.
        """
        print("\n" + "="*60)
        print("EXAMPLE 4: Progressive Migration Strategy")
        print("="*60)
        
        learning_context = {
            "subject": "Mathematics",
            "topic": "Calculus",
            "learning_stage": "intermediate",
            "difficulty_level": "medium"
        }
        
        user_query = "Help me understand derivatives with step-by-step examples"
        
        # Step 1: Start with fixed-only mode
        print("🔧 Step 1: Fixed-Only Mode (Current State)")
        try:
            fixed_result = get_constellation_with_dynamic_capabilities(
                llm=self.llm,
                learning_context=learning_context,
                user_query=user_query,
                integration_mode=IntegrationMode.FIXED_ONLY
            )
            print(f"   Result: {fixed_result.get('source', 'unknown')} constellation")
            print(f"   Type: {fixed_result.get('constellation_type', 'unknown')}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Step 2: Hybrid mode (recommended for migration)
        print("\n🔄 Step 2: Hybrid Mode (Migration Phase)")
        try:
            hybrid_result = get_constellation_with_dynamic_capabilities(
                llm=self.llm,
                learning_context=learning_context,
                user_query=user_query,
                integration_mode=IntegrationMode.HYBRID
            )
            print(f"   Result: {hybrid_result.get('source', 'unknown')} constellation")
            print(f"   Selection Reason: {hybrid_result.get('selection_reason', 'unknown')}")
            print(f"   Enhanced: {hybrid_result.get('enhanced', False)}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Step 3: Dynamic-preferred mode
        print("\n⚡ Step 3: Dynamic-Preferred Mode (Advanced)")
        try:
            dynamic_result = get_constellation_with_dynamic_capabilities(
                llm=self.llm,
                learning_context=learning_context,
                user_query=user_query,
                integration_mode=IntegrationMode.DYNAMIC_PREFERRED
            )
            print(f"   Result: {dynamic_result.get('source', 'unknown')} constellation")
            print(f"   Capabilities Used: {', '.join(dynamic_result.get('capabilities_used', []))}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Step 4: Show migration benefits
        print("\n📊 Migration Benefits:")
        print("   ✅ Gradual adoption reduces risk")
        print("   ✅ Fallback mechanisms ensure reliability")
        print("   ✅ Performance monitoring guides optimization")
        print("   ✅ User experience improves progressively")
        
        self.examples_run.append("progressive_migration")
    
    def example_5_complete_learning_experience(self):
        """
        Example 5: Complete enhanced learning experience.
        
        This example shows how to get a complete learning experience
        with all dynamic features in a single call.
        """
        print("\n" + "="*60)
        print("EXAMPLE 5: Complete Enhanced Learning Experience")
        print("="*60)
        
        learning_context = {
            "subject": "Artificial Intelligence",
            "topic": "Neural Networks",
            "learning_stage": "advanced",
            "difficulty_level": "high",
            "learning_style": "theoretical_and_practical",
            "time_available": 90,  # minutes
            "background_knowledge": ["linear_algebra", "calculus", "python"],
            "learning_goals": ["understand backpropagation", "implement from scratch"]
        }
        
        user_query = "I want to deeply understand neural networks and implement one from scratch"
        
        try:
            # Get complete enhanced experience
            enhanced_experience = get_enhanced_learning_experience(
                llm=self.llm,
                learning_context=learning_context,
                user_query=user_query,
                user_id="advanced_learner"
            )
            
            print("🎯 Complete Learning Experience Generated:")
            
            # Constellation details
            constellation = enhanced_experience.get('constellation', {})
            print(f"\n🏗️ Constellation:")
            print(f"   Type: {constellation.get('constellation_type', 'unknown')}")
            print(f"   Source: {constellation.get('source', 'unknown')}")
            print(f"   Integration Mode: {constellation.get('integration_mode', 'unknown')}")
            
            # Activity recommendations
            activities = enhanced_experience.get('activity_recommendations', {})
            print(f"\n📋 Activity Recommendations:")
            if 'recommended_activities' in activities:
                for i, activity in enumerate(activities['recommended_activities'][:3], 1):
                    print(f"   {i}. {activity.get('name', 'Unknown')} ({activity.get('duration', 0)} min)")
            
            # Learning path
            learning_path = enhanced_experience.get('learning_path', {})
            print(f"\n🗺️ Learning Path:")
            if 'learning_path' in learning_path:
                path = learning_path['learning_path']
                print(f"   Path: {path.get('path_name', 'Unknown')}")
                print(f"   Duration: {path.get('total_duration', 'Unknown')}")
                print(f"   Modules: {len(path.get('learning_modules', []))}")
            
            print(f"\n✨ Enhanced Experience: {enhanced_experience.get('enhanced_experience', False)}")
            
            self.examples_run.append("complete_experience")
            return enhanced_experience
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def example_6_performance_monitoring(self):
        """
        Example 6: Performance monitoring and optimization.
        
        This example shows how to monitor the performance of
        dynamic capabilities and optimize usage.
        """
        print("\n" + "="*60)
        print("EXAMPLE 6: Performance Monitoring")
        print("="*60)
        
        # Create manager with monitoring enabled
        manager = DynamicIntegrationManager(
            llm=self.llm,
            integration_mode=IntegrationMode.HYBRID,
            is_logging=True
        )
        
        # Simulate multiple requests
        contexts = [
            {"subject": "Python", "topic": "Functions", "learning_stage": "beginner"},
            {"subject": "JavaScript", "topic": "Async/Await", "learning_stage": "intermediate"},
            {"subject": "Machine Learning", "topic": "Deep Learning", "learning_stage": "advanced"}
        ]
        
        queries = [
            "How do I write functions in Python?",
            "Explain async/await in JavaScript",
            "I need help with deep learning concepts"
        ]
        
        print("🔄 Simulating multiple requests...")
        
        for i, (context, query) in enumerate(zip(contexts, queries), 1):
            try:
                result = manager.get_constellation_for_context(
                    learning_context=context,
                    user_query=query,
                    user_id=f"user_{i}"
                )
                print(f"   Request {i}: {result.get('source', 'unknown')} ({result.get('processing_time', 0):.2f}s)")
            except Exception as e:
                print(f"   Request {i}: ❌ Error - {e}")
        
        # Get performance statistics
        stats = manager.get_integration_stats()
        
        print("\n📊 Performance Statistics:")
        print(f"   Total Requests: {stats.get('total_requests', 0)}")
        print(f"   Dynamic Used: {stats.get('dynamic_used', 0)} ({stats.get('dynamic_percentage', 0):.1f}%)")
        print(f"   Fixed Used: {stats.get('fixed_used', 0)} ({stats.get('fixed_percentage', 0):.1f}%)")
        print(f"   Hybrid Used: {stats.get('hybrid_used', 0)} ({stats.get('hybrid_percentage', 0):.1f}%)")
        print(f"   Fallback Used: {stats.get('fallback_used', 0)} ({stats.get('fallback_percentage', 0):.1f}%)")
        print(f"   Error Rate: {stats.get('error_rate', 0):.1f}%")
        
        # Performance recommendations
        print("\n💡 Performance Recommendations:")
        if stats.get('error_rate', 0) > 10:
            print("   ⚠️ High error rate - consider increasing fallback usage")
        if stats.get('dynamic_percentage', 0) < 30:
            print("   📈 Low dynamic usage - consider optimizing LLM performance")
        if stats.get('fallback_percentage', 0) > 20:
            print("   🔧 High fallback usage - investigate dynamic generation issues")
        
        self.examples_run.append("performance_monitoring")
        return stats
    
    def run_all_examples(self):
        """
        Run all examples in sequence.
        """
        print("🚀 Running All Dynamic Capabilities Examples")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run all examples
        examples = [
            self.example_1_basic_dynamic_constellation,
            self.example_2_learning_parameter_calibration,
            self.example_3_intelligent_recommendations,
            self.example_4_progressive_migration,
            self.example_5_complete_learning_experience,
            self.example_6_performance_monitoring
        ]
        
        results = {}
        for example in examples:
            try:
                result = example()
                results[example.__name__] = result
            except Exception as e:
                print(f"❌ Example {example.__name__} failed: {e}")
                results[example.__name__] = None
        
        total_time = time.time() - start_time
        
        # Summary
        print("\n" + "="*80)
        print("📋 EXAMPLES SUMMARY")
        print("="*80)
        print(f"Total Examples Run: {len(self.examples_run)}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Examples Completed: {', '.join(self.examples_run)}")
        
        successful = sum(1 for r in results.values() if r is not None)
        print(f"Success Rate: {successful}/{len(examples)} ({(successful/len(examples)*100):.1f}%)")
        
        return results


def run_integration_examples(llm: Optional[BaseLanguageModel] = None):
    """
    Run integration examples with a provided or mock LLM.
    
    Parameters:
    ----------
    llm : BaseLanguageModel, optional
        Language model to use. If None, will attempt to create a mock LLM.
    """
    if llm is None:
        print("⚠️ No LLM provided. Creating mock LLM for demonstration.")
        print("   In real usage, provide a proper LLM instance.")
        
        # Create a simple mock LLM for demonstration
        class MockLLM:
            def invoke(self, prompt, **kwargs):
                return "Mock LLM response for demonstration purposes."
            
            def __call__(self, prompt, **kwargs):
                return self.invoke(prompt, **kwargs)
        
        llm = MockLLM()
    
    # Create demo instance and run examples
    demo = DynamicCapabilitiesDemo(llm)
    return demo.run_all_examples()


def demonstrate_migration_strategy():
    """
    Demonstrate a complete migration strategy from fixed to dynamic.
    """
    print("\n" + "="*80)
    print("🔄 MIGRATION STRATEGY DEMONSTRATION")
    print("="*80)
    
    print("""
    PHASE 1: Assessment (Current State)
    ✅ Identify fixed constellation types in use
    ✅ Analyze current performance and limitations
    ✅ Define migration goals and success criteria
    
    PHASE 2: Preparation
    ✅ Set up LLM integration
    ✅ Configure dynamic components
    ✅ Implement monitoring and fallback mechanisms
    
    PHASE 3: Pilot (Hybrid Mode)
    ✅ Enable hybrid mode for low-risk scenarios
    ✅ Monitor performance and user feedback
    ✅ Gradually increase dynamic usage
    
    PHASE 4: Optimization
    ✅ Fine-tune LLM prompts and parameters
    ✅ Optimize caching and performance
    ✅ Expand to more complex scenarios
    
    PHASE 5: Full Deployment
    ✅ Switch to dynamic-preferred mode
    ✅ Maintain fallback mechanisms
    ✅ Continuous monitoring and improvement
    """)
    
    print("\n💡 Key Success Factors:")
    print("   🎯 Start with low-complexity scenarios")
    print("   📊 Monitor performance metrics continuously")
    print("   🔄 Maintain robust fallback mechanisms")
    print("   👥 Gather user feedback throughout migration")
    print("   ⚡ Optimize LLM performance iteratively")


if __name__ == "__main__":
    # Run examples if script is executed directly
    print("🚀 Dynamic Capabilities Examples")
    print("This script demonstrates the new LLM-driven dynamic capabilities.")
    print("\nNote: Requires a proper LLM instance for full functionality.")
    
    # Run with mock LLM for demonstration
    results = run_integration_examples()
    
    # Show migration strategy
    demonstrate_migration_strategy()
    
    print("\n✅ Examples completed. Check the output above for detailed results.")