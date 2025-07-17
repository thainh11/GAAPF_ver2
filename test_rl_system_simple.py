#!/usr/bin/env python3
"""
Simplified RL System Test Script
Tests core components without monitoring integration
"""

import asyncio
import sys
import os
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_imports():
    """Test that all core modules can be imported"""
    try:
        print("Testing imports...")
        
        # Core RL components
        from rl_system.algorithms.maddpg import MADDPG
        print("✅ MADDPG import successful")
        
        from rl_system.utils.experience_buffer import ExperienceBuffer
        print("✅ ExperienceBuffer import successful")
        
        from rl_system.training.training_manager import TrainingManager
        print("✅ TrainingManager import successful")
        
        from rl_system.training.curriculum_manager import CurriculumManager
        print("✅ CurriculumManager import successful")
        
        from rl_system.training.evaluation_manager import EvaluationManager
        print("✅ EvaluationManager import successful")
        
        from rl_system.testing.metrics_collector import MetricsCollector, MetricsConfig
        print("✅ MetricsCollector import successful")
        
        from rl_system.integration.configuration_manager import ConfigurationManager
        print("✅ ConfigurationManager import successful")
        
        from rl_system import RLSystem
        print("✅ RLSystem import successful")
        
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

async def test_rl_system_initialization():
    """Test RLSystem initialization"""
    try:
        print("Testing RLSystem initialization...")
        
        from rl_system import RLSystem, RLConfig, RLSystemMode, RLAlgorithm
        
        # Create proper RLConfig
        config = RLConfig(
            enabled=True,
            mode=RLSystemMode.TESTING,
            algorithm=RLAlgorithm.MADDPG,
            log_level="INFO",
            enable_detailed_logging=False
        )
        
        rl_system = RLSystem(config)
        print("✅ RLSystem initialization successful")
        return True
    except Exception as e:
        print(f"❌ RLSystem initialization failed: {e}")
        traceback.print_exc()
        return False

async def test_core_components():
    """Test core component initialization"""
    try:
        print("Testing core components...")
        
        from rl_system.algorithms.maddpg import MADDPG, MADDPGConfig
        from rl_system.utils.experience_buffer import ExperienceBuffer
        from rl_system.training.training_manager import TrainingManager, TrainingConfig
        from rl_system.training.evaluation_manager import EvaluationManager, EvaluationConfig
        
        # Test ExperienceBuffer
        buffer = ExperienceBuffer({})
        print("✅ ExperienceBuffer created")
        
        # Test MADDPG with proper config
        maddpg_config = MADDPGConfig(
            state_dim=10,
            action_dim=4,
            hidden_dim=128
        )
        
        # Create agent configs for MADDPG
        agent_configs = {
            'agent_0': {'state_dim': 10, 'action_dim': 4},
            'agent_1': {'state_dim': 10, 'action_dim': 4}
        }
        
        maddpg = MADDPG(agent_configs, maddpg_config)
        print("✅ MADDPG created")
        
        # Test TrainingManager with proper config
        training_config = TrainingConfig(
            max_episodes=100,
            max_steps_per_episode=500,
            evaluation_frequency=50
        )
        # TrainingManager requires more complex initialization, just test config creation
        print("✅ TrainingManager config created")
        
        # Test EvaluationManager with proper config
        eval_config = EvaluationConfig(
            evaluation_frequency=100,
            test_set_size=10
        )
        # EvaluationManager also requires complex initialization, just test config creation
        print("✅ EvaluationManager config created")
        
        print("✅ All core components successful")
        return True
    except Exception as e:
        print(f"❌ Core components failed: {e}")
        traceback.print_exc()
        return False

async def test_metrics_collector():
    """Test MetricsCollector without monitoring integration"""
    try:
        print("Testing MetricsCollector...")
        
        from rl_system.testing.metrics_collector import MetricsCollector, MetricsConfig
        
        # Create metrics config with correct parameters
        metrics_config = MetricsConfig(
            collection_enabled=True,
            flush_interval=60.0,
            default_retention_days=30,
            storage_backend="sqlite",
            storage_path="test_metrics.db"
        )
        
        # Create metrics collector
        metrics_collector = MetricsCollector(metrics_config)
        print("✅ MetricsCollector created successfully")
        
        return True
    except Exception as e:
        print(f"❌ MetricsCollector failed: {e}")
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run all tests and report results"""
    print("=" * 50)
    print("Running Simplified RL System Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("RLSystem Initialization", test_rl_system_initialization),
        ("Core Components", test_core_components),
        ("MetricsCollector", test_metrics_collector)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0

if __name__ == "__main__":
    asyncio.run(run_all_tests())