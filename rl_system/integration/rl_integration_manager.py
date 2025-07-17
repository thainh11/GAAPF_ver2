"""RL Integration Manager for GAAPF System

This module provides the main integration manager that coordinates
the seamless integration of RL capabilities with the existing GAAPF framework.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime

# RL System imports
from ..utils.reward_system import RewardCalculator, RewardShaper
from ..utils.experience_buffer import ExperienceBuffer
from ..agents.rl_specialized_agent import RLSpecializedAgent
from ..managers.rl_constellation_manager import RLConstellationManager
from ..training.training_manager import TrainingManager
from ..testing.metrics_collector import MetricsCollector
from ..testing.ab_testing import ABTestManager
from ..testing.performance_comparison import PerformanceComparator

logger = logging.getLogger(__name__)

class IntegrationMode(Enum):
    """Integration modes"""
    DISABLED = "disabled"  # RL system disabled
    SHADOW = "shadow"      # RL runs in parallel, no impact
    GRADUAL = "gradual"    # Gradual rollout
    FULL = "full"          # Full RL integration
    TESTING = "testing"    # A/B testing mode

class SystemState(Enum):
    """System states"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    TRAINING = "training"
    EVALUATING = "evaluating"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class IntegrationConfig:
    """Configuration for RL integration"""
    # Integration settings
    integration_mode: IntegrationMode = IntegrationMode.SHADOW
    enable_training: bool = True
    enable_evaluation: bool = True
    enable_ab_testing: bool = True
    
    # Rollout settings
    gradual_rollout_percentage: float = 10.0  # Start with 10%
    rollout_increment: float = 10.0  # Increase by 10% each step
    rollout_interval_hours: int = 24  # Increase every 24 hours
    
    # Performance thresholds
    min_performance_threshold: float = 0.8  # Minimum performance to continue rollout
    rollback_threshold: float = 0.6  # Performance threshold to trigger rollback
    
    # Training settings
    training_interval_hours: int = 6  # Train every 6 hours
    min_training_samples: int = 100
    max_training_time_minutes: int = 60
    
    # Evaluation settings
    evaluation_interval_hours: int = 12  # Evaluate every 12 hours
    evaluation_duration_minutes: int = 30
    
    # Safety settings
    enable_safety_checks: bool = True
    max_error_rate: float = 0.05  # 5% max error rate
    circuit_breaker_threshold: int = 10  # Number of consecutive errors
    
    # Monitoring settings
    metrics_collection_enabled: bool = True
    detailed_logging: bool = True
    performance_alerts: bool = True
    
    # Storage settings
    data_storage_path: str = "rl_integration_data"
    backup_enabled: bool = True
    backup_interval_hours: int = 24

@dataclass
class IntegrationMetrics:
    """Integration performance metrics"""
    # System metrics
    total_requests: int = 0
    rl_requests: int = 0
    traditional_requests: int = 0
    
    # Performance metrics
    rl_success_rate: float = 0.0
    traditional_success_rate: float = 0.0
    rl_avg_response_time: float = 0.0
    traditional_avg_response_time: float = 0.0
    
    # Quality metrics
    rl_avg_quality: float = 0.0
    traditional_avg_quality: float = 0.0
    user_satisfaction_rl: float = 0.0
    user_satisfaction_traditional: float = 0.0
    
    # Error metrics
    rl_error_count: int = 0
    traditional_error_count: int = 0
    
    # Training metrics
    training_sessions: int = 0
    total_training_time: float = 0.0
    last_training_time: Optional[float] = None
    
    # Rollout metrics
    current_rollout_percentage: float = 0.0
    rollout_start_time: Optional[float] = None
    last_rollout_update: Optional[float] = None

class RLIntegrationManager:
    """Main integration manager for RL system"""
    
    def __init__(self, config: IntegrationConfig):
        """
        Initialize RL integration manager.
        
        Parameters:
        ----------
        config : IntegrationConfig
            Integration configuration
        """
        self.config = config
        self.state = SystemState.INITIALIZING
        self.metrics = IntegrationMetrics()
        
        # Core components
        self.reward_calculator = None
        self.reward_shaper = None
        self.experience_buffer = None
        self.training_manager = None
        self.metrics_collector = None
        self.ab_test_manager = None
        self.performance_comparator = None
        
        # Agent management
        self.rl_agents = {}
        self.traditional_agents = {}
        self.rl_constellation_manager = None
        
        # Integration state
        self.integration_lock = threading.Lock()
        self.circuit_breaker_count = 0
        self.last_error_time = 0
        
        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.background_tasks = []
        self.running = False
        
        # Create storage directory
        os.makedirs(self.config.data_storage_path, exist_ok=True)
        
        logger.info("Initialized RL integration manager")
    
    async def initialize(self) -> None:
        """Initialize all RL components"""
        try:
            self.state = SystemState.INITIALIZING
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize agent management
            await self._initialize_agent_management()
            
            # Initialize monitoring
            await self._initialize_monitoring()
            
            # Load previous state if available
            await self._load_state()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.state = SystemState.READY
            logger.info("RL integration manager initialized successfully")
            
        except Exception as e:
            self.state = SystemState.ERROR
            logger.error(f"Failed to initialize RL integration manager: {e}")
            raise
    
    async def _initialize_core_components(self) -> None:
        """Initialize core RL components"""
        # Initialize reward system
        self.reward_calculator = RewardCalculator()
        self.reward_shaper = RewardShaper()
        
        # Initialize experience buffer
        self.experience_buffer = ExperienceBuffer(max_size=10000)
        
        # Initialize training manager
        if self.config.enable_training:
            from ..training.training_manager import TrainingConfig
            training_config = TrainingConfig(
                algorithm="maddpg",
                training_enabled=True,
                max_training_time=self.config.max_training_time_minutes * 60
            )
            self.training_manager = TrainingManager(training_config)
        
        logger.debug("Core RL components initialized")
    
    async def _initialize_agent_management(self) -> None:
        """Initialize agent management components"""
        # Initialize RL constellation manager
        self.rl_constellation_manager = RLConstellationManager()
        
        logger.debug("Agent management components initialized")
    
    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring components"""
        if self.config.metrics_collection_enabled:
            from ..testing.metrics_collector import MetricsConfig
            metrics_config = MetricsConfig(
                collection_enabled=True,
                storage_path=os.path.join(self.config.data_storage_path, "metrics.db")
            )
            self.metrics_collector = MetricsCollector(metrics_config)
            self.metrics_collector.start_background_processing()
        
        if self.config.enable_ab_testing:
            from ..testing.ab_testing import ABTestConfig
            ab_config = ABTestConfig(
                test_duration_days=7,
                min_sample_size=100,
                significance_level=0.05
            )
            self.ab_test_manager = ABTestManager(ab_config)
        
        if self.config.enable_evaluation:
            from ..testing.performance_comparison import ComparisonConfig
            comparison_config = ComparisonConfig(
                significance_level=0.05,
                minimum_sample_size=30
            )
            self.performance_comparator = PerformanceComparator(comparison_config)
        
        logger.debug("Monitoring components initialized")
    
    async def _load_state(self) -> None:
        """Load previous integration state"""
        state_file = os.path.join(self.config.data_storage_path, "integration_state.json")
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Restore metrics
                if 'metrics' in state_data:
                    metrics_data = state_data['metrics']
                    for key, value in metrics_data.items():
                        if hasattr(self.metrics, key):
                            setattr(self.metrics, key, value)
                
                # Restore rollout state
                if 'rollout_percentage' in state_data:
                    self.metrics.current_rollout_percentage = state_data['rollout_percentage']
                
                logger.info(f"Loaded previous integration state from {state_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load previous state: {e}")
    
    async def _save_state(self) -> None:
        """Save current integration state"""
        state_file = os.path.join(self.config.data_storage_path, "integration_state.json")
        
        try:
            state_data = {
                'timestamp': time.time(),
                'integration_mode': self.config.integration_mode.value,
                'rollout_percentage': self.metrics.current_rollout_percentage,
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'rl_requests': self.metrics.rl_requests,
                    'traditional_requests': self.metrics.traditional_requests,
                    'rl_success_rate': self.metrics.rl_success_rate,
                    'traditional_success_rate': self.metrics.traditional_success_rate,
                    'training_sessions': self.metrics.training_sessions,
                    'current_rollout_percentage': self.metrics.current_rollout_percentage
                }
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.debug(f"Saved integration state to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save integration state: {e}")
    
    async def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        self.running = True
        
        # Training task
        if self.config.enable_training:
            task = asyncio.create_task(self._training_task())
            self.background_tasks.append(task)
        
        # Evaluation task
        if self.config.enable_evaluation:
            task = asyncio.create_task(self._evaluation_task())
            self.background_tasks.append(task)
        
        # Rollout management task
        if self.config.integration_mode == IntegrationMode.GRADUAL:
            task = asyncio.create_task(self._rollout_management_task())
            self.background_tasks.append(task)
        
        # Monitoring task
        task = asyncio.create_task(self._monitoring_task())
        self.background_tasks.append(task)
        
        # State saving task
        task = asyncio.create_task(self._state_saving_task())
        self.background_tasks.append(task)
        
        logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def _training_task(self) -> None:
        """Background training task"""
        while self.running:
            try:
                await asyncio.sleep(self.config.training_interval_hours * 3600)
                
                if self.state == SystemState.RUNNING:
                    await self._run_training_session()
                
            except Exception as e:
                logger.error(f"Error in training task: {e}")
    
    async def _evaluation_task(self) -> None:
        """Background evaluation task"""
        while self.running:
            try:
                await asyncio.sleep(self.config.evaluation_interval_hours * 3600)
                
                if self.state == SystemState.RUNNING:
                    await self._run_evaluation_session()
                
            except Exception as e:
                logger.error(f"Error in evaluation task: {e}")
    
    async def _rollout_management_task(self) -> None:
        """Background rollout management task"""
        while self.running:
            try:
                await asyncio.sleep(self.config.rollout_interval_hours * 3600)
                
                if self.state == SystemState.RUNNING:
                    await self._update_rollout_percentage()
                
            except Exception as e:
                logger.error(f"Error in rollout management task: {e}")
    
    async def _monitoring_task(self) -> None:
        """Background monitoring task"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                await self._check_system_health()
                await self._update_metrics()
                
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
    
    async def _state_saving_task(self) -> None:
        """Background state saving task"""
        while self.running:
            try:
                await asyncio.sleep(1800)  # Save every 30 minutes
                await self._save_state()
                
            except Exception as e:
                logger.error(f"Error in state saving task: {e}")
    
    async def should_use_rl(self, user_id: Optional[str] = None, 
                           session_id: Optional[str] = None,
                           task_type: Optional[str] = None) -> bool:
        """Determine if RL should be used for this request"""
        if self.config.integration_mode == IntegrationMode.DISABLED:
            return False
        
        if self.config.integration_mode == IntegrationMode.FULL:
            return True
        
        if self.config.integration_mode == IntegrationMode.SHADOW:
            # In shadow mode, always use traditional but run RL in parallel
            return False
        
        if self.config.integration_mode == IntegrationMode.TESTING:
            # Use A/B testing to determine
            if self.ab_test_manager:
                group = self.ab_test_manager.assign_to_group(user_id or session_id or str(uuid.uuid4()))
                return group.value == "treatment"
            return False
        
        if self.config.integration_mode == IntegrationMode.GRADUAL:
            # Use rollout percentage
            import random
            return random.random() < (self.metrics.current_rollout_percentage / 100.0)
        
        return False
    
    async def process_request(self, request_data: Dict[str, Any],
                            user_id: Optional[str] = None,
                            session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a request through the integrated system"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Update request count
            self.metrics.total_requests += 1
            
            # Determine which system to use
            use_rl = await self.should_use_rl(user_id, session_id, request_data.get('task_type'))
            
            # Process request
            if use_rl:
                result = await self._process_rl_request(request_data, user_id, session_id, request_id)
                self.metrics.rl_requests += 1
            else:
                result = await self._process_traditional_request(request_data, user_id, session_id, request_id)
                self.metrics.traditional_requests += 1
                
                # In shadow mode, also run RL in parallel for comparison
                if self.config.integration_mode == IntegrationMode.SHADOW:
                    asyncio.create_task(self._process_rl_request_shadow(
                        request_data, user_id, session_id, request_id
                    ))
            
            # Record metrics
            response_time = time.time() - start_time
            await self._record_request_metrics(use_rl, result, response_time, user_id, session_id)
            
            return result
            
        except Exception as e:
            # Handle errors
            await self._handle_request_error(e, use_rl, user_id, session_id)
            
            # Fallback to traditional system if RL fails
            if use_rl:
                logger.warning(f"RL request failed, falling back to traditional: {e}")
                try:
                    result = await self._process_traditional_request(request_data, user_id, session_id, request_id)
                    self.metrics.traditional_requests += 1
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise
            else:
                raise
    
    async def _process_rl_request(self, request_data: Dict[str, Any],
                                 user_id: Optional[str],
                                 session_id: Optional[str],
                                 request_id: str) -> Dict[str, Any]:
        """Process request using RL system"""
        # This would integrate with the actual RL agents
        # For now, return a placeholder response
        
        # Get or create RL agent for this session
        agent_key = session_id or user_id or "default"
        if agent_key not in self.rl_agents:
            # Create new RL agent (this would integrate with actual GAAPF agents)
            self.rl_agents[agent_key] = {
                'agent_id': str(uuid.uuid4()),
                'created_at': time.time(),
                'request_count': 0
            }
        
        agent_info = self.rl_agents[agent_key]
        agent_info['request_count'] += 1
        
        # Simulate RL processing
        await asyncio.sleep(0.1)  # Simulate processing time
        
        result = {
            'response': f"RL-enhanced response for request {request_id}",
            'agent_id': agent_info['agent_id'],
            'system_type': 'rl_enhanced',
            'confidence': 0.85,
            'processing_time': 0.1,
            'metadata': {
                'rl_features_used': ['reward_shaping', 'experience_replay'],
                'exploration_rate': 0.1
            }
        }
        
        return result
    
    async def _process_traditional_request(self, request_data: Dict[str, Any],
                                         user_id: Optional[str],
                                         session_id: Optional[str],
                                         request_id: str) -> Dict[str, Any]:
        """Process request using traditional system"""
        # This would integrate with the existing GAAPF system
        # For now, return a placeholder response
        
        # Simulate traditional processing
        await asyncio.sleep(0.15)  # Simulate slightly longer processing time
        
        result = {
            'response': f"Traditional response for request {request_id}",
            'system_type': 'traditional',
            'confidence': 0.80,
            'processing_time': 0.15,
            'metadata': {
                'traditional_features_used': ['rule_based', 'template_matching']
            }
        }
        
        return result
    
    async def _process_rl_request_shadow(self, request_data: Dict[str, Any],
                                       user_id: Optional[str],
                                       session_id: Optional[str],
                                       request_id: str) -> None:
        """Process request using RL in shadow mode (for comparison)"""
        try:
            rl_result = await self._process_rl_request(request_data, user_id, session_id, f"{request_id}_shadow")
            
            # Record shadow metrics for comparison
            if self.performance_comparator:
                from ..testing.performance_comparison import SystemType
                self.performance_comparator.record_performance(
                    system_type=SystemType.RL_ENHANCED,
                    metric_name="response_time",
                    value=rl_result.get('processing_time', 0),
                    session_id=session_id,
                    user_id=user_id
                )
                
                self.performance_comparator.record_performance(
                    system_type=SystemType.RL_ENHANCED,
                    metric_name="response_quality",
                    value=rl_result.get('confidence', 0),
                    session_id=session_id,
                    user_id=user_id
                )
        
        except Exception as e:
            logger.warning(f"Shadow RL processing failed: {e}")
    
    async def _record_request_metrics(self, used_rl: bool, result: Dict[str, Any],
                                    response_time: float, user_id: Optional[str],
                                    session_id: Optional[str]) -> None:
        """Record metrics for a processed request"""
        if self.metrics_collector:
            # Record basic metrics
            self.metrics_collector.record_metric(
                "response_time",
                response_time,
                tags={'system_type': 'rl' if used_rl else 'traditional'},
                session_id=session_id,
                user_id=user_id
            )
            
            self.metrics_collector.record_metric(
                "response_quality",
                result.get('confidence', 0),
                tags={'system_type': 'rl' if used_rl else 'traditional'},
                session_id=session_id,
                user_id=user_id
            )
        
        # Record in performance comparator
        if self.performance_comparator:
            from ..testing.performance_comparison import SystemType
            system_type = SystemType.RL_ENHANCED if used_rl else SystemType.TRADITIONAL
            
            self.performance_comparator.record_performance(
                system_type=system_type,
                metric_name="response_time",
                value=response_time,
                session_id=session_id,
                user_id=user_id
            )
            
            self.performance_comparator.record_performance(
                system_type=system_type,
                metric_name="response_quality",
                value=result.get('confidence', 0),
                session_id=session_id,
                user_id=user_id
            )
    
    async def _handle_request_error(self, error: Exception, used_rl: bool,
                                  user_id: Optional[str], session_id: Optional[str]) -> None:
        """Handle request processing errors"""
        if used_rl:
            self.metrics.rl_error_count += 1
        else:
            self.metrics.traditional_error_count += 1
        
        # Update circuit breaker
        self.circuit_breaker_count += 1
        self.last_error_time = time.time()
        
        # Record error metrics
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                "error_rate",
                1.0,
                tags={'system_type': 'rl' if used_rl else 'traditional', 'error_type': type(error).__name__},
                session_id=session_id,
                user_id=user_id
            )
        
        # Check if circuit breaker should trigger
        if (self.config.enable_safety_checks and 
            self.circuit_breaker_count >= self.config.circuit_breaker_threshold):
            await self._trigger_circuit_breaker()
    
    async def _trigger_circuit_breaker(self) -> None:
        """Trigger circuit breaker to disable RL temporarily"""
        logger.critical("Circuit breaker triggered - disabling RL system temporarily")
        
        # Temporarily disable RL
        original_mode = self.config.integration_mode
        self.config.integration_mode = IntegrationMode.DISABLED
        
        # Reset circuit breaker after some time
        async def reset_circuit_breaker():
            await asyncio.sleep(300)  # Wait 5 minutes
            self.config.integration_mode = original_mode
            self.circuit_breaker_count = 0
            logger.info("Circuit breaker reset - RL system re-enabled")
        
        asyncio.create_task(reset_circuit_breaker())
    
    async def _run_training_session(self) -> None:
        """Run a training session"""
        if not self.training_manager:
            return
        
        try:
            self.state = SystemState.TRAINING
            start_time = time.time()
            
            logger.info("Starting RL training session")
            
            # Check if we have enough samples
            if len(self.experience_buffer.experiences) < self.config.min_training_samples:
                logger.info(f"Insufficient training samples: {len(self.experience_buffer.experiences)}")
                return
            
            # Run training
            training_result = await self.training_manager.train_step()
            
            # Update metrics
            training_time = time.time() - start_time
            self.metrics.training_sessions += 1
            self.metrics.total_training_time += training_time
            self.metrics.last_training_time = time.time()
            
            logger.info(f"Training session completed in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Training session failed: {e}")
        
        finally:
            self.state = SystemState.RUNNING
    
    async def _run_evaluation_session(self) -> None:
        """Run an evaluation session"""
        if not self.performance_comparator:
            return
        
        try:
            self.state = SystemState.EVALUATING
            
            logger.info("Starting evaluation session")
            
            # Compare RL vs traditional performance
            from ..testing.performance_comparison import SystemType
            
            comparison_results = self.performance_comparator.compare_all_metrics(
                SystemType.RL_ENHANCED,
                SystemType.TRADITIONAL,
                time_range=(time.time() - 24*3600, time.time())  # Last 24 hours
            )
            
            # Check if performance is acceptable
            overall_performance = self._evaluate_overall_performance(comparison_results)
            
            if overall_performance < self.config.rollback_threshold:
                logger.warning(f"Performance below rollback threshold: {overall_performance}")
                await self._trigger_rollback()
            
            elif overall_performance < self.config.min_performance_threshold:
                logger.warning(f"Performance below minimum threshold: {overall_performance}")
                # Pause rollout but don't rollback
                if self.config.integration_mode == IntegrationMode.GRADUAL:
                    logger.info("Pausing gradual rollout due to performance concerns")
            
            logger.info(f"Evaluation completed - overall performance: {overall_performance}")
            
        except Exception as e:
            logger.error(f"Evaluation session failed: {e}")
        
        finally:
            self.state = SystemState.RUNNING
    
    def _evaluate_overall_performance(self, comparison_results: Dict[str, Any]) -> float:
        """Evaluate overall performance from comparison results"""
        if not comparison_results:
            return 0.5  # Neutral score if no data
        
        # Calculate weighted performance score
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, result in comparison_results.items():
            if metric_name in self.performance_comparator.comparison_metrics:
                metric_def = self.performance_comparator.comparison_metrics[metric_name]
                
                # Calculate performance score for this metric
                if result.winner and result.winner.value == "rl_enhanced":
                    score = 0.6 + min(0.4, abs(result.effect_size) * 0.2)  # 0.6-1.0 range
                elif result.winner and result.winner.value == "traditional":
                    score = 0.4 - min(0.4, abs(result.effect_size) * 0.2)  # 0.0-0.4 range
                else:
                    score = 0.5  # Neutral if no significant difference
                
                weight = metric_def.weight
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    async def _trigger_rollback(self) -> None:
        """Trigger rollback to traditional system"""
        logger.critical("Triggering rollback to traditional system")
        
        # Set to shadow mode to collect data while using traditional system
        self.config.integration_mode = IntegrationMode.SHADOW
        self.metrics.current_rollout_percentage = 0.0
        
        # Record rollback event
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                "system_rollback",
                1.0,
                tags={'reason': 'performance_threshold'}
            )
    
    async def _update_rollout_percentage(self) -> None:
        """Update gradual rollout percentage"""
        if self.config.integration_mode != IntegrationMode.GRADUAL:
            return
        
        # Check recent performance
        current_time = time.time()
        recent_performance = await self._get_recent_performance()
        
        if recent_performance >= self.config.min_performance_threshold:
            # Increase rollout percentage
            new_percentage = min(
                100.0,
                self.metrics.current_rollout_percentage + self.config.rollout_increment
            )
            
            self.metrics.current_rollout_percentage = new_percentage
            self.metrics.last_rollout_update = current_time
            
            logger.info(f"Increased rollout percentage to {new_percentage}%")
            
            # Switch to full mode if we reach 100%
            if new_percentage >= 100.0:
                self.config.integration_mode = IntegrationMode.FULL
                logger.info("Rollout complete - switched to full RL mode")
        
        else:
            logger.warning(f"Rollout paused due to performance: {recent_performance}")
    
    async def _get_recent_performance(self) -> float:
        """Get recent performance score"""
        if not self.performance_comparator:
            return 0.5
        
        try:
            # Get performance for last few hours
            recent_time = time.time() - 6 * 3600  # Last 6 hours
            
            from ..testing.performance_comparison import SystemType
            comparison_results = self.performance_comparator.compare_all_metrics(
                SystemType.RL_ENHANCED,
                SystemType.TRADITIONAL,
                time_range=(recent_time, time.time())
            )
            
            return self._evaluate_overall_performance(comparison_results)
        
        except Exception as e:
            logger.error(f"Error getting recent performance: {e}")
            return 0.5
    
    async def _check_system_health(self) -> None:
        """Check overall system health"""
        if not self.config.enable_safety_checks:
            return
        
        # Check error rates
        total_requests = self.metrics.total_requests
        if total_requests > 0:
            rl_error_rate = self.metrics.rl_error_count / max(1, self.metrics.rl_requests)
            traditional_error_rate = self.metrics.traditional_error_count / max(1, self.metrics.traditional_requests)
            
            if rl_error_rate > self.config.max_error_rate:
                logger.warning(f"RL error rate too high: {rl_error_rate:.3f}")
                
                # Consider disabling RL temporarily
                if rl_error_rate > self.config.max_error_rate * 2:
                    await self._trigger_circuit_breaker()
        
        # Reset circuit breaker if enough time has passed without errors
        if (self.circuit_breaker_count > 0 and 
            time.time() - self.last_error_time > 1800):  # 30 minutes
            self.circuit_breaker_count = max(0, self.circuit_breaker_count - 1)
    
    async def _update_metrics(self) -> None:
        """Update internal metrics"""
        # Calculate success rates
        if self.metrics.rl_requests > 0:
            self.metrics.rl_success_rate = 1.0 - (self.metrics.rl_error_count / self.metrics.rl_requests)
        
        if self.metrics.traditional_requests > 0:
            self.metrics.traditional_success_rate = 1.0 - (self.metrics.traditional_error_count / self.metrics.traditional_requests)
        
        # Record system-level metrics
        if self.metrics_collector:
            self.metrics_collector.record_multiple_metrics({
                'total_requests': self.metrics.total_requests,
                'rl_requests': self.metrics.rl_requests,
                'traditional_requests': self.metrics.traditional_requests,
                'rl_success_rate': self.metrics.rl_success_rate * 100,
                'traditional_success_rate': self.metrics.traditional_success_rate * 100,
                'current_rollout_percentage': self.metrics.current_rollout_percentage
            })
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'state': self.state.value,
            'integration_mode': self.config.integration_mode.value,
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'rl_requests': self.metrics.rl_requests,
                'traditional_requests': self.metrics.traditional_requests,
                'rl_success_rate': self.metrics.rl_success_rate,
                'traditional_success_rate': self.metrics.traditional_success_rate,
                'current_rollout_percentage': self.metrics.current_rollout_percentage,
                'training_sessions': self.metrics.training_sessions,
                'circuit_breaker_count': self.circuit_breaker_count
            },
            'components': {
                'reward_calculator': self.reward_calculator is not None,
                'experience_buffer': self.experience_buffer is not None,
                'training_manager': self.training_manager is not None,
                'metrics_collector': self.metrics_collector is not None,
                'ab_test_manager': self.ab_test_manager is not None,
                'performance_comparator': self.performance_comparator is not None
            },
            'background_tasks': len(self.background_tasks),
            'running': self.running
        }
    
    async def shutdown(self) -> None:
        """Shutdown the integration manager"""
        logger.info("Shutting down RL integration manager")
        
        self.state = SystemState.SHUTDOWN
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save final state
        await self._save_state()
        
        # Cleanup components
        if self.metrics_collector:
            self.metrics_collector.cleanup()
        
        if self.training_manager:
            await self.training_manager.cleanup()
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("RL integration manager shutdown complete")