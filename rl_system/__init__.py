"""GAAPF Reinforcement Learning System

This module provides a comprehensive reinforcement learning enhancement
for the GAAPF (Generative AI Agent Programming Framework) system.

The RL system includes:
- Multi-agent reinforcement learning algorithms (MADDPG, DQN, Policy Gradient)
- RL-enhanced constellation management for optimal agent team formation
- Comprehensive training and evaluation infrastructure
- A/B testing and performance comparison capabilities
- Integration with the core GAAPF framework
- Monitoring and observability features

Main Components:
- algorithms: Core RL algorithms and agents
- managers: RL-enhanced constellation managers
- training: Training infrastructure and curriculum learning
- testing: A/B testing and performance evaluation
- integration: GAAPF framework integration
- utils: Utility functions and helpers

Usage:
    from rl_system import RLSystem, RLConfig
    
    # Initialize RL system
    config = RLConfig()
    rl_system = RLSystem(config)
    
    # Start RL system
    await rl_system.start()
    
    # Use RL-enhanced constellation manager
    constellation_manager = rl_system.get_constellation_manager()
    
    # Stop RL system
    await rl_system.stop()
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import time
from pathlib import Path

# Core RL algorithms
from .algorithms import (
    MADDPG, MADDPGConfig, MADDPGAgent,
    DQNAgent, DQNConfig,
    PolicyGradientAgent, PolicyGradientConfig
)

# RL-enhanced managers
from .managers import RLConstellationManager

# Training infrastructure
from .training import (
    TrainingManager, TrainingConfig,
    CurriculumManager, CurriculumConfig,
    EvaluationManager, EvaluationConfig
)

# Testing and evaluation
from .testing import (
    ABTestManager, ABTestConfig,
    MetricsCollector, MetricsConfig,
    PerformanceComparator, ComparisonConfig
)

# Integration components
from .integration import (
    RLIntegrationManager, IntegrationConfig,
    GAAPFAdapter, AdapterConfig,
    ConfigurationManager,
    MonitoringIntegration, MonitoringConfig
)

# Utilities
from .utils import (
    RewardCalculator, RewardShaper, RewardType, RewardEvent,
    ExperienceBuffer, Experience, Episode,
    StateExtractor, StateConfig
)

logger = logging.getLogger(__name__)

class RLSystemMode(Enum):
    """RL system operation modes"""
    DISABLED = "disabled"
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    TESTING = "testing"
    FULL = "full"

class RLAlgorithm(Enum):
    """Available RL algorithms"""
    MADDPG = "maddpg"
    DQN = "dqn"
    POLICY_GRADIENT = "policy_gradient"
    AUTO = "auto"  # Automatically select best algorithm

@dataclass
class RLConfig:
    """Main configuration for RL system"""
    # General settings
    enabled: bool = True
    mode: RLSystemMode = RLSystemMode.FULL
    algorithm: RLAlgorithm = RLAlgorithm.AUTO
    
    # System paths
    base_path: str = "rl_system_data"
    models_path: str = "models"
    logs_path: str = "logs"
    data_path: str = "data"
    
    # Algorithm configurations
    maddpg_config: MADDPGConfig = field(default_factory=MADDPGConfig)
    dqn_config: DQNConfig = field(default_factory=DQNConfig)
    policy_gradient_config: PolicyGradientConfig = field(default_factory=PolicyGradientConfig)
    
    # Training configurations
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum_config: CurriculumConfig = field(default_factory=CurriculumConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Testing configurations
    ab_test_config: ABTestConfig = field(default_factory=lambda: ABTestConfig(
        test_name="default_test",
        test_description="Default RL system test",
        control_group_size=100,
        treatment_group_size=100
    ))
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)
    comparison_config: ComparisonConfig = field(default_factory=ComparisonConfig)
    
    # Integration configurations
    integration_config: IntegrationConfig = field(default_factory=IntegrationConfig)
    adapter_config: AdapterConfig = field(default_factory=AdapterConfig)
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Utility configurations
    reward_config: Dict[str, Any] = field(default_factory=dict)
    buffer_config: Dict[str, Any] = field(default_factory=dict)
    state_config: StateConfig = field(default_factory=StateConfig)
    constellation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    max_concurrent_training: int = 2
    max_concurrent_evaluation: int = 4
    background_task_interval: int = 60  # seconds
    
    # Logging settings
    log_level: str = "INFO"
    enable_detailed_logging: bool = False
    log_to_file: bool = True

class RLSystem:
    """Main RL system class that orchestrates all components"""
    
    def __init__(self, config: RLConfig):
        """
        Initialize the RL system.
        
        Parameters:
        ----------
        config : RLConfig
            RL system configuration
        """
        self.config = config
        self.system_id = f"rl_system_{int(time.time())}"
        self.started_at = None
        self.running = False
        
        # Setup logging
        self._setup_logging()
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self._initialize_components()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info(f"Initialized RL system: {self.system_id}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Configure logger
        logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file:
            log_file = Path(self.config.base_path) / self.config.logs_path / "rl_system.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        logger.debug("Logging configured")
    
    def _create_directories(self) -> None:
        """Create necessary directories"""
        base_path = Path(self.config.base_path)
        
        directories = [
            base_path,
            base_path / self.config.models_path,
            base_path / self.config.logs_path,
            base_path / self.config.data_path,
            base_path / "checkpoints",
            base_path / "exports",
            base_path / "evaluations",
            base_path / "tests"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Created directories under {base_path}")
    
    def _initialize_components(self) -> None:
        """Initialize all RL system components"""
        try:
            # Initialize utility components
            self.reward_calculator = RewardCalculator(self.config.reward_config)
            self.reward_shaper = RewardShaper(self.config.reward_config)
            self.experience_buffer = ExperienceBuffer(self.config.buffer_config)
            self.state_extractor = StateExtractor(self.config.state_config)
            
            # Initialize RL-enhanced managers first
            self.constellation_manager = RLConstellationManager(
                agents={},
                config=self.config.constellation_config,
                reward_calculator=self.reward_calculator,
                experience_buffer=self.experience_buffer,
                is_logging=self.config.enable_detailed_logging
            )
            
            # Initialize agents dictionary (empty for now)
            self.agents = {}
            
            # Initialize training components
            self.training_manager = TrainingManager(
                self.config.training_config,
                self.reward_calculator,
                self.experience_buffer,
                self.constellation_manager,
                self.agents
            )
            self.curriculum_manager = CurriculumManager(self.config.curriculum_config)
            self.evaluation_manager = EvaluationManager(self.config.evaluation_config)
            
            # Initialize testing components
            self.ab_test_manager = ABTestManager(self.config.ab_test_config)
            self.metrics_collector = MetricsCollector(self.config.metrics_config)
            self.performance_comparator = PerformanceComparator(self.config.comparison_config)
            
            # Initialize integration components
            self.integration_manager = RLIntegrationManager(self.config.integration_config)
            self.gaapf_adapter = GAAPFAdapter(self.config.adapter_config)
            self.configuration_manager = ConfigurationManager()
            self.monitoring_integration = MonitoringIntegration(self.config.monitoring_config)
            
            # Initialize algorithms based on configuration
            self._initialize_algorithms()
            
            logger.info("All RL system components initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize RL system components: {e}")
            raise
    
    def _initialize_algorithms(self) -> None:
        """Initialize RL algorithms based on configuration"""
        self.algorithms = {}
        
        if self.config.algorithm in [RLAlgorithm.MADDPG, RLAlgorithm.AUTO]:
            try:
                self.algorithms['maddpg'] = MADDPG(self.config.maddpg_config)
                logger.debug("Initialized MADDPG algorithm")
            except Exception as e:
                logger.warning(f"Failed to initialize MADDPG: {e}")
        
        if self.config.algorithm in [RLAlgorithm.DQN, RLAlgorithm.AUTO]:
            try:
                self.algorithms['dqn'] = DQNAgent(self.config.dqn_config)
                logger.debug("Initialized DQN algorithm")
            except Exception as e:
                logger.warning(f"Failed to initialize DQN: {e}")
        
        if self.config.algorithm in [RLAlgorithm.POLICY_GRADIENT, RLAlgorithm.AUTO]:
            try:
                self.algorithms['policy_gradient'] = PolicyGradientAgent(self.config.policy_gradient_config)
                logger.debug("Initialized Policy Gradient algorithm")
            except Exception as e:
                logger.warning(f"Failed to initialize Policy Gradient: {e}")
        
        if not self.algorithms:
            logger.warning("No RL algorithms successfully initialized")
        else:
            logger.info(f"Initialized {len(self.algorithms)} RL algorithms")
    
    async def start(self) -> None:
        """Start the RL system"""
        if self.running:
            logger.warning("RL system is already running")
            return
        
        if not self.config.enabled:
            logger.info("RL system is disabled")
            return
        
        logger.info(f"Starting RL system in {self.config.mode.value} mode")
        
        try:
            self.started_at = time.time()
            self.running = True
            
            # Start monitoring
            await self.monitoring_integration.start_monitoring()
            
            # Start metrics collection
            await self.metrics_collector.start()
            
            # Start integration manager
            await self.integration_manager.start()
            
            # Start training if in training mode
            if self.config.mode in [RLSystemMode.TRAINING, RLSystemMode.FULL]:
                await self.training_manager.start_training()
            
            # Start evaluation if in evaluation mode
            if self.config.mode in [RLSystemMode.EVALUATION, RLSystemMode.FULL]:
                await self.evaluation_manager.start_evaluation()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info(f"RL system started successfully: {self.system_id}")
        
        except Exception as e:
            logger.error(f"Failed to start RL system: {e}")
            self.running = False
            raise
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks"""
        # System health monitoring
        task = asyncio.create_task(self._health_monitoring_task())
        self.background_tasks.append(task)
        
        # Performance monitoring
        task = asyncio.create_task(self._performance_monitoring_task())
        self.background_tasks.append(task)
        
        # Periodic cleanup
        task = asyncio.create_task(self._cleanup_task())
        self.background_tasks.append(task)
        
        logger.debug(f"Started {len(self.background_tasks)} background tasks")
    
    async def _health_monitoring_task(self) -> None:
        """Background task for health monitoring"""
        while self.running:
            try:
                # Check system health
                health_status = self.monitoring_integration.get_system_health()
                
                # Record health metrics
                self.metrics_collector.record_metric(
                    "rl_system.health.overall",
                    1.0 if health_status['overall_healthy'] else 0.0
                )
                
                self.metrics_collector.record_metric(
                    "rl_system.health.active_alerts",
                    health_status['active_alerts']
                )
                
                await asyncio.sleep(self.config.background_task_interval)
            
            except Exception as e:
                logger.error(f"Health monitoring task error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring_task(self) -> None:
        """Background task for performance monitoring"""
        while self.running:
            try:
                # Monitor system performance
                uptime = time.time() - self.started_at if self.started_at else 0
                
                self.metrics_collector.record_metric(
                    "rl_system.uptime",
                    uptime
                )
                
                # Monitor algorithm performance
                for name, algorithm in self.algorithms.items():
                    if hasattr(algorithm, 'get_metrics'):
                        metrics = algorithm.get_metrics()
                        for metric_name, value in metrics.items():
                            self.metrics_collector.record_metric(
                                f"rl_algorithm.{name}.{metric_name}",
                                value
                            )
                
                await asyncio.sleep(self.config.background_task_interval)
            
            except Exception as e:
                logger.error(f"Performance monitoring task error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_task(self) -> None:
        """Background task for periodic cleanup"""
        while self.running:
            try:
                # Cleanup old data
                await self.experience_buffer.cleanup_old_experiences()
                
                # Cleanup metrics
                await self.metrics_collector.cleanup_old_data()
                
                # Cleanup logs
                await self._cleanup_old_logs()
                
                await asyncio.sleep(3600)  # Run every hour
            
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_logs(self) -> None:
        """Cleanup old log files"""
        try:
            logs_path = Path(self.config.base_path) / self.config.logs_path
            current_time = time.time()
            
            for log_file in logs_path.glob("*.log*"):
                file_age = current_time - log_file.stat().st_mtime
                if file_age > (7 * 24 * 3600):  # Keep 7 days
                    log_file.unlink()
                    logger.debug(f"Deleted old log file: {log_file}")
        
        except Exception as e:
            logger.warning(f"Error cleaning up log files: {e}")
    
    def get_constellation_manager(self) -> RLConstellationManager:
        """Get the RL-enhanced constellation manager"""
        return self.constellation_manager
    
    def get_algorithm(self, name: str) -> Optional[Any]:
        """Get a specific RL algorithm"""
        return self.algorithms.get(name)
    
    def get_training_manager(self) -> TrainingManager:
        """Get the training manager"""
        return self.training_manager
    
    def get_evaluation_manager(self) -> EvaluationManager:
        """Get the evaluation manager"""
        return self.evaluation_manager
    
    def get_metrics_collector(self) -> MetricsCollector:
        """Get the metrics collector"""
        return self.metrics_collector
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_id': self.system_id,
            'started_at': self.started_at,
            'uptime': time.time() - self.started_at if self.started_at else 0,
            'running': self.running,
            'mode': self.config.mode.value,
            'enabled': self.config.enabled,
            'algorithms': list(self.algorithms.keys()),
            'components': {
                'training_manager': self.training_manager is not None,
                'evaluation_manager': self.evaluation_manager is not None,
                'constellation_manager': self.constellation_manager is not None,
                'metrics_collector': self.metrics_collector is not None,
                'monitoring_integration': self.monitoring_integration is not None
            },
            'background_tasks': len(self.background_tasks),
            'health': self.monitoring_integration.get_system_health() if self.monitoring_integration else {},
            'performance': self.performance_comparator.get_summary() if self.performance_comparator else {}
        }
    
    async def stop(self) -> None:
        """Stop the RL system"""
        if not self.running:
            logger.warning("RL system is not running")
            return
        
        logger.info("Stopping RL system")
        
        try:
            self.running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Stop components
            if self.training_manager:
                await self.training_manager.stop_training()
            
            if self.evaluation_manager:
                await self.evaluation_manager.stop_evaluation()
            
            if self.integration_manager:
                await self.integration_manager.stop()
            
            if self.metrics_collector:
                await self.metrics_collector.stop()
            
            if self.monitoring_integration:
                await self.monitoring_integration.stop_monitoring()
            
            logger.info("RL system stopped successfully")
        
        except Exception as e:
            logger.error(f"Error stopping RL system: {e}")
            raise
    
    def cleanup(self) -> None:
        """Cleanup RL system resources"""
        logger.info("Cleaning up RL system")
        
        try:
            # Cleanup components
            if self.monitoring_integration:
                self.monitoring_integration.cleanup()
            
            if self.metrics_collector:
                self.metrics_collector.cleanup()
            
            if self.experience_buffer:
                self.experience_buffer.cleanup()
            
            # Clear algorithms
            self.algorithms.clear()
            
            # Clear background tasks
            self.background_tasks.clear()
            
            logger.info("RL system cleanup complete")
        
        except Exception as e:
            logger.error(f"Error during RL system cleanup: {e}")

# Convenience functions for easy usage
def create_rl_system(config: Optional[RLConfig] = None) -> RLSystem:
    """Create an RL system with default or provided configuration"""
    if config is None:
        config = RLConfig()
    
    return RLSystem(config)

async def start_rl_system(config: Optional[RLConfig] = None) -> RLSystem:
    """Create and start an RL system"""
    rl_system = create_rl_system(config)
    await rl_system.start()
    return rl_system

# Version information
__version__ = "1.0.0"
__author__ = "GAAPF RL Team"
__description__ = "Reinforcement Learning enhancement for GAAPF"

# Export main classes and functions
__all__ = [
    # Main classes
    'RLSystem',
    'RLConfig',
    'RLSystemMode',
    'RLAlgorithm',
    
    # Convenience functions
    'create_rl_system',
    'start_rl_system',
    
    # Algorithm classes
    'MADDPG',
    'MADDPGConfig',
    'MADDPGAgent',
    'DQNAgent',
    'DQNConfig',
    'PolicyGradientAgent',
    'PolicyGradientConfig',
    
    # Manager classes
    'RLConstellationManager',
    
    # Training classes
    'TrainingManager',
    'TrainingConfig',
    'CurriculumManager',
    'CurriculumConfig',
    'EvaluationManager',
    'EvaluationConfig',
    
    # Testing classes
    'ABTestManager',
    'ABTestConfig',
    'MetricsCollector',
    'MetricsConfig',
    'PerformanceComparator',
    'ComparisonConfig',
    
    # Integration classes
    'RLIntegrationManager',
    'IntegrationConfig',
    'GAAPFAdapter',
    'AdapterConfig',
    'ConfigurationManager',
    'MonitoringIntegration',
    'MonitoringConfig',
    
    # Utility classes
    'RewardSystem',
    'RewardConfig',
    'ExperienceBuffer',
    'BufferConfig',
    'StateExtractor',
    'StateConfig',
    
    # Version info
    '__version__',
    '__author__',
    '__description__'
]