"""Monitoring Integration for RL System

This module provides comprehensive monitoring and observability capabilities
for the RL system integration with GAAPF.
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class MonitoringScope(Enum):
    """Monitoring scopes"""
    SYSTEM = "system"
    AGENT = "agent"
    CONSTELLATION = "constellation"
    TRAINING = "training"
    INTEGRATION = "integration"
    USER = "user"

@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    metric_type: MetricType
    scope: MonitoringScope
    description: str = ""
    unit: str = ""
    tags: List[str] = field(default_factory=list)
    alert_thresholds: Dict[AlertLevel, float] = field(default_factory=dict)
    retention_days: int = 30
    aggregation_window: int = 300  # seconds

@dataclass
class MetricValue:
    """A metric value with timestamp and metadata"""
    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    description: str
    check_function: Callable[[], bool]
    interval_seconds: int = 60
    timeout_seconds: int = 10
    enabled: bool = True
    last_check: Optional[float] = None
    last_result: Optional[bool] = None
    consecutive_failures: int = 0
    max_failures: int = 3

@dataclass
class MonitoringConfig:
    """Configuration for monitoring integration"""
    # General settings
    enabled: bool = True
    collection_interval: int = 60  # seconds
    retention_days: int = 30
    
    # Storage settings
    storage_path: str = "monitoring_data"
    max_memory_metrics: int = 10000
    flush_interval: int = 300  # seconds
    
    # Alert settings
    enable_alerts: bool = True
    alert_cooldown: int = 300  # seconds
    max_alerts_per_hour: int = 100
    
    # Health check settings
    enable_health_checks: bool = True
    health_check_interval: int = 60  # seconds
    
    # Performance settings
    enable_performance_monitoring: bool = True
    enable_resource_monitoring: bool = True
    enable_custom_metrics: bool = True
    
    # Export settings
    enable_prometheus_export: bool = False
    prometheus_port: int = 8090
    enable_json_export: bool = True
    json_export_interval: int = 300

class MonitoringIntegration:
    """Main monitoring integration for RL system"""
    
    def __init__(self, config: MonitoringConfig):
        """
        Initialize monitoring integration.
        
        Parameters:
        ----------
        config : MonitoringConfig
            Monitoring configuration
        """
        self.config = config
        self.monitoring_id = f"monitor_{int(time.time())}"
        self.started_at = time.time()
        
        # Metric storage
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_memory_metrics))
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Alert management
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_cooldowns: Dict[str, float] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.system_health: Dict[str, Any] = {}
        
        # Background tasks
        self.running = False
        self.background_tasks: List[asyncio.Task] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Storage
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        # Initialize default health checks
        self._initialize_default_health_checks()
        
        logger.info(f"Initialized monitoring integration: {self.monitoring_id}")
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default system metrics"""
        # System metrics
        self.register_metric(
            "system.uptime",
            MetricType.GAUGE,
            MonitoringScope.SYSTEM,
            "System uptime in seconds",
            unit="seconds"
        )
        
        self.register_metric(
            "system.memory_usage",
            MetricType.GAUGE,
            MonitoringScope.SYSTEM,
            "Memory usage percentage",
            unit="percent",
            alert_thresholds={AlertLevel.WARNING: 80.0, AlertLevel.CRITICAL: 95.0}
        )
        
        self.register_metric(
            "system.cpu_usage",
            MetricType.GAUGE,
            MonitoringScope.SYSTEM,
            "CPU usage percentage",
            unit="percent",
            alert_thresholds={AlertLevel.WARNING: 80.0, AlertLevel.CRITICAL: 95.0}
        )
        
        # RL System metrics
        self.register_metric(
            "rl.requests_total",
            MetricType.COUNTER,
            MonitoringScope.INTEGRATION,
            "Total RL requests processed"
        )
        
        self.register_metric(
            "rl.requests_success_rate",
            MetricType.GAUGE,
            MonitoringScope.INTEGRATION,
            "RL request success rate",
            unit="percent",
            alert_thresholds={AlertLevel.WARNING: 90.0, AlertLevel.CRITICAL: 80.0}
        )
        
        self.register_metric(
            "rl.response_time",
            MetricType.HISTOGRAM,
            MonitoringScope.INTEGRATION,
            "RL response time",
            unit="seconds",
            alert_thresholds={AlertLevel.WARNING: 5.0, AlertLevel.CRITICAL: 10.0}
        )
        
        self.register_metric(
            "rl.error_rate",
            MetricType.RATE,
            MonitoringScope.INTEGRATION,
            "RL error rate",
            unit="errors/minute",
            alert_thresholds={AlertLevel.WARNING: 5.0, AlertLevel.CRITICAL: 10.0}
        )
        
        # Agent metrics
        self.register_metric(
            "agent.active_count",
            MetricType.GAUGE,
            MonitoringScope.AGENT,
            "Number of active agents"
        )
        
        self.register_metric(
            "agent.performance_score",
            MetricType.GAUGE,
            MonitoringScope.AGENT,
            "Agent performance score",
            unit="score",
            alert_thresholds={AlertLevel.WARNING: 0.7, AlertLevel.CRITICAL: 0.5}
        )
        
        # Training metrics
        self.register_metric(
            "training.episodes_completed",
            MetricType.COUNTER,
            MonitoringScope.TRAINING,
            "Training episodes completed"
        )
        
        self.register_metric(
            "training.reward_average",
            MetricType.GAUGE,
            MonitoringScope.TRAINING,
            "Average training reward",
            unit="reward"
        )
        
        self.register_metric(
            "training.loss",
            MetricType.GAUGE,
            MonitoringScope.TRAINING,
            "Training loss",
            unit="loss"
        )
        
        logger.debug("Initialized default metrics")
    
    def _initialize_default_health_checks(self) -> None:
        """Initialize default health checks"""
        # System health checks
        self.register_health_check(
            "memory_usage",
            "Check system memory usage",
            self._check_memory_usage,
            interval_seconds=60
        )
        
        self.register_health_check(
            "disk_space",
            "Check available disk space",
            self._check_disk_space,
            interval_seconds=300
        )
        
        # RL system health checks
        self.register_health_check(
            "rl_system_responsive",
            "Check if RL system is responsive",
            self._check_rl_system_responsive,
            interval_seconds=30
        )
        
        logger.debug("Initialized default health checks")
    
    def register_metric(self, name: str, metric_type: MetricType, scope: MonitoringScope,
                       description: str = "", unit: str = "", tags: Optional[List[str]] = None,
                       alert_thresholds: Optional[Dict[AlertLevel, float]] = None,
                       retention_days: int = 30) -> None:
        """Register a metric definition"""
        with self.lock:
            definition = MetricDefinition(
                name=name,
                metric_type=metric_type,
                scope=scope,
                description=description,
                unit=unit,
                tags=tags or [],
                alert_thresholds=alert_thresholds or {},
                retention_days=retention_days
            )
            
            self.metric_definitions[name] = definition
            logger.debug(f"Registered metric: {name}")
    
    def record_metric(self, name: str, value: Union[int, float],
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     timestamp: Optional[float] = None) -> None:
        """Record a metric value"""
        if not self.config.enabled:
            return
        
        with self.lock:
            if name not in self.metric_definitions:
                logger.warning(f"Recording undefined metric: {name}")
                return
            
            metric_value = MetricValue(
                name=name,
                value=value,
                timestamp=timestamp or time.time(),
                tags=tags or {},
                metadata=metadata or {}
            )
            
            self.metric_values[name].append(metric_value)
            
            # Check for alerts
            self._check_metric_alerts(name, value, tags or {})
            
            logger.debug(f"Recorded metric {name}: {value}")
    
    def record_multiple_metrics(self, metrics: Dict[str, Union[int, float]],
                              tags: Optional[Dict[str, str]] = None,
                              timestamp: Optional[float] = None) -> None:
        """Record multiple metrics at once"""
        for name, value in metrics.items():
            self.record_metric(name, value, tags, timestamp=timestamp)
    
    def get_metric_values(self, name: str, start_time: Optional[float] = None,
                         end_time: Optional[float] = None,
                         tags: Optional[Dict[str, str]] = None) -> List[MetricValue]:
        """Get metric values within time range"""
        with self.lock:
            if name not in self.metric_values:
                return []
            
            values = list(self.metric_values[name])
            
            # Filter by time range
            if start_time or end_time:
                values = [
                    v for v in values
                    if (start_time is None or v.timestamp >= start_time) and
                       (end_time is None or v.timestamp <= end_time)
                ]
            
            # Filter by tags
            if tags:
                values = [
                    v for v in values
                    if all(v.tags.get(k) == v for k, v in tags.items())
                ]
            
            return values
    
    def get_metric_statistics(self, name: str, start_time: Optional[float] = None,
                            end_time: Optional[float] = None) -> Dict[str, float]:
        """Get statistical summary of metric values"""
        values = self.get_metric_values(name, start_time, end_time)
        
        if not values:
            return {}
        
        numeric_values = [v.value for v in values]
        
        return {
            'count': len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': statistics.mean(numeric_values),
            'median': statistics.median(numeric_values),
            'std_dev': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
            'sum': sum(numeric_values),
            'latest': numeric_values[-1] if numeric_values else 0.0
        }
    
    def _check_metric_alerts(self, name: str, value: float, tags: Dict[str, str]) -> None:
        """Check if metric value triggers any alerts"""
        if not self.config.enable_alerts:
            return
        
        definition = self.metric_definitions.get(name)
        if not definition or not definition.alert_thresholds:
            return
        
        for level, threshold in definition.alert_thresholds.items():
            alert_key = f"{name}_{level.value}"
            
            # Check cooldown
            if alert_key in self.alert_cooldowns:
                if time.time() - self.alert_cooldowns[alert_key] < self.config.alert_cooldown:
                    continue
            
            # Check threshold
            triggered = False
            if level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]:
                # For warning/error/critical, trigger if value exceeds threshold
                triggered = value > threshold
            else:
                # For info, trigger if value equals threshold
                triggered = value == threshold
            
            if triggered:
                self._create_alert(name, level, threshold, value, tags)
                self.alert_cooldowns[alert_key] = time.time()
    
    def _create_alert(self, metric_name: str, level: AlertLevel, threshold: float,
                     current_value: float, tags: Dict[str, str]) -> None:
        """Create a new alert"""
        alert_id = f"{metric_name}_{level.value}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            name=f"{metric_name} {level.value}",
            level=level,
            message=f"Metric {metric_name} {level.value}: {current_value} (threshold: {threshold})",
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value,
            timestamp=time.time(),
            tags=tags
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.warning(f"Alert handler failed: {e}")
        
        logger.warning(f"Alert created: {alert.message}")
    
    def register_health_check(self, name: str, description: str,
                            check_function: Callable[[], bool],
                            interval_seconds: int = 60,
                            timeout_seconds: int = 10,
                            max_failures: int = 3) -> None:
        """Register a health check"""
        health_check = HealthCheck(
            name=name,
            description=description,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            max_failures=max_failures
        )
        
        self.health_checks[name] = health_check
        logger.debug(f"Registered health check: {name}")
    
    def _check_memory_usage(self) -> bool:
        """Check system memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            self.record_metric("system.memory_usage", usage_percent)
            
            return usage_percent < 90.0  # Consider healthy if under 90%
        except Exception as e:
            logger.warning(f"Memory usage check failed: {e}")
            return False
    
    def _check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.storage_path)
            usage_percent = (used / total) * 100
            
            self.record_metric("system.disk_usage", usage_percent)
            
            return usage_percent < 90.0  # Consider healthy if under 90%
        except Exception as e:
            logger.warning(f"Disk space check failed: {e}")
            return False
    
    def _check_rl_system_responsive(self) -> bool:
        """Check if RL system is responsive"""
        try:
            # Simple responsiveness check
            # In a real implementation, this would ping the RL system
            return True
        except Exception as e:
            logger.warning(f"RL system responsiveness check failed: {e}")
            return False
    
    async def run_health_check(self, name: str) -> bool:
        """Run a specific health check"""
        if name not in self.health_checks:
            return False
        
        health_check = self.health_checks[name]
        
        if not health_check.enabled:
            return True
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, health_check.check_function
                ),
                timeout=health_check.timeout_seconds
            )
            
            health_check.last_check = time.time()
            health_check.last_result = result
            
            if result:
                health_check.consecutive_failures = 0
            else:
                health_check.consecutive_failures += 1
                
                # Create alert if max failures reached
                if health_check.consecutive_failures >= health_check.max_failures:
                    self._create_alert(
                        f"health_check.{name}",
                        AlertLevel.CRITICAL,
                        health_check.max_failures,
                        health_check.consecutive_failures,
                        {'health_check': name}
                    )
            
            return result
        
        except asyncio.TimeoutError:
            logger.warning(f"Health check {name} timed out")
            health_check.consecutive_failures += 1
            return False
        
        except Exception as e:
            logger.error(f"Health check {name} failed: {e}")
            health_check.consecutive_failures += 1
            return False
    
    async def run_all_health_checks(self) -> Dict[str, bool]:
        """Run all enabled health checks"""
        results = {}
        
        for name, health_check in self.health_checks.items():
            if health_check.enabled:
                results[name] = await self.run_health_check(name)
        
        return results
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler"""
        self.alert_handlers.append(handler)
        logger.debug(f"Added alert handler: {handler.__name__}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get active (unresolved) alerts"""
        alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        health_results = {}
        overall_healthy = True
        
        for name, health_check in self.health_checks.items():
            if health_check.enabled:
                is_healthy = (
                    health_check.last_result is True and
                    health_check.consecutive_failures < health_check.max_failures
                )
                health_results[name] = {
                    'healthy': is_healthy,
                    'last_check': health_check.last_check,
                    'consecutive_failures': health_check.consecutive_failures,
                    'description': health_check.description
                }
                
                if not is_healthy:
                    overall_healthy = False
        
        active_alerts = self.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
        
        return {
            'overall_healthy': overall_healthy and len(critical_alerts) == 0,
            'health_checks': health_results,
            'active_alerts': len(active_alerts),
            'critical_alerts': len(critical_alerts),
            'uptime': time.time() - self.started_at,
            'last_updated': time.time()
        }
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks"""
        if self.running:
            return
        
        self.running = True
        
        # Start metric collection task
        task = asyncio.create_task(self._metric_collection_task())
        self.background_tasks.append(task)
        
        # Start health check task
        if self.config.enable_health_checks:
            task = asyncio.create_task(self._health_check_task())
            self.background_tasks.append(task)
        
        # Start data export task
        if self.config.enable_json_export:
            task = asyncio.create_task(self._data_export_task())
            self.background_tasks.append(task)
        
        # Start cleanup task
        task = asyncio.create_task(self._cleanup_task())
        self.background_tasks.append(task)
        
        logger.info(f"Started {len(self.background_tasks)} monitoring tasks")
    
    async def _metric_collection_task(self) -> None:
        """Background task for metric collection"""
        while self.running:
            try:
                # Record system uptime
                self.record_metric("system.uptime", time.time() - self.started_at)
                
                # Aggregate metrics
                await self._aggregate_metrics()
                
                await asyncio.sleep(self.config.collection_interval)
            
            except Exception as e:
                logger.error(f"Metric collection task error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _health_check_task(self) -> None:
        """Background task for health checks"""
        while self.running:
            try:
                await self.run_all_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            
            except Exception as e:
                logger.error(f"Health check task error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _data_export_task(self) -> None:
        """Background task for data export"""
        while self.running:
            try:
                await self._export_metrics_to_json()
                await asyncio.sleep(self.config.json_export_interval)
            
            except Exception as e:
                logger.error(f"Data export task error: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _cleanup_task(self) -> None:
        """Background task for data cleanup"""
        while self.running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Run every hour
            
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(3600)  # Wait before retrying
    
    async def _aggregate_metrics(self) -> None:
        """Aggregate metrics for performance"""
        current_time = time.time()
        
        with self.lock:
            for name, values in self.metric_values.items():
                if not values:
                    continue
                
                definition = self.metric_definitions.get(name)
                if not definition:
                    continue
                
                # Get values from last aggregation window
                window_start = current_time - definition.aggregation_window
                window_values = [
                    v.value for v in values
                    if v.timestamp >= window_start
                ]
                
                if window_values:
                    aggregated = {
                        'count': len(window_values),
                        'sum': sum(window_values),
                        'avg': sum(window_values) / len(window_values),
                        'min': min(window_values),
                        'max': max(window_values),
                        'timestamp': current_time
                    }
                    
                    self.aggregated_metrics[name] = aggregated
    
    async def _export_metrics_to_json(self) -> None:
        """Export metrics to JSON file"""
        try:
            export_data = {
                'timestamp': time.time(),
                'monitoring_id': self.monitoring_id,
                'metrics': {},
                'alerts': [alert.__dict__ for alert in self.get_active_alerts()],
                'health': self.get_system_health()
            }
            
            # Export recent metric values
            current_time = time.time()
            one_hour_ago = current_time - 3600
            
            with self.lock:
                for name, values in self.metric_values.items():
                    recent_values = [
                        {
                            'value': v.value,
                            'timestamp': v.timestamp,
                            'tags': v.tags
                        }
                        for v in values
                        if v.timestamp >= one_hour_ago
                    ]
                    
                    if recent_values:
                        export_data['metrics'][name] = {
                            'definition': self.metric_definitions[name].__dict__,
                            'values': recent_values,
                            'aggregated': self.aggregated_metrics.get(name, {})
                        }
            
            # Save to file
            export_file = self.storage_path / f"metrics_{int(current_time)}.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.debug(f"Exported metrics to {export_file}")
        
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metric data"""
        current_time = time.time()
        
        with self.lock:
            # Clean up metric values
            for name, values in self.metric_values.items():
                definition = self.metric_definitions.get(name)
                if not definition:
                    continue
                
                retention_seconds = definition.retention_days * 24 * 3600
                cutoff_time = current_time - retention_seconds
                
                # Remove old values
                while values and values[0].timestamp < cutoff_time:
                    values.popleft()
            
            # Clean up alert history
            self.alert_history = [
                alert for alert in self.alert_history
                if current_time - alert.timestamp < (30 * 24 * 3600)  # Keep 30 days
            ]
        
        # Clean up old export files
        try:
            for file_path in self.storage_path.glob("metrics_*.json"):
                file_age = current_time - file_path.stat().st_mtime
                if file_age > (7 * 24 * 3600):  # Keep 7 days
                    file_path.unlink()
                    logger.debug(f"Deleted old export file: {file_path}")
        
        except Exception as e:
            logger.warning(f"Error cleaning up export files: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        with self.lock:
            return {
                'monitoring_id': self.monitoring_id,
                'started_at': self.started_at,
                'uptime': time.time() - self.started_at,
                'running': self.running,
                'config': {
                    'enabled': self.config.enabled,
                    'collection_interval': self.config.collection_interval,
                    'retention_days': self.config.retention_days,
                    'enable_alerts': self.config.enable_alerts,
                    'enable_health_checks': self.config.enable_health_checks
                },
                'metrics': {
                    'definitions': len(self.metric_definitions),
                    'total_values': sum(len(values) for values in self.metric_values.values()),
                    'aggregated': len(self.aggregated_metrics)
                },
                'alerts': {
                    'active': len(self.get_active_alerts()),
                    'total_history': len(self.alert_history),
                    'handlers': len(self.alert_handlers)
                },
                'health_checks': {
                    'registered': len(self.health_checks),
                    'enabled': len([hc for hc in self.health_checks.values() if hc.enabled])
                },
                'background_tasks': len(self.background_tasks)
            }
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring and cleanup"""
        logger.info("Stopping monitoring integration")
        
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Final data export
        if self.config.enable_json_export:
            await self._export_metrics_to_json()
        
        logger.info("Monitoring integration stopped")
    
    def cleanup(self) -> None:
        """Cleanup monitoring resources"""
        logger.info("Cleaning up monitoring integration")
        
        with self.lock:
            self.metric_values.clear()
            self.aggregated_metrics.clear()
            self.alerts.clear()
            self.alert_history.clear()
            self.alert_cooldowns.clear()
            self.alert_handlers.clear()
            self.health_checks.clear()
            self.system_health.clear()
        
        logger.info("Monitoring integration cleanup complete")