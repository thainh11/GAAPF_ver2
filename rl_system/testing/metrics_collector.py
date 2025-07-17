"""Metrics Collection Module for GAAPF RL System

This module provides comprehensive metrics collection and analysis capabilities
for monitoring and evaluating the performance of the RL-enhanced system.
"""

import numpy as np
import json
import os
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
from scipy import stats
import pickle

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"  # Cumulative count
    GAUGE = "gauge"      # Current value
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"      # Duration measurements
    RATE = "rate"        # Events per time unit
    PERCENTAGE = "percentage"  # Percentage values

class AggregationType(Enum):
    """Types of aggregation"""
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    STD = "std"
    VARIANCE = "variance"

class MetricCategory(Enum):
    """Categories of metrics"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    USER_EXPERIENCE = "user_experience"
    SYSTEM_HEALTH = "system_health"
    LEARNING = "learning"
    COLLABORATION = "collaboration"
    BUSINESS = "business"

@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    description: str
    metric_type: MetricType
    category: MetricCategory
    unit: str = ""
    
    # Aggregation settings
    default_aggregation: AggregationType = AggregationType.MEAN
    supported_aggregations: List[AggregationType] = field(default_factory=list)
    
    # Value constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Collection settings
    collection_interval: float = 1.0  # seconds
    retention_period: int = 7  # days
    
    # Alert thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    higher_is_better: bool = True
    
    def __post_init__(self):
        if not self.supported_aggregations:
            if self.metric_type == MetricType.COUNTER:
                self.supported_aggregations = [AggregationType.SUM, AggregationType.COUNT]
            elif self.metric_type == MetricType.GAUGE:
                self.supported_aggregations = [AggregationType.MEAN, AggregationType.MIN, AggregationType.MAX]
            elif self.metric_type == MetricType.HISTOGRAM:
                self.supported_aggregations = [
                    AggregationType.MEAN, AggregationType.MEDIAN, 
                    AggregationType.PERCENTILE_95, AggregationType.PERCENTILE_99
                ]
            elif self.metric_type == MetricType.TIMER:
                self.supported_aggregations = [
                    AggregationType.MEAN, AggregationType.MEDIAN,
                    AggregationType.PERCENTILE_95, AggregationType.PERCENTILE_99
                ]
            else:
                self.supported_aggregations = [AggregationType.MEAN]

@dataclass
class MetricValue:
    """Individual metric value"""
    metric_name: str
    value: Union[float, int]
    timestamp: float
    
    # Context information
    tags: Dict[str, str] = field(default_factory=dict)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    # Quality indicators
    confidence: float = 1.0
    source: str = "system"
    
@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    # Storage settings
    storage_backend: str = "sqlite"  # sqlite, influxdb, prometheus
    storage_path: str = "metrics.db"
    
    # Collection settings
    collection_enabled: bool = True
    batch_size: int = 100
    flush_interval: float = 10.0  # seconds
    
    # Retention settings
    default_retention_days: int = 30
    high_frequency_retention_days: int = 7
    
    # Performance settings
    max_memory_usage_mb: int = 100
    compression_enabled: bool = True
    
    # Export settings
    export_enabled: bool = True
    export_formats: List[str] = field(default_factory=lambda: ['json', 'csv'])
    export_interval_hours: int = 24
    
    # Alert settings
    alerting_enabled: bool = True
    alert_check_interval: float = 60.0  # seconds
    
class MetricsCollector:
    """Comprehensive metrics collection and analysis system"""
    
    def __init__(self, config: MetricsConfig):
        """
        Initialize metrics collector.
        
        Parameters:
        ----------
        config : MetricsConfig
            Metrics collection configuration
        """
        self.config = config
        
        # Metric definitions
        self.metric_definitions = {}
        self.metric_values = defaultdict(list)
        
        # Storage
        self.storage_lock = threading.Lock()
        self.pending_values = deque()
        
        # Aggregation cache
        self.aggregation_cache = {}
        self.cache_lock = threading.Lock()
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False
        
        # Initialize storage
        self._initialize_storage()
        
        # Register default metrics
        self._register_default_metrics()
        
        logger.info("Initialized metrics collector")
    
    def _initialize_storage(self) -> None:
        """Initialize storage backend"""
        if self.config.storage_backend == "sqlite":
            self._initialize_sqlite_storage()
        else:
            logger.warning(f"Unsupported storage backend: {self.config.storage_backend}")
    
    def _initialize_sqlite_storage(self) -> None:
        """Initialize SQLite storage"""
        self.db_path = self.config.storage_path
        
        # Create database and tables
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_definitions (
                    name TEXT PRIMARY KEY,
                    definition TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    tags TEXT,
                    session_id TEXT,
                    user_id TEXT,
                    agent_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    source TEXT DEFAULT 'system'
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_values_name_timestamp 
                ON metric_values(metric_name, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_values_timestamp 
                ON metric_values(timestamp)
            """)
            
            conn.commit()
        
        logger.info(f"Initialized SQLite storage at {self.db_path}")
    
    def _register_default_metrics(self) -> None:
        """Register default metrics for GAAPF RL system"""
        default_metrics = [
            # Performance metrics
            MetricDefinition(
                name="task_completion_rate",
                description="Rate of successful task completions",
                metric_type=MetricType.PERCENTAGE,
                category=MetricCategory.PERFORMANCE,
                unit="%",
                min_value=0.0,
                max_value=100.0,
                warning_threshold=70.0,
                critical_threshold=50.0
            ),
            
            MetricDefinition(
                name="response_time",
                description="Time taken to generate responses",
                metric_type=MetricType.TIMER,
                category=MetricCategory.EFFICIENCY,
                unit="seconds",
                min_value=0.0,
                warning_threshold=5.0,
                critical_threshold=10.0,
                higher_is_better=False
            ),
            
            MetricDefinition(
                name="user_satisfaction",
                description="User satisfaction score",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.USER_EXPERIENCE,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                warning_threshold=0.7,
                critical_threshold=0.5
            ),
            
            MetricDefinition(
                name="response_quality",
                description="Quality of generated responses",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.QUALITY,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                warning_threshold=0.75,
                critical_threshold=0.6
            ),
            
            MetricDefinition(
                name="learning_effectiveness",
                description="Effectiveness of learning interactions",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.LEARNING,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                warning_threshold=0.6,
                critical_threshold=0.4
            ),
            
            MetricDefinition(
                name="agent_utilization",
                description="Agent resource utilization",
                metric_type=MetricType.PERCENTAGE,
                category=MetricCategory.EFFICIENCY,
                unit="%",
                min_value=0.0,
                max_value=100.0,
                warning_threshold=90.0,
                critical_threshold=95.0,
                higher_is_better=False
            ),
            
            MetricDefinition(
                name="collaboration_score",
                description="Multi-agent collaboration effectiveness",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.COLLABORATION,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                warning_threshold=0.6,
                critical_threshold=0.4
            ),
            
            MetricDefinition(
                name="error_rate",
                description="Rate of errors in system operations",
                metric_type=MetricType.PERCENTAGE,
                category=MetricCategory.SYSTEM_HEALTH,
                unit="%",
                min_value=0.0,
                max_value=100.0,
                warning_threshold=5.0,
                critical_threshold=10.0,
                higher_is_better=False
            ),
            
            MetricDefinition(
                name="memory_usage",
                description="System memory usage",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM_HEALTH,
                unit="MB",
                min_value=0.0,
                warning_threshold=1000.0,
                critical_threshold=1500.0,
                higher_is_better=False
            ),
            
            MetricDefinition(
                name="rl_reward",
                description="Reinforcement learning reward signal",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.LEARNING,
                unit="reward",
                collection_interval=0.1  # High frequency for RL
            ),
            
            MetricDefinition(
                name="exploration_rate",
                description="RL exploration rate",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.LEARNING,
                unit="rate",
                min_value=0.0,
                max_value=1.0
            ),
            
            MetricDefinition(
                name="constellation_formation_time",
                description="Time to form agent constellations",
                metric_type=MetricType.TIMER,
                category=MetricCategory.EFFICIENCY,
                unit="seconds",
                min_value=0.0,
                warning_threshold=2.0,
                critical_threshold=5.0,
                higher_is_better=False
            )
        ]
        
        for metric_def in default_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_def: MetricDefinition) -> None:
        """Register a new metric definition"""
        self.metric_definitions[metric_def.name] = metric_def
        
        # Store in database
        if self.config.storage_backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO metric_definitions (name, definition, created_at) VALUES (?, ?, ?)",
                    (metric_def.name, json.dumps(asdict(metric_def), default=str), time.time())
                )
                conn.commit()
        
        logger.debug(f"Registered metric: {metric_def.name}")
    
    def record_metric(self, metric_name: str, value: Union[float, int], 
                     tags: Optional[Dict[str, str]] = None,
                     session_id: Optional[str] = None,
                     user_id: Optional[str] = None,
                     agent_id: Optional[str] = None,
                     confidence: float = 1.0,
                     source: str = "system") -> None:
        """Record a metric value"""
        if not self.config.collection_enabled:
            return
        
        if metric_name not in self.metric_definitions:
            logger.warning(f"Unknown metric: {metric_name}")
            return
        
        metric_def = self.metric_definitions[metric_name]
        
        # Validate value
        if metric_def.min_value is not None and value < metric_def.min_value:
            logger.warning(f"Metric {metric_name} value {value} below minimum {metric_def.min_value}")
            value = metric_def.min_value
        
        if metric_def.max_value is not None and value > metric_def.max_value:
            logger.warning(f"Metric {metric_name} value {value} above maximum {metric_def.max_value}")
            value = metric_def.max_value
        
        # Create metric value
        metric_value = MetricValue(
            metric_name=metric_name,
            value=float(value),
            timestamp=time.time(),
            tags=tags or {},
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            confidence=confidence,
            source=source
        )
        
        # Add to pending queue
        self.pending_values.append(metric_value)
        
        # Flush if batch size reached
        if len(self.pending_values) >= self.config.batch_size:
            self._flush_pending_values()
        
        # Check alerts
        if self.config.alerting_enabled:
            self._check_alert_thresholds(metric_def, value)
    
    def record_multiple_metrics(self, metrics: Dict[str, Union[float, int]],
                              tags: Optional[Dict[str, str]] = None,
                              session_id: Optional[str] = None,
                              user_id: Optional[str] = None,
                              agent_id: Optional[str] = None) -> None:
        """Record multiple metrics at once"""
        for metric_name, value in metrics.items():
            self.record_metric(
                metric_name=metric_name,
                value=value,
                tags=tags,
                session_id=session_id,
                user_id=user_id,
                agent_id=agent_id
            )
    
    def _flush_pending_values(self) -> None:
        """Flush pending metric values to storage"""
        if not self.pending_values:
            return
        
        with self.storage_lock:
            values_to_flush = list(self.pending_values)
            self.pending_values.clear()
        
        if self.config.storage_backend == "sqlite":
            self._flush_to_sqlite(values_to_flush)
        
        logger.debug(f"Flushed {len(values_to_flush)} metric values")
    
    def _flush_to_sqlite(self, values: List[MetricValue]) -> None:
        """Flush values to SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = []
                for value in values:
                    data.append((
                        value.metric_name,
                        value.value,
                        value.timestamp,
                        json.dumps(value.tags) if value.tags else None,
                        value.session_id,
                        value.user_id,
                        value.agent_id,
                        value.confidence,
                        value.source
                    ))
                
                conn.executemany(
                    """INSERT INTO metric_values 
                       (metric_name, value, timestamp, tags, session_id, user_id, agent_id, confidence, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    data
                )
                conn.commit()
        
        except Exception as e:
            logger.error(f"Error flushing to SQLite: {e}")
    
    def _check_alert_thresholds(self, metric_def: MetricDefinition, value: float) -> None:
        """Check if metric value exceeds alert thresholds"""
        if metric_def.critical_threshold is not None:
            if (metric_def.higher_is_better and value < metric_def.critical_threshold) or \
               (not metric_def.higher_is_better and value > metric_def.critical_threshold):
                logger.critical(f"CRITICAL: {metric_def.name} = {value} (threshold: {metric_def.critical_threshold})")
        
        elif metric_def.warning_threshold is not None:
            if (metric_def.higher_is_better and value < metric_def.warning_threshold) or \
               (not metric_def.higher_is_better and value > metric_def.warning_threshold):
                logger.warning(f"WARNING: {metric_def.name} = {value} (threshold: {metric_def.warning_threshold})")
    
    def get_metric_values(self, metric_name: str, 
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None,
                         tags: Optional[Dict[str, str]] = None,
                         limit: Optional[int] = None) -> List[MetricValue]:
        """Get metric values from storage"""
        if self.config.storage_backend == "sqlite":
            return self._get_sqlite_values(metric_name, start_time, end_time, tags, limit)
        else:
            return []
    
    def _get_sqlite_values(self, metric_name: str,
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None,
                          tags: Optional[Dict[str, str]] = None,
                          limit: Optional[int] = None) -> List[MetricValue]:
        """Get values from SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM metric_values WHERE metric_name = ?"
                params = [metric_name]
                
                if start_time is not None:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time is not None:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC"
                
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                values = []
                for row in rows:
                    tags_data = json.loads(row[4]) if row[4] else {}
                    
                    # Filter by tags if specified
                    if tags:
                        if not all(tags_data.get(k) == v for k, v in tags.items()):
                            continue
                    
                    value = MetricValue(
                        metric_name=row[1],
                        value=row[2],
                        timestamp=row[3],
                        tags=tags_data,
                        session_id=row[5],
                        user_id=row[6],
                        agent_id=row[7],
                        confidence=row[8] or 1.0,
                        source=row[9] or "system"
                    )
                    values.append(value)
                
                return values
        
        except Exception as e:
            logger.error(f"Error getting SQLite values: {e}")
            return []
    
    def aggregate_metric(self, metric_name: str,
                        aggregation_type: AggregationType,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        tags: Optional[Dict[str, str]] = None,
                        group_by_interval: Optional[float] = None) -> Union[float, List[Tuple[float, float]]]:
        """Aggregate metric values"""
        values = self.get_metric_values(metric_name, start_time, end_time, tags)
        
        if not values:
            return 0.0 if group_by_interval is None else []
        
        if group_by_interval is None:
            # Single aggregation
            return self._calculate_aggregation([v.value for v in values], aggregation_type)
        else:
            # Time-series aggregation
            return self._calculate_time_series_aggregation(values, aggregation_type, group_by_interval)
    
    def _calculate_aggregation(self, values: List[float], aggregation_type: AggregationType) -> float:
        """Calculate aggregation for a list of values"""
        if not values:
            return 0.0
        
        values_array = np.array(values)
        
        if aggregation_type == AggregationType.SUM:
            return float(np.sum(values_array))
        elif aggregation_type == AggregationType.MEAN:
            return float(np.mean(values_array))
        elif aggregation_type == AggregationType.MEDIAN:
            return float(np.median(values_array))
        elif aggregation_type == AggregationType.MIN:
            return float(np.min(values_array))
        elif aggregation_type == AggregationType.MAX:
            return float(np.max(values_array))
        elif aggregation_type == AggregationType.COUNT:
            return float(len(values_array))
        elif aggregation_type == AggregationType.PERCENTILE_95:
            return float(np.percentile(values_array, 95))
        elif aggregation_type == AggregationType.PERCENTILE_99:
            return float(np.percentile(values_array, 99))
        elif aggregation_type == AggregationType.STD:
            return float(np.std(values_array, ddof=1))
        elif aggregation_type == AggregationType.VARIANCE:
            return float(np.var(values_array, ddof=1))
        else:
            return float(np.mean(values_array))  # Default to mean
    
    def _calculate_time_series_aggregation(self, values: List[MetricValue],
                                         aggregation_type: AggregationType,
                                         interval: float) -> List[Tuple[float, float]]:
        """Calculate time-series aggregation"""
        if not values:
            return []
        
        # Sort by timestamp
        sorted_values = sorted(values, key=lambda v: v.timestamp)
        
        # Group by time intervals
        start_time = sorted_values[0].timestamp
        end_time = sorted_values[-1].timestamp
        
        result = []
        current_time = start_time
        
        while current_time < end_time:
            interval_end = current_time + interval
            
            # Get values in this interval
            interval_values = [
                v.value for v in sorted_values 
                if current_time <= v.timestamp < interval_end
            ]
            
            if interval_values:
                aggregated_value = self._calculate_aggregation(interval_values, aggregation_type)
                result.append((current_time, aggregated_value))
            
            current_time = interval_end
        
        return result
    
    def get_metric_summary(self, metric_name: str,
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None,
                          tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get comprehensive summary of a metric"""
        values = self.get_metric_values(metric_name, start_time, end_time, tags)
        
        if not values:
            return {}
        
        value_list = [v.value for v in values]
        
        summary = {
            'count': len(value_list),
            'mean': self._calculate_aggregation(value_list, AggregationType.MEAN),
            'median': self._calculate_aggregation(value_list, AggregationType.MEDIAN),
            'min': self._calculate_aggregation(value_list, AggregationType.MIN),
            'max': self._calculate_aggregation(value_list, AggregationType.MAX),
            'std': self._calculate_aggregation(value_list, AggregationType.STD),
            'p95': self._calculate_aggregation(value_list, AggregationType.PERCENTILE_95),
            'p99': self._calculate_aggregation(value_list, AggregationType.PERCENTILE_99)
        }
        
        return summary
    
    def get_all_metrics_summary(self, start_time: Optional[float] = None,
                               end_time: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """Get summary for all registered metrics"""
        summary = {}
        
        for metric_name in self.metric_definitions.keys():
            metric_summary = self.get_metric_summary(metric_name, start_time, end_time)
            if metric_summary:
                summary[metric_name] = metric_summary
        
        return summary
    
    def export_metrics(self, format_type: str = "json",
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      output_path: Optional[str] = None) -> str:
        """Export metrics data"""
        if not self.config.export_enabled:
            raise ValueError("Export is disabled")
        
        # Get all metrics data
        all_data = {}
        for metric_name in self.metric_definitions.keys():
            values = self.get_metric_values(metric_name, start_time, end_time)
            all_data[metric_name] = [asdict(v) for v in values]
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"metrics_export_{timestamp}.{format_type}"
        
        # Export in specified format
        if format_type == "json":
            with open(output_path, 'w') as f:
                json.dump(all_data, f, indent=2, default=str)
        
        elif format_type == "csv":
            # Flatten data for CSV
            rows = []
            for metric_name, values in all_data.items():
                for value_dict in values:
                    row = {
                        'metric_name': metric_name,
                        **value_dict
                    }
                    rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
        
        elif format_type == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(all_data, f)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        logger.info(f"Exported metrics to {output_path}")
        return output_path
    
    def cleanup_old_data(self, retention_days: Optional[int] = None) -> int:
        """Clean up old metric data"""
        if retention_days is None:
            retention_days = self.config.default_retention_days
        
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        if self.config.storage_backend == "sqlite":
            return self._cleanup_sqlite_data(cutoff_time)
        
        return 0
    
    def _cleanup_sqlite_data(self, cutoff_time: float) -> int:
        """Clean up old SQLite data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM metric_values WHERE timestamp < ?",
                    (cutoff_time,)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old metric records")
                return deleted_count
        
        except Exception as e:
            logger.error(f"Error cleaning up SQLite data: {e}")
            return 0
    
    def start_background_processing(self) -> None:
        """Start background processing tasks"""
        if self.running:
            return
        
        self.running = True
        
        # Start flush task
        self.executor.submit(self._background_flush_task)
        
        # Start cleanup task
        self.executor.submit(self._background_cleanup_task)
        
        logger.info("Started background processing")
    
    def _background_flush_task(self) -> None:
        """Background task to flush pending values"""
        while self.running:
            try:
                time.sleep(self.config.flush_interval)
                if self.pending_values:
                    self._flush_pending_values()
            except Exception as e:
                logger.error(f"Error in background flush task: {e}")
    
    def _background_cleanup_task(self) -> None:
        """Background task to clean up old data"""
        while self.running:
            try:
                # Run cleanup once per day
                time.sleep(24 * 3600)
                self.cleanup_old_data()
            except Exception as e:
                logger.error(f"Error in background cleanup task: {e}")
    
    def stop_background_processing(self) -> None:
        """Stop background processing tasks"""
        self.running = False
        
        # Flush any remaining values
        self._flush_pending_values()
        
        logger.info("Stopped background processing")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'registered_metrics': len(self.metric_definitions),
            'pending_values': len(self.pending_values),
            'storage_backend': self.config.storage_backend,
            'collection_enabled': self.config.collection_enabled,
            'background_processing': self.running
        }
        
        # Add storage-specific stats
        if self.config.storage_backend == "sqlite":
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM metric_values")
                    stats['total_stored_values'] = cursor.fetchone()[0]
                    
                    cursor = conn.execute(
                        "SELECT metric_name, COUNT(*) FROM metric_values GROUP BY metric_name"
                    )
                    stats['values_by_metric'] = dict(cursor.fetchall())
            except Exception as e:
                logger.error(f"Error getting storage stats: {e}")
        
        return stats
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.stop_background_processing()
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Metrics collector cleanup completed")