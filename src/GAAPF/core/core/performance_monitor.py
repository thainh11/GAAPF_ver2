"""
Performance monitoring module for the GAAPF framework.
"""

import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Enhanced performance metric data class."""
    name: str
    value: float
    unit: str
    timestamp: str
    component: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorRecord:
    """Error record with detailed information."""
    operation: str
    component: str
    error_message: str
    severity: ErrorSeverity
    timestamp: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Advanced performance monitoring system with comprehensive metrics."""
    
    def __init__(self, is_logging: bool = False, max_history: int = 10000):
        self.is_logging = is_logging
        self.max_history = max_history
        
        # Core metrics storage
        self.metrics = defaultdict(deque)
        self.response_times = defaultdict(deque)
        self.error_counts = defaultdict(int)
        self.operation_counts = defaultdict(int)
        
        # Enhanced tracking
        self.error_records = deque(maxlen=max_history)
        self.memory_usage = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=1000)
        self.throughput_metrics = defaultdict(deque)
        
        # Performance thresholds
        self.response_time_thresholds = {
            'fast': 0.1,    # < 100ms
            'normal': 1.0,  # < 1s
            'slow': 5.0,    # < 5s
            'critical': 10.0 # >= 10s
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_active = True
        self._start_system_monitoring()
        
        if self.is_logging:
            logger.info("Enhanced PerformanceMonitor initialized")
    
    def record_operation_start(self, operation_name: str, component: str = "system") -> str:
        """Record the start of an operation for timing."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        self._operation_start_times = getattr(self, '_operation_start_times', {})
        self._operation_start_times[operation_id] = {
            "start_time": time.time(),
            "operation_name": operation_name,
            "component": component
        }
        
        return operation_id
    
    def record_operation_end(self, operation_id: str, success: bool = True, error_message: str = None) -> None:
        """Record the end of an operation and calculate metrics."""
        try:
            if not hasattr(self, '_operation_start_times'):
                return
            
            if operation_id not in self._operation_start_times:
                return
            
            operation_data = self._operation_start_times[operation_id]
            end_time = time.time()
            duration = end_time - operation_data["start_time"]
            
            operation_name = operation_data["operation_name"]
            component = operation_data["component"]
            
            # Record response time
            self.record_response_time(operation_name, duration, component)
            
            # Record operation count
            self.operation_counts[f"{component}.{operation_name}"] += 1
            
            # Record error if failed
            if not success:
                self.record_error(operation_name, error_message, component)
            
            # Cleanup
            del self._operation_start_times[operation_id]
            
        except Exception as e:
            logger.error(f"Error recording operation end: {str(e)}")
    
    def record_response_time(self, operation: str, duration: float, component: str = "system") -> None:
        """Record response time for an operation."""
        metric = PerformanceMetric(
            name=f"{operation}_response_time",
            value=duration,
            unit="seconds",
            timestamp=datetime.now().isoformat(),
            component=component
        )
        
        self._add_metric(metric)
        
        # Also add to response times for quick access
        key = f"{component}.{operation}"
        self.response_times[key].append(duration)
        
        # Limit history size
        if len(self.response_times[key]) > 1000:
            self.response_times[key].popleft()
    
    def record_error(self, operation: str, error_message: str = None, component: str = "system", 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM, stack_trace: str = None, 
                    context: Dict[str, Any] = None) -> None:
        """Record an error occurrence with detailed information."""
        with self._lock:
            key = f"{component}.{operation}"
            self.error_counts[key] += 1
            
            # Create detailed error record
            error_record = ErrorRecord(
                operation=operation,
                component=component,
                error_message=error_message or "Unknown error",
                severity=severity,
                timestamp=datetime.now().isoformat(),
                stack_trace=stack_trace,
                context=context or {}
            )
            
            self.error_records.append(error_record)
            
            if self.is_logging:
                logger.warning(f"Error recorded for {key} [{severity.value}]: {error_message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            with self._lock:
                # Calculate average response times and categorize
                avg_response_times = {}
                response_time_categories = {'fast': 0, 'normal': 0, 'slow': 0, 'critical': 0}
                
                for key, times in self.response_times.items():
                    if times:
                        avg_time = sum(times) / len(times)
                        avg_response_times[key] = avg_time
                        
                        # Categorize response time
                        if avg_time < self.response_time_thresholds['fast']:
                            response_time_categories['fast'] += 1
                        elif avg_time < self.response_time_thresholds['normal']:
                            response_time_categories['normal'] += 1
                        elif avg_time < self.response_time_thresholds['slow']:
                            response_time_categories['slow'] += 1
                        else:
                            response_time_categories['critical'] += 1
                
                # Calculate error rates and categorize by severity
                error_rates = {}
                error_severity_counts = {severity.value: 0 for severity in ErrorSeverity}
                
                for key in self.error_counts:
                    operation_count = self.operation_counts.get(key, 0)
                    error_count = self.error_counts[key]
                    
                    if operation_count > 0:
                        error_rates[key] = (error_count / operation_count) * 100
                    else:
                        error_rates[key] = 0
                
                # Count errors by severity
                for error_record in self.error_records:
                    error_severity_counts[error_record.severity.value] += 1
                
                # System resource usage
                current_memory = self.memory_usage[-1] if self.memory_usage else 0
                current_cpu = self.cpu_usage[-1] if self.cpu_usage else 0
                avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
                avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
                
                return {
                    "summary_generated": datetime.now().isoformat(),
                    "response_times": {
                        "averages": avg_response_times,
                        "categories": response_time_categories,
                        "thresholds": self.response_time_thresholds
                    },
                    "errors": {
                        "rates": error_rates,
                        "severity_breakdown": error_severity_counts,
                        "total_errors": sum(self.error_counts.values()),
                        "recent_errors": len([e for e in self.error_records 
                                             if datetime.fromisoformat(e.timestamp) > 
                                             datetime.now() - timedelta(hours=1)])
                    },
                    "operations": {
                        "counts": dict(self.operation_counts),
                        "total_operations": sum(self.operation_counts.values())
                    },
                    "system_resources": {
                        "memory": {
                            "current_mb": current_memory,
                            "average_mb": avg_memory
                        },
                        "cpu": {
                            "current_percent": current_cpu,
                            "average_percent": avg_cpu
                        }
                    },
                    "health_score": self._calculate_health_score()
                }
                
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return {"error": f"Failed to generate summary: {str(e)}"}
    
    def _add_metric(self, metric: PerformanceMetric) -> None:
        """Add a metric to the storage."""
        key = f"{metric.component}.{metric.name}"
        self.metrics[key].append(metric)
        
        # Limit history size
        if len(self.metrics[key]) > self.max_history:
            self.metrics[key].popleft()
    
    def _start_system_monitoring(self):
        """Start background system resource monitoring."""
        def monitor_resources():
            while self._monitoring_active:
                try:
                    # Get current system metrics
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    
                    with self._lock:
                        self.memory_usage.append(memory_mb)
                        self.cpu_usage.append(cpu_percent)
                    
                    time.sleep(5)  # Monitor every 5 seconds
                except Exception as e:
                    if self.is_logging:
                        logger.error(f"Error in system monitoring: {e}")
                    time.sleep(10)  # Wait longer on error
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        try:
            score = 100.0
            
            # Response time penalty
            if self.response_times:
                avg_response_time = sum(sum(times) / len(times) for times in self.response_times.values() if times) / len(self.response_times)
                if avg_response_time > self.response_time_thresholds['critical']:
                    score -= 40
                elif avg_response_time > self.response_time_thresholds['slow']:
                    score -= 20
                elif avg_response_time > self.response_time_thresholds['normal']:
                    score -= 10
            
            # Error rate penalty
            total_operations = sum(self.operation_counts.values())
            total_errors = sum(self.error_counts.values())
            if total_operations > 0:
                error_rate = (total_errors / total_operations) * 100
                if error_rate > 10:
                    score -= 30
                elif error_rate > 5:
                    score -= 15
                elif error_rate > 1:
                    score -= 5
            
            # Memory usage penalty
            if self.memory_usage:
                avg_memory = sum(self.memory_usage) / len(self.memory_usage)
                if avg_memory > 1000:  # > 1GB
                    score -= 15
                elif avg_memory > 500:  # > 500MB
                    score -= 10
            
            # CPU usage penalty
            if self.cpu_usage:
                avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage)
                if avg_cpu > 80:
                    score -= 15
                elif avg_cpu > 60:
                    score -= 10
            
            # Critical errors penalty
            critical_errors = len([e for e in self.error_records if e.severity == ErrorSeverity.CRITICAL])
            score -= critical_errors * 5
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 50.0  # Default neutral score
    
    def get_error_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Get detailed error analysis for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = [e for e in self.error_records 
                        if datetime.fromisoformat(e.timestamp) > cutoff_time]
        
        # Group errors by component and operation
        error_breakdown = defaultdict(lambda: defaultdict(int))
        severity_trends = defaultdict(int)
        
        for error in recent_errors:
            error_breakdown[error.component][error.operation] += 1
            severity_trends[error.severity.value] += 1
        
        return {
            "analysis_period_hours": hours,
            "total_recent_errors": len(recent_errors),
            "error_breakdown": dict(error_breakdown),
            "severity_trends": dict(severity_trends),
            "most_problematic_components": sorted(
                error_breakdown.items(), 
                key=lambda x: sum(x[1].values()), 
                reverse=True
            )[:5]
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Calculate trends for response times
        recent_metrics = []
        for key, metrics in self.metrics.items():
            for metric in metrics:
                if datetime.fromisoformat(metric.timestamp) > cutoff_time:
                    recent_metrics.append(metric)
        
        # Group by hour for trend analysis
        hourly_data = defaultdict(list)
        for metric in recent_metrics:
            hour = datetime.fromisoformat(metric.timestamp).replace(minute=0, second=0, microsecond=0)
            hourly_data[hour.isoformat()].append(metric.value)
        
        # Calculate hourly averages
        hourly_averages = {}
        for hour, values in hourly_data.items():
            hourly_averages[hour] = sum(values) / len(values) if values else 0
        
        return {
            "trend_period_hours": hours,
            "hourly_averages": hourly_averages,
            "total_data_points": len(recent_metrics)
        }
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self.is_logging:
            logger.info("Performance monitoring stopped")
    
    def reset_metrics(self):
        """Reset all metrics and counters."""
        with self._lock:
            self.metrics.clear()
            self.response_times.clear()
            self.error_counts.clear()
            self.operation_counts.clear()
            self.error_records.clear()
            self.memory_usage.clear()
            self.cpu_usage.clear()
            self.throughput_metrics.clear()
        
        if self.is_logging:
            logger.info("Performance metrics reset")

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(is_logging=True)
    return _performance_monitor