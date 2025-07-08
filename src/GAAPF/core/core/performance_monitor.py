"""
Performance monitoring module for the GAAPF framework.
"""

import logging
import time
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data class."""
    name: str
    value: float
    unit: str
    timestamp: str
    component: str

class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, is_logging: bool = False):
        self.is_logging = is_logging
        self.metrics = defaultdict(deque)
        self.response_times = defaultdict(deque)
        self.error_counts = defaultdict(int)
        self.operation_counts = defaultdict(int)
        
        if self.is_logging:
            logger.info("PerformanceMonitor initialized")
    
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
    
    def record_error(self, operation: str, error_message: str = None, component: str = "system") -> None:
        """Record an error occurrence."""
        key = f"{component}.{operation}"
        self.error_counts[key] += 1
        
        if self.is_logging:
            logger.warning(f"Error recorded for {key}: {error_message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        try:
            # Calculate average response times
            avg_response_times = {}
            for key, times in self.response_times.items():
                if times:
                    avg_response_times[key] = sum(times) / len(times)
            
            # Calculate error rates
            error_rates = {}
            for key in self.error_counts:
                operation_count = self.operation_counts.get(key, 0)
                error_count = self.error_counts[key]
                
                if operation_count > 0:
                    error_rates[key] = (error_count / operation_count) * 100
                else:
                    error_rates[key] = 0
            
            return {
                "summary_generated": datetime.now().isoformat(),
                "average_response_times": avg_response_times,
                "error_rates": error_rates,
                "operation_counts": dict(self.operation_counts),
                "total_operations": sum(self.operation_counts.values()),
                "total_errors": sum(self.error_counts.values())
            }
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return {"error": f"Failed to generate summary: {str(e)}"}
    
    def _add_metric(self, metric: PerformanceMetric) -> None:
        """Add a metric to the storage."""
        key = f"{metric.component}.{metric.name}"
        self.metrics[key].append(metric)
        
        # Limit history size
        if len(self.metrics[key]) > 1000:
            self.metrics[key].popleft()

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(is_logging=True)
    return _performance_monitor 