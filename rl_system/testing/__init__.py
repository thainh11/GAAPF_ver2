"""Testing Module for GAAPF RL System

This module provides comprehensive testing capabilities for the reinforcement
learning system, including A/B testing, performance comparison, and statistical
analysis.
"""

from .ab_testing import (
    ABTestConfig,
    ABTestResult,
    ABTestManager,
    TestGroup,
    StatisticalTest
)

from .metrics_collector import (
    MetricsConfig,
    MetricDefinition,
    MetricsCollector,
    MetricType,
    AggregationType
)

from .performance_comparison import (
    ComparisonConfig,
    ComparisonResult,
    PerformanceComparator,
    ComparisonMetric
)

__all__ = [
    # A/B Testing
    'ABTestConfig',
    'ABTestResult', 
    'ABTestManager',
    'TestGroup',
    'StatisticalTest',
    
    # Metrics Collection
    'MetricsConfig',
    'MetricDefinition',
    'MetricsCollector',
    'MetricType',
    'AggregationType',
    
    # Performance Comparison
    'ComparisonConfig',
    'ComparisonResult',
    'PerformanceComparator',
    'ComparisonMetric'
]