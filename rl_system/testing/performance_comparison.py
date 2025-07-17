"""Performance Comparison Module for GAAPF RL System

This module provides comprehensive performance comparison capabilities
between RL-enhanced agents and traditional approaches.
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
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

logger = logging.getLogger(__name__)

class ComparisonType(Enum):
    """Types of performance comparisons"""
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    USER_SATISFACTION = "user_satisfaction"
    LEARNING_EFFECTIVENESS = "learning_effectiveness"
    RESPONSE_QUALITY = "response_quality"
    COLLABORATION = "collaboration"
    ADAPTABILITY = "adaptability"
    ROBUSTNESS = "robustness"

class SystemType(Enum):
    """Types of systems being compared"""
    TRADITIONAL = "traditional"
    RL_ENHANCED = "rl_enhanced"
    HYBRID = "hybrid"
    BASELINE = "baseline"

class MetricImportance(Enum):
    """Importance levels for metrics"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ComparisonMetric:
    """Definition of a comparison metric"""
    name: str
    description: str
    comparison_type: ComparisonType
    importance: MetricImportance
    
    # Value properties
    higher_is_better: bool = True
    unit: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Statistical properties
    expected_distribution: str = "normal"  # normal, exponential, uniform
    significance_threshold: float = 0.05
    effect_size_threshold: float = 0.2  # Cohen's d
    
    # Weighting
    weight: float = 1.0
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceResult:
    """Individual performance measurement result"""
    system_type: SystemType
    metric_name: str
    value: float
    timestamp: float
    
    # Context
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    scenario: Optional[str] = None
    
    # Quality indicators
    confidence: float = 1.0
    sample_size: int = 1
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComparisonResult:
    """Result of comparing two systems"""
    metric_name: str
    system_a: SystemType
    system_b: SystemType
    
    # Statistical results
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    
    # Comparison statistics
    difference: float  # mean_b - mean_a
    percent_change: float  # (mean_b - mean_a) / mean_a * 100
    effect_size: float  # Cohen's d
    
    # Statistical significance
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    
    # Sample information
    sample_size_a: int
    sample_size_b: int
    
    # Interpretation
    winner: Optional[SystemType] = None
    improvement_category: str = ""  # negligible, small, medium, large
    
    # Additional analysis
    distribution_test_p: Optional[float] = None
    variance_test_p: Optional[float] = None
    
@dataclass
class ComparisonConfig:
    """Configuration for performance comparison"""
    # Statistical settings
    significance_level: float = 0.05
    minimum_sample_size: int = 30
    bootstrap_iterations: int = 1000
    
    # Effect size thresholds (Cohen's d)
    small_effect_size: float = 0.2
    medium_effect_size: float = 0.5
    large_effect_size: float = 0.8
    
    # Comparison settings
    enable_multiple_comparisons_correction: bool = True
    correction_method: str = "bonferroni"  # bonferroni, holm, fdr_bh
    
    # Visualization settings
    generate_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # Export settings
    export_detailed_results: bool = True
    export_summary_only: bool = False
    
class PerformanceComparator:
    """Comprehensive performance comparison system"""
    
    def __init__(self, config: ComparisonConfig):
        """
        Initialize performance comparator.
        
        Parameters:
        ----------
        config : ComparisonConfig
            Comparison configuration
        """
        self.config = config
        
        # Metric definitions
        self.comparison_metrics = {}
        
        # Performance data storage
        self.performance_data = defaultdict(list)
        
        # Comparison results cache
        self.comparison_cache = {}
        self.cache_lock = threading.Lock()
        
        # Register default metrics
        self._register_default_metrics()
        
        logger.info("Initialized performance comparator")
    
    def _register_default_metrics(self) -> None:
        """Register default comparison metrics"""
        default_metrics = [
            ComparisonMetric(
                name="task_completion_rate",
                description="Rate of successful task completions",
                comparison_type=ComparisonType.ACCURACY,
                importance=MetricImportance.CRITICAL,
                unit="%",
                min_value=0.0,
                max_value=100.0,
                weight=2.0
            ),
            
            ComparisonMetric(
                name="response_time",
                description="Average response time",
                comparison_type=ComparisonType.EFFICIENCY,
                importance=MetricImportance.HIGH,
                higher_is_better=False,
                unit="seconds",
                min_value=0.0,
                expected_distribution="exponential",
                weight=1.5
            ),
            
            ComparisonMetric(
                name="user_satisfaction",
                description="User satisfaction score",
                comparison_type=ComparisonType.USER_SATISFACTION,
                importance=MetricImportance.CRITICAL,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                weight=2.0
            ),
            
            ComparisonMetric(
                name="response_quality",
                description="Quality of generated responses",
                comparison_type=ComparisonType.RESPONSE_QUALITY,
                importance=MetricImportance.HIGH,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                weight=1.8
            ),
            
            ComparisonMetric(
                name="learning_effectiveness",
                description="Effectiveness of learning interactions",
                comparison_type=ComparisonType.LEARNING_EFFECTIVENESS,
                importance=MetricImportance.HIGH,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                weight=1.7
            ),
            
            ComparisonMetric(
                name="collaboration_score",
                description="Multi-agent collaboration effectiveness",
                comparison_type=ComparisonType.COLLABORATION,
                importance=MetricImportance.MEDIUM,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                weight=1.3
            ),
            
            ComparisonMetric(
                name="adaptability_score",
                description="System adaptability to new scenarios",
                comparison_type=ComparisonType.ADAPTABILITY,
                importance=MetricImportance.MEDIUM,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                weight=1.2
            ),
            
            ComparisonMetric(
                name="error_rate",
                description="Rate of errors in system operations",
                comparison_type=ComparisonType.ROBUSTNESS,
                importance=MetricImportance.HIGH,
                higher_is_better=False,
                unit="%",
                min_value=0.0,
                max_value=100.0,
                weight=1.6
            ),
            
            ComparisonMetric(
                name="memory_efficiency",
                description="Memory usage efficiency",
                comparison_type=ComparisonType.EFFICIENCY,
                importance=MetricImportance.MEDIUM,
                higher_is_better=False,
                unit="MB",
                min_value=0.0,
                weight=1.0
            ),
            
            ComparisonMetric(
                name="throughput",
                description="Number of tasks processed per unit time",
                comparison_type=ComparisonType.EFFICIENCY,
                importance=MetricImportance.HIGH,
                unit="tasks/hour",
                min_value=0.0,
                weight=1.4
            )
        ]
        
        for metric in default_metrics:
            self.register_metric(metric)
    
    def register_metric(self, metric: ComparisonMetric) -> None:
        """Register a comparison metric"""
        self.comparison_metrics[metric.name] = metric
        logger.debug(f"Registered comparison metric: {metric.name}")
    
    def record_performance(self, system_type: SystemType, 
                          metric_name: str, 
                          value: float,
                          session_id: Optional[str] = None,
                          user_id: Optional[str] = None,
                          task_id: Optional[str] = None,
                          scenario: Optional[str] = None,
                          confidence: float = 1.0,
                          sample_size: int = 1,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a performance measurement"""
        if metric_name not in self.comparison_metrics:
            logger.warning(f"Unknown comparison metric: {metric_name}")
            return
        
        result = PerformanceResult(
            system_type=system_type,
            metric_name=metric_name,
            value=value,
            timestamp=time.time(),
            session_id=session_id,
            user_id=user_id,
            task_id=task_id,
            scenario=scenario,
            confidence=confidence,
            sample_size=sample_size,
            metadata=metadata or {}
        )
        
        key = f"{system_type.value}_{metric_name}"
        self.performance_data[key].append(result)
        
        # Clear cache for this metric
        with self.cache_lock:
            cache_keys_to_remove = [
                k for k in self.comparison_cache.keys() 
                if metric_name in k
            ]
            for k in cache_keys_to_remove:
                del self.comparison_cache[k]
    
    def compare_systems(self, system_a: SystemType, 
                       system_b: SystemType,
                       metric_name: str,
                       scenario_filter: Optional[str] = None,
                       time_range: Optional[Tuple[float, float]] = None) -> Optional[ComparisonResult]:
        """Compare two systems on a specific metric"""
        # Check cache
        cache_key = f"{system_a.value}_{system_b.value}_{metric_name}_{scenario_filter}_{time_range}"
        with self.cache_lock:
            if cache_key in self.comparison_cache:
                return self.comparison_cache[cache_key]
        
        # Get data for both systems
        data_a = self._get_filtered_data(system_a, metric_name, scenario_filter, time_range)
        data_b = self._get_filtered_data(system_b, metric_name, scenario_filter, time_range)
        
        if len(data_a) < self.config.minimum_sample_size or len(data_b) < self.config.minimum_sample_size:
            logger.warning(f"Insufficient data for comparison: {len(data_a)} vs {len(data_b)} samples")
            return None
        
        # Extract values
        values_a = [d.value for d in data_a]
        values_b = [d.value for d in data_b]
        
        # Calculate basic statistics
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        std_a = np.std(values_a, ddof=1)
        std_b = np.std(values_b, ddof=1)
        
        # Calculate difference and effect size
        difference = mean_b - mean_a
        percent_change = (difference / mean_a * 100) if mean_a != 0 else 0
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) / 
                            (len(values_a) + len(values_b) - 2))
        effect_size = difference / pooled_std if pooled_std != 0 else 0
        
        # Statistical significance test
        statistic, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
        
        # Confidence interval for difference
        se_diff = np.sqrt(std_a**2/len(values_a) + std_b**2/len(values_b))
        df = len(values_a) + len(values_b) - 2
        t_critical = stats.t.ppf(1 - self.config.significance_level/2, df)
        ci_lower = difference - t_critical * se_diff
        ci_upper = difference + t_critical * se_diff
        
        # Determine winner
        metric_def = self.comparison_metrics[metric_name]
        is_significant = p_value < self.config.significance_level
        
        winner = None
        if is_significant:
            if metric_def.higher_is_better:
                winner = system_b if mean_b > mean_a else system_a
            else:
                winner = system_b if mean_b < mean_a else system_a
        
        # Categorize improvement
        abs_effect_size = abs(effect_size)
        if abs_effect_size < self.config.small_effect_size:
            improvement_category = "negligible"
        elif abs_effect_size < self.config.medium_effect_size:
            improvement_category = "small"
        elif abs_effect_size < self.config.large_effect_size:
            improvement_category = "medium"
        else:
            improvement_category = "large"
        
        # Additional statistical tests
        distribution_test_p = None
        variance_test_p = None
        
        try:
            # Test for normality (Shapiro-Wilk)
            if len(values_a) <= 5000 and len(values_b) <= 5000:
                _, p_norm_a = stats.shapiro(values_a)
                _, p_norm_b = stats.shapiro(values_b)
                distribution_test_p = min(p_norm_a, p_norm_b)
            
            # Test for equal variances (Levene's test)
            _, variance_test_p = stats.levene(values_a, values_b)
        
        except Exception as e:
            logger.warning(f"Error in additional statistical tests: {e}")
        
        # Create result
        result = ComparisonResult(
            metric_name=metric_name,
            system_a=system_a,
            system_b=system_b,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            difference=difference,
            percent_change=percent_change,
            effect_size=effect_size,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=(ci_lower, ci_upper),
            sample_size_a=len(values_a),
            sample_size_b=len(values_b),
            winner=winner,
            improvement_category=improvement_category,
            distribution_test_p=distribution_test_p,
            variance_test_p=variance_test_p
        )
        
        # Cache result
        with self.cache_lock:
            self.comparison_cache[cache_key] = result
        
        return result
    
    def _get_filtered_data(self, system_type: SystemType, 
                          metric_name: str,
                          scenario_filter: Optional[str] = None,
                          time_range: Optional[Tuple[float, float]] = None) -> List[PerformanceResult]:
        """Get filtered performance data"""
        key = f"{system_type.value}_{metric_name}"
        data = self.performance_data.get(key, [])
        
        filtered_data = []
        for result in data:
            # Apply scenario filter
            if scenario_filter and result.scenario != scenario_filter:
                continue
            
            # Apply time range filter
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= result.timestamp <= end_time):
                    continue
            
            filtered_data.append(result)
        
        return filtered_data
    
    def compare_all_metrics(self, system_a: SystemType, 
                           system_b: SystemType,
                           scenario_filter: Optional[str] = None,
                           time_range: Optional[Tuple[float, float]] = None) -> Dict[str, ComparisonResult]:
        """Compare two systems across all registered metrics"""
        results = {}
        
        for metric_name in self.comparison_metrics.keys():
            result = self.compare_systems(
                system_a, system_b, metric_name, 
                scenario_filter, time_range
            )
            if result:
                results[metric_name] = result
        
        # Apply multiple comparisons correction if enabled
        if self.config.enable_multiple_comparisons_correction and results:
            results = self._apply_multiple_comparisons_correction(results)
        
        return results
    
    def _apply_multiple_comparisons_correction(self, results: Dict[str, ComparisonResult]) -> Dict[str, ComparisonResult]:
        """Apply multiple comparisons correction"""
        p_values = [result.p_value for result in results.values()]
        
        if self.config.correction_method == "bonferroni":
            corrected_alpha = self.config.significance_level / len(p_values)
            for result in results.values():
                result.is_significant = result.p_value < corrected_alpha
        
        elif self.config.correction_method == "holm":
            # Holm-Bonferroni correction
            sorted_items = sorted(results.items(), key=lambda x: x[1].p_value)
            
            for i, (metric_name, result) in enumerate(sorted_items):
                corrected_alpha = self.config.significance_level / (len(p_values) - i)
                result.is_significant = result.p_value < corrected_alpha
                
                # If this test is not significant, all subsequent tests are not significant
                if not result.is_significant:
                    for j in range(i + 1, len(sorted_items)):
                        sorted_items[j][1].is_significant = False
                    break
        
        elif self.config.correction_method == "fdr_bh":
            # Benjamini-Hochberg FDR correction
            sorted_items = sorted(results.items(), key=lambda x: x[1].p_value)
            
            for i, (metric_name, result) in enumerate(sorted_items):
                corrected_alpha = (i + 1) / len(p_values) * self.config.significance_level
                result.is_significant = result.p_value <= corrected_alpha
        
        return results
    
    def calculate_overall_score(self, system_type: SystemType,
                               scenario_filter: Optional[str] = None,
                               time_range: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
        """Calculate overall performance score for a system"""
        scores = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, metric_def in self.comparison_metrics.items():
            data = self._get_filtered_data(system_type, metric_name, scenario_filter, time_range)
            
            if not data:
                continue
            
            values = [d.value for d in data]
            mean_value = np.mean(values)
            
            # Normalize score (0-1 scale)
            if metric_def.min_value is not None and metric_def.max_value is not None:
                normalized_score = (mean_value - metric_def.min_value) / (metric_def.max_value - metric_def.min_value)
            else:
                # Use z-score normalization
                all_values = []
                for sys_type in SystemType:
                    sys_data = self._get_filtered_data(sys_type, metric_name, scenario_filter, time_range)
                    all_values.extend([d.value for d in sys_data])
                
                if all_values:
                    mean_all = np.mean(all_values)
                    std_all = np.std(all_values)
                    normalized_score = (mean_value - mean_all) / std_all if std_all > 0 else 0
                    normalized_score = (normalized_score + 3) / 6  # Map to 0-1 range (assuming ±3σ)
                else:
                    normalized_score = 0.5
            
            # Invert if lower is better
            if not metric_def.higher_is_better:
                normalized_score = 1 - normalized_score
            
            # Ensure score is in [0, 1] range
            normalized_score = max(0, min(1, normalized_score))
            
            scores[metric_name] = normalized_score
            
            # Add to weighted total
            weight = metric_def.weight
            if metric_def.importance == MetricImportance.CRITICAL:
                weight *= 2.0
            elif metric_def.importance == MetricImportance.HIGH:
                weight *= 1.5
            elif metric_def.importance == MetricImportance.LOW:
                weight *= 0.5
            
            total_weighted_score += normalized_score * weight
            total_weight += weight
        
        # Calculate overall score
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0
        scores['overall'] = overall_score
        
        return scores
    
    def generate_comparison_report(self, system_a: SystemType, 
                                  system_b: SystemType,
                                  scenario_filter: Optional[str] = None,
                                  time_range: Optional[Tuple[float, float]] = None,
                                  output_path: Optional[str] = None) -> str:
        """Generate comprehensive comparison report"""
        # Get all comparisons
        comparisons = self.compare_all_metrics(system_a, system_b, scenario_filter, time_range)
        
        # Calculate overall scores
        scores_a = self.calculate_overall_score(system_a, scenario_filter, time_range)
        scores_b = self.calculate_overall_score(system_b, scenario_filter, time_range)
        
        # Generate report
        report = {
            'metadata': {
                'system_a': system_a.value,
                'system_b': system_b.value,
                'scenario_filter': scenario_filter,
                'time_range': time_range,
                'generated_at': datetime.now().isoformat(),
                'total_metrics': len(comparisons)
            },
            'overall_scores': {
                system_a.value: scores_a,
                system_b.value: scores_b
            },
            'metric_comparisons': {},
            'summary': {
                'significant_improvements': [],
                'significant_degradations': [],
                'no_significant_difference': [],
                'winner': None,
                'confidence': 0.0
            }
        }
        
        # Process individual metric comparisons
        significant_improvements = 0
        significant_degradations = 0
        total_effect_size = 0.0
        
        for metric_name, comparison in comparisons.items():
            metric_def = self.comparison_metrics[metric_name]
            
            report['metric_comparisons'][metric_name] = {
                'comparison_result': asdict(comparison),
                'metric_definition': asdict(metric_def)
            }
            
            # Categorize results
            if comparison.is_significant:
                if comparison.winner == system_b:
                    significant_improvements += 1
                    report['summary']['significant_improvements'].append({
                        'metric': metric_name,
                        'improvement': comparison.percent_change,
                        'effect_size': comparison.effect_size
                    })
                elif comparison.winner == system_a:
                    significant_degradations += 1
                    report['summary']['significant_degradations'].append({
                        'metric': metric_name,
                        'degradation': comparison.percent_change,
                        'effect_size': comparison.effect_size
                    })
            else:
                report['summary']['no_significant_difference'].append(metric_name)
            
            total_effect_size += abs(comparison.effect_size) * metric_def.weight
        
        # Determine overall winner
        overall_score_a = scores_a.get('overall', 0)
        overall_score_b = scores_b.get('overall', 0)
        
        if overall_score_b > overall_score_a:
            report['summary']['winner'] = system_b.value
            report['summary']['confidence'] = (overall_score_b - overall_score_a) / max(overall_score_a, overall_score_b)
        elif overall_score_a > overall_score_b:
            report['summary']['winner'] = system_a.value
            report['summary']['confidence'] = (overall_score_a - overall_score_b) / max(overall_score_a, overall_score_b)
        else:
            report['summary']['winner'] = "tie"
            report['summary']['confidence'] = 0.0
        
        # Add statistical summary
        report['summary']['total_significant_differences'] = significant_improvements + significant_degradations
        report['summary']['improvement_ratio'] = significant_improvements / len(comparisons) if comparisons else 0
        report['summary']['average_effect_size'] = total_effect_size / len(comparisons) if comparisons else 0
        
        # Save report
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"comparison_report_{system_a.value}_vs_{system_b.value}_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Generated comparison report: {output_path}")
        return output_path
    
    def visualize_comparison(self, system_a: SystemType, 
                           system_b: SystemType,
                           metric_names: Optional[List[str]] = None,
                           scenario_filter: Optional[str] = None,
                           time_range: Optional[Tuple[float, float]] = None,
                           output_dir: Optional[str] = None) -> List[str]:
        """Generate visualization plots for comparison"""
        if not self.config.generate_plots:
            return []
        
        if metric_names is None:
            metric_names = list(self.comparison_metrics.keys())
        
        if output_dir is None:
            output_dir = "comparison_plots"
        
        os.makedirs(output_dir, exist_ok=True)
        
        plot_files = []
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        for metric_name in metric_names:
            try:
                # Get data
                data_a = self._get_filtered_data(system_a, metric_name, scenario_filter, time_range)
                data_b = self._get_filtered_data(system_b, metric_name, scenario_filter, time_range)
                
                if not data_a or not data_b:
                    continue
                
                values_a = [d.value for d in data_a]
                values_b = [d.value for d in data_b]
                
                # Create comparison plot
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'Performance Comparison: {metric_name}', fontsize=16)
                
                # Box plot
                axes[0, 0].boxplot([values_a, values_b], labels=[system_a.value, system_b.value])
                axes[0, 0].set_title('Distribution Comparison')
                axes[0, 0].set_ylabel(self.comparison_metrics[metric_name].unit)
                
                # Histogram
                axes[0, 1].hist(values_a, alpha=0.7, label=system_a.value, bins=20)
                axes[0, 1].hist(values_b, alpha=0.7, label=system_b.value, bins=20)
                axes[0, 1].set_title('Value Distribution')
                axes[0, 1].set_xlabel(self.comparison_metrics[metric_name].unit)
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
                
                # Time series (if enough data points)
                if len(data_a) > 5 and len(data_b) > 5:
                    times_a = [d.timestamp for d in data_a]
                    times_b = [d.timestamp for d in data_b]
                    
                    axes[1, 0].scatter(times_a, values_a, alpha=0.6, label=system_a.value)
                    axes[1, 0].scatter(times_b, values_b, alpha=0.6, label=system_b.value)
                    axes[1, 0].set_title('Time Series')
                    axes[1, 0].set_xlabel('Time')
                    axes[1, 0].set_ylabel(self.comparison_metrics[metric_name].unit)
                    axes[1, 0].legend()
                
                # Statistical summary
                comparison = self.compare_systems(system_a, system_b, metric_name, scenario_filter, time_range)
                if comparison:
                    summary_text = f"""Mean A: {comparison.mean_a:.3f}
Mean B: {comparison.mean_b:.3f}
Difference: {comparison.difference:.3f}
Effect Size: {comparison.effect_size:.3f}
P-value: {comparison.p_value:.3f}
Significant: {comparison.is_significant}"""
                    
                    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                                   fontsize=10, verticalalignment='center',
                                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    axes[1, 1].set_title('Statistical Summary')
                    axes[1, 1].axis('off')
                
                plt.tight_layout()
                
                # Save plot
                plot_file = os.path.join(output_dir, f"{metric_name}_comparison.{self.config.plot_format}")
                plt.savefig(plot_file, dpi=self.config.plot_dpi, bbox_inches='tight')
                plt.close()
                
                plot_files.append(plot_file)
                
            except Exception as e:
                logger.error(f"Error creating plot for {metric_name}: {e}")
        
        # Create overall comparison plot
        try:
            scores_a = self.calculate_overall_score(system_a, scenario_filter, time_range)
            scores_b = self.calculate_overall_score(system_b, scenario_filter, time_range)
            
            # Remove overall score for individual metric plotting
            metric_scores_a = {k: v for k, v in scores_a.items() if k != 'overall'}
            metric_scores_b = {k: v for k, v in scores_b.items() if k != 'overall'}
            
            if metric_scores_a and metric_scores_b:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                metrics = list(metric_scores_a.keys())
                x = np.arange(len(metrics))
                width = 0.35
                
                scores_a_vals = [metric_scores_a[m] for m in metrics]
                scores_b_vals = [metric_scores_b[m] for m in metrics]
                
                ax.bar(x - width/2, scores_a_vals, width, label=system_a.value, alpha=0.8)
                ax.bar(x + width/2, scores_b_vals, width, label=system_b.value, alpha=0.8)
                
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Normalized Score')
                ax.set_title('Overall Performance Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                overall_plot_file = os.path.join(output_dir, f"overall_comparison.{self.config.plot_format}")
                plt.savefig(overall_plot_file, dpi=self.config.plot_dpi, bbox_inches='tight')
                plt.close()
                
                plot_files.append(overall_plot_file)
        
        except Exception as e:
            logger.error(f"Error creating overall comparison plot: {e}")
        
        logger.info(f"Generated {len(plot_files)} comparison plots in {output_dir}")
        return plot_files
    
    def export_data(self, output_path: Optional[str] = None,
                   format_type: str = "json") -> str:
        """Export all performance data"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"performance_data_{timestamp}.{format_type}"
        
        # Prepare data for export
        export_data = {
            'metrics': {name: asdict(metric) for name, metric in self.comparison_metrics.items()},
            'performance_data': {}
        }
        
        for key, results in self.performance_data.items():
            export_data['performance_data'][key] = [asdict(result) for result in results]
        
        # Export in specified format
        if format_type == "json":
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format_type == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(export_data, f)
        
        elif format_type == "csv":
            # Flatten data for CSV
            rows = []
            for key, results in self.performance_data.items():
                for result in results:
                    row = asdict(result)
                    row['data_key'] = key
                    rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        logger.info(f"Exported performance data to {output_path}")
        return output_path
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'registered_metrics': len(self.comparison_metrics),
            'total_data_points': sum(len(data) for data in self.performance_data.values()),
            'systems_with_data': list(set(
                key.split('_')[0] for key in self.performance_data.keys()
            )),
            'cached_comparisons': len(self.comparison_cache),
            'config': asdict(self.config)
        }
        
        # Add per-system statistics
        system_stats = {}
        for system_type in SystemType:
            system_data_points = sum(
                len(data) for key, data in self.performance_data.items()
                if key.startswith(system_type.value)
            )
            system_stats[system_type.value] = system_data_points
        
        stats['data_points_by_system'] = system_stats
        
        return stats
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        # Clear caches
        with self.cache_lock:
            self.comparison_cache.clear()
        
        # Clear data if needed
        # self.performance_data.clear()
        
        logger.info("Performance comparator cleanup completed")