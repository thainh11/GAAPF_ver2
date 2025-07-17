"""A/B Testing Module for GAAPF RL System

This module provides comprehensive A/B testing capabilities to compare
the performance of RL-enhanced agents against traditional approaches.
"""

import numpy as np
import json
import os
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class TestGroup(Enum):
    """Test group types"""
    CONTROL = "control"  # Traditional GAAPF system
    TREATMENT = "treatment"  # RL-enhanced system
    HYBRID = "hybrid"  # Mixed approach

class StatisticalTest(Enum):
    """Statistical test types"""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    CHI_SQUARE = "chi_square"
    BOOTSTRAP = "bootstrap"
    BAYESIAN = "bayesian"

class TestStatus(Enum):
    """Test status"""
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""
    # Test identification
    test_name: str
    test_description: str
    
    # Test design
    control_group_size: int = 1000
    treatment_group_size: int = 1000
    hybrid_group_size: int = 500
    
    # Traffic allocation
    control_traffic_ratio: float = 0.4
    treatment_traffic_ratio: float = 0.4
    hybrid_traffic_ratio: float = 0.2
    
    # Test duration
    min_test_duration_hours: int = 24
    max_test_duration_hours: int = 168  # 1 week
    
    # Statistical parameters
    significance_level: float = 0.05
    power: float = 0.8
    minimum_detectable_effect: float = 0.05  # 5% improvement
    
    # Primary metrics
    primary_metrics: List[str] = None
    secondary_metrics: List[str] = None
    
    # Test conditions
    user_segments: List[str] = None
    scenario_types: List[str] = None
    
    # Quality controls
    max_sample_ratio_imbalance: float = 0.1
    outlier_detection_threshold: float = 3.0
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_threshold: float = 0.01
    min_samples_for_early_stopping: int = 100
    
    def __post_init__(self):
        if self.primary_metrics is None:
            self.primary_metrics = [
                'task_completion_rate',
                'user_satisfaction',
                'response_quality'
            ]
        
        if self.secondary_metrics is None:
            self.secondary_metrics = [
                'interaction_efficiency',
                'learning_effectiveness',
                'response_time',
                'error_rate'
            ]
        
        if self.user_segments is None:
            self.user_segments = ['beginner', 'intermediate', 'advanced']
        
        if self.scenario_types is None:
            self.scenario_types = [
                'coding_task',
                'concept_explanation',
                'problem_solving',
                'debugging'
            ]

@dataclass
class TestSample:
    """Individual test sample"""
    sample_id: str
    test_group: TestGroup
    timestamp: float
    
    # User context
    user_segment: str
    scenario_type: str
    scenario_difficulty: float
    
    # Metrics
    metrics: Dict[str, float]
    
    # Metadata
    session_duration: float
    interaction_count: int
    error_occurred: bool = False
    outlier: bool = False

@dataclass
class StatisticalResult:
    """Statistical test result"""
    test_type: StatisticalTest
    metric_name: str
    
    # Test statistics
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    
    # Interpretation
    is_significant: bool
    practical_significance: bool
    
    # Sample information
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    control_n: int
    treatment_n: int

@dataclass
class ABTestResult:
    """A/B test result"""
    test_id: str
    test_name: str
    status: TestStatus
    
    # Test timeline
    start_time: float
    end_time: Optional[float]
    duration_hours: float
    
    # Sample information
    total_samples: int
    control_samples: int
    treatment_samples: int
    hybrid_samples: int
    
    # Statistical results
    primary_results: Dict[str, StatisticalResult]
    secondary_results: Dict[str, StatisticalResult]
    
    # Overall conclusions
    overall_winner: Optional[TestGroup]
    confidence_level: float
    recommendation: str
    
    # Detailed analysis
    segment_analysis: Dict[str, Dict[str, StatisticalResult]]
    scenario_analysis: Dict[str, Dict[str, StatisticalResult]]
    
    # Quality metrics
    sample_ratio_balance: float
    outlier_rate: float
    data_quality_score: float

class ABTestManager:
    """Manages A/B testing for GAAPF RL system"""
    
    def __init__(self, config: ABTestConfig):
        """
        Initialize A/B test manager.
        
        Parameters:
        ----------
        config : ABTestConfig
            A/B test configuration
        """
        self.config = config
        
        # Test state
        self.test_id = str(uuid.uuid4())
        self.status = TestStatus.PLANNING
        self.start_time = None
        self.end_time = None
        
        # Sample collection
        self.samples = []
        self.samples_by_group = defaultdict(list)
        self.samples_lock = threading.Lock()
        
        # Statistical tracking
        self.statistical_results = {}
        self.early_stopping_checks = []
        
        # Quality monitoring
        self.outliers_detected = 0
        self.quality_issues = []
        
        # Threading for concurrent testing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized A/B test: {config.test_name} (ID: {self.test_id})")
    
    def start_test(self) -> str:
        """Start the A/B test"""
        if self.status != TestStatus.PLANNING:
            raise ValueError(f"Cannot start test in status: {self.status}")
        
        self.status = TestStatus.RUNNING
        self.start_time = time.time()
        
        logger.info(f"Started A/B test {self.config.test_name} at {datetime.fromtimestamp(self.start_time)}")
        
        return self.test_id
    
    def assign_test_group(self, user_context: Dict[str, Any]) -> TestGroup:
        """Assign user to test group based on traffic allocation"""
        if self.status != TestStatus.RUNNING:
            return TestGroup.CONTROL  # Default to control if test not running
        
        # Use deterministic assignment based on user ID if available
        user_id = user_context.get('user_id', str(uuid.uuid4()))
        
        # Hash user ID to get consistent assignment
        hash_value = hash(user_id + self.test_id) % 1000
        
        # Assign based on traffic ratios
        control_threshold = self.config.control_traffic_ratio * 1000
        treatment_threshold = control_threshold + (self.config.treatment_traffic_ratio * 1000)
        
        if hash_value < control_threshold:
            return TestGroup.CONTROL
        elif hash_value < treatment_threshold:
            return TestGroup.TREATMENT
        else:
            return TestGroup.HYBRID
    
    def record_sample(self, test_group: TestGroup, user_context: Dict[str, Any],
                     scenario_info: Dict[str, Any], metrics: Dict[str, float],
                     session_metadata: Dict[str, Any]) -> str:
        """Record a test sample"""
        if self.status != TestStatus.RUNNING:
            logger.warning(f"Attempted to record sample while test status is {self.status}")
            return None
        
        # Create sample
        sample = TestSample(
            sample_id=str(uuid.uuid4()),
            test_group=test_group,
            timestamp=time.time(),
            user_segment=user_context.get('experience_level', 'unknown'),
            scenario_type=scenario_info.get('type', 'unknown'),
            scenario_difficulty=scenario_info.get('difficulty', 0.5),
            metrics=metrics.copy(),
            session_duration=session_metadata.get('duration', 0.0),
            interaction_count=session_metadata.get('interaction_count', 1),
            error_occurred=session_metadata.get('error_occurred', False)
        )
        
        # Check for outliers
        sample.outlier = self._detect_outlier(sample)
        if sample.outlier:
            self.outliers_detected += 1
        
        # Store sample
        with self.samples_lock:
            self.samples.append(sample)
            self.samples_by_group[test_group].append(sample)
        
        # Check for early stopping if enabled
        if (self.config.enable_early_stopping and 
            len(self.samples) >= self.config.min_samples_for_early_stopping and
            len(self.samples) % 50 == 0):  # Check every 50 samples
            
            self._check_early_stopping()
        
        # Check test completion conditions
        self._check_completion_conditions()
        
        logger.debug(f"Recorded sample {sample.sample_id} for group {test_group.value}")
        
        return sample.sample_id
    
    def _detect_outlier(self, sample: TestSample) -> bool:
        """Detect if sample is an outlier"""
        # Simple outlier detection based on primary metrics
        for metric_name in self.config.primary_metrics:
            if metric_name in sample.metrics:
                value = sample.metrics[metric_name]
                
                # Get historical values for this metric and group
                historical_values = []
                for existing_sample in self.samples_by_group[sample.test_group]:
                    if metric_name in existing_sample.metrics and not existing_sample.outlier:
                        historical_values.append(existing_sample.metrics[metric_name])
                
                if len(historical_values) >= 10:  # Need minimum samples for outlier detection
                    mean_val = np.mean(historical_values)
                    std_val = np.std(historical_values)
                    
                    if std_val > 0:
                        z_score = abs(value - mean_val) / std_val
                        if z_score > self.config.outlier_detection_threshold:
                            return True
        
        return False
    
    def _check_early_stopping(self) -> None:
        """Check if test should be stopped early"""
        try:
            # Run statistical tests on primary metrics
            early_stop_results = []
            
            for metric_name in self.config.primary_metrics:
                result = self._run_statistical_test(metric_name, StatisticalTest.T_TEST)
                if result and result.is_significant:
                    early_stop_results.append(result)
            
            # Check if we have strong evidence for early stopping
            if len(early_stop_results) >= len(self.config.primary_metrics) * 0.7:  # 70% of primary metrics
                avg_p_value = np.mean([r.p_value for r in early_stop_results])
                
                if avg_p_value < self.config.early_stopping_threshold:
                    logger.info(f"Early stopping triggered. Average p-value: {avg_p_value:.6f}")
                    self._complete_test(early_stop=True)
                    return
            
            # Record early stopping check
            self.early_stopping_checks.append({
                'timestamp': time.time(),
                'sample_count': len(self.samples),
                'significant_metrics': len(early_stop_results),
                'avg_p_value': np.mean([r.p_value for r in early_stop_results]) if early_stop_results else 1.0
            })
            
        except Exception as e:
            logger.error(f"Error in early stopping check: {e}")
    
    def _check_completion_conditions(self) -> None:
        """Check if test should be completed"""
        if self.status != TestStatus.RUNNING:
            return
        
        current_time = time.time()
        duration_hours = (current_time - self.start_time) / 3600
        
        # Check minimum duration
        if duration_hours < self.config.min_test_duration_hours:
            return
        
        # Check maximum duration
        if duration_hours >= self.config.max_test_duration_hours:
            logger.info(f"Test reached maximum duration: {duration_hours:.1f} hours")
            self._complete_test()
            return
        
        # Check sample size requirements
        control_count = len(self.samples_by_group[TestGroup.CONTROL])
        treatment_count = len(self.samples_by_group[TestGroup.TREATMENT])
        
        if (control_count >= self.config.control_group_size and 
            treatment_count >= self.config.treatment_group_size):
            logger.info(f"Test reached target sample sizes: Control={control_count}, Treatment={treatment_count}")
            self._complete_test()
            return
    
    def _complete_test(self, early_stop: bool = False) -> None:
        """Complete the test and generate results"""
        if self.status != TestStatus.RUNNING:
            return
        
        self.status = TestStatus.COMPLETED
        self.end_time = time.time()
        
        logger.info(f"Completing A/B test {self.config.test_name} (early_stop={early_stop})")
        
        # Generate final results
        try:
            self._generate_final_results()
        except Exception as e:
            logger.error(f"Error generating final results: {e}")
            self.status = TestStatus.FAILED
    
    def _generate_final_results(self) -> ABTestResult:
        """Generate comprehensive test results"""
        # Run statistical tests for all metrics
        primary_results = {}
        for metric_name in self.config.primary_metrics:
            result = self._run_statistical_test(metric_name, StatisticalTest.T_TEST)
            if result:
                primary_results[metric_name] = result
        
        secondary_results = {}
        for metric_name in self.config.secondary_metrics:
            result = self._run_statistical_test(metric_name, StatisticalTest.T_TEST)
            if result:
                secondary_results[metric_name] = result
        
        # Segment analysis
        segment_analysis = self._analyze_by_segment()
        
        # Scenario analysis
        scenario_analysis = self._analyze_by_scenario()
        
        # Determine overall winner
        overall_winner = self._determine_overall_winner(primary_results)
        
        # Calculate quality metrics
        sample_ratio_balance = self._calculate_sample_ratio_balance()
        outlier_rate = self.outliers_detected / len(self.samples) if self.samples else 0.0
        data_quality_score = self._calculate_data_quality_score()
        
        # Generate recommendation
        recommendation = self._generate_recommendation(primary_results, overall_winner, data_quality_score)
        
        # Create result object
        result = ABTestResult(
            test_id=self.test_id,
            test_name=self.config.test_name,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_hours=(self.end_time - self.start_time) / 3600,
            total_samples=len(self.samples),
            control_samples=len(self.samples_by_group[TestGroup.CONTROL]),
            treatment_samples=len(self.samples_by_group[TestGroup.TREATMENT]),
            hybrid_samples=len(self.samples_by_group[TestGroup.HYBRID]),
            primary_results=primary_results,
            secondary_results=secondary_results,
            overall_winner=overall_winner,
            confidence_level=1.0 - self.config.significance_level,
            recommendation=recommendation,
            segment_analysis=segment_analysis,
            scenario_analysis=scenario_analysis,
            sample_ratio_balance=sample_ratio_balance,
            outlier_rate=outlier_rate,
            data_quality_score=data_quality_score
        )
        
        # Store results
        self.final_result = result
        
        # Save results to file
        self._save_results(result)
        
        return result
    
    def _run_statistical_test(self, metric_name: str, test_type: StatisticalTest) -> Optional[StatisticalResult]:
        """Run statistical test for a specific metric"""
        try:
            # Get metric values for control and treatment groups
            control_values = []
            treatment_values = []
            
            for sample in self.samples_by_group[TestGroup.CONTROL]:
                if metric_name in sample.metrics and not sample.outlier:
                    control_values.append(sample.metrics[metric_name])
            
            for sample in self.samples_by_group[TestGroup.TREATMENT]:
                if metric_name in sample.metrics and not sample.outlier:
                    treatment_values.append(sample.metrics[metric_name])
            
            if len(control_values) < 10 or len(treatment_values) < 10:
                logger.warning(f"Insufficient samples for {metric_name}: Control={len(control_values)}, Treatment={len(treatment_values)}")
                return None
            
            control_array = np.array(control_values)
            treatment_array = np.array(treatment_values)
            
            # Run appropriate statistical test
            if test_type == StatisticalTest.T_TEST:
                statistic, p_value = stats.ttest_ind(treatment_array, control_array)
            elif test_type == StatisticalTest.MANN_WHITNEY:
                statistic, p_value = stats.mannwhitneyu(treatment_array, control_array, alternative='two-sided')
            else:
                # Default to t-test
                statistic, p_value = stats.ttest_ind(treatment_array, control_array)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_array) - 1) * np.var(control_array, ddof=1) + 
                                 (len(treatment_array) - 1) * np.var(treatment_array, ddof=1)) / 
                                (len(control_array) + len(treatment_array) - 2))
            
            effect_size = (np.mean(treatment_array) - np.mean(control_array)) / pooled_std if pooled_std > 0 else 0.0
            
            # Calculate confidence interval for difference in means
            diff_mean = np.mean(treatment_array) - np.mean(control_array)
            se_diff = np.sqrt(np.var(treatment_array, ddof=1) / len(treatment_array) + 
                             np.var(control_array, ddof=1) / len(control_array))
            
            t_critical = stats.t.ppf(1 - self.config.significance_level / 2, 
                                   len(control_array) + len(treatment_array) - 2)
            
            ci_lower = diff_mean - t_critical * se_diff
            ci_upper = diff_mean + t_critical * se_diff
            
            # Determine significance
            is_significant = p_value < self.config.significance_level
            practical_significance = abs(effect_size) >= self.config.minimum_detectable_effect
            
            result = StatisticalResult(
                test_type=test_type,
                metric_name=metric_name,
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=float(effect_size),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                is_significant=is_significant,
                practical_significance=practical_significance,
                control_mean=float(np.mean(control_array)),
                treatment_mean=float(np.mean(treatment_array)),
                control_std=float(np.std(control_array, ddof=1)),
                treatment_std=float(np.std(treatment_array, ddof=1)),
                control_n=len(control_array),
                treatment_n=len(treatment_array)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error running statistical test for {metric_name}: {e}")
            return None
    
    def _analyze_by_segment(self) -> Dict[str, Dict[str, StatisticalResult]]:
        """Analyze results by user segment"""
        segment_analysis = {}
        
        for segment in self.config.user_segments:
            segment_samples = [s for s in self.samples if s.user_segment == segment]
            
            if len(segment_samples) < 20:  # Minimum samples for segment analysis
                continue
            
            segment_results = {}
            
            # Create temporary samples by group for this segment
            segment_by_group = defaultdict(list)
            for sample in segment_samples:
                segment_by_group[sample.test_group].append(sample)
            
            # Run tests for primary metrics
            for metric_name in self.config.primary_metrics:
                # Get values for this segment
                control_values = []
                treatment_values = []
                
                for sample in segment_by_group[TestGroup.CONTROL]:
                    if metric_name in sample.metrics and not sample.outlier:
                        control_values.append(sample.metrics[metric_name])
                
                for sample in segment_by_group[TestGroup.TREATMENT]:
                    if metric_name in sample.metrics and not sample.outlier:
                        treatment_values.append(sample.metrics[metric_name])
                
                if len(control_values) >= 5 and len(treatment_values) >= 5:
                    try:
                        statistic, p_value = stats.ttest_ind(treatment_values, control_values)
                        
                        # Simplified result for segment analysis
                        result = StatisticalResult(
                            test_type=StatisticalTest.T_TEST,
                            metric_name=metric_name,
                            statistic=float(statistic),
                            p_value=float(p_value),
                            effect_size=0.0,  # Simplified
                            confidence_interval=(0.0, 0.0),  # Simplified
                            is_significant=p_value < self.config.significance_level,
                            practical_significance=False,  # Simplified
                            control_mean=float(np.mean(control_values)),
                            treatment_mean=float(np.mean(treatment_values)),
                            control_std=float(np.std(control_values, ddof=1)),
                            treatment_std=float(np.std(treatment_values, ddof=1)),
                            control_n=len(control_values),
                            treatment_n=len(treatment_values)
                        )
                        
                        segment_results[metric_name] = result
                        
                    except Exception as e:
                        logger.warning(f"Error in segment analysis for {segment}/{metric_name}: {e}")
            
            if segment_results:
                segment_analysis[segment] = segment_results
        
        return segment_analysis
    
    def _analyze_by_scenario(self) -> Dict[str, Dict[str, StatisticalResult]]:
        """Analyze results by scenario type"""
        scenario_analysis = {}
        
        for scenario_type in self.config.scenario_types:
            scenario_samples = [s for s in self.samples if s.scenario_type == scenario_type]
            
            if len(scenario_samples) < 20:  # Minimum samples for scenario analysis
                continue
            
            scenario_results = {}
            
            # Create temporary samples by group for this scenario
            scenario_by_group = defaultdict(list)
            for sample in scenario_samples:
                scenario_by_group[sample.test_group].append(sample)
            
            # Run tests for primary metrics (simplified version)
            for metric_name in self.config.primary_metrics:
                control_values = []
                treatment_values = []
                
                for sample in scenario_by_group[TestGroup.CONTROL]:
                    if metric_name in sample.metrics and not sample.outlier:
                        control_values.append(sample.metrics[metric_name])
                
                for sample in scenario_by_group[TestGroup.TREATMENT]:
                    if metric_name in sample.metrics and not sample.outlier:
                        treatment_values.append(sample.metrics[metric_name])
                
                if len(control_values) >= 5 and len(treatment_values) >= 5:
                    try:
                        statistic, p_value = stats.ttest_ind(treatment_values, control_values)
                        
                        result = StatisticalResult(
                            test_type=StatisticalTest.T_TEST,
                            metric_name=metric_name,
                            statistic=float(statistic),
                            p_value=float(p_value),
                            effect_size=0.0,  # Simplified
                            confidence_interval=(0.0, 0.0),  # Simplified
                            is_significant=p_value < self.config.significance_level,
                            practical_significance=False,  # Simplified
                            control_mean=float(np.mean(control_values)),
                            treatment_mean=float(np.mean(treatment_values)),
                            control_std=float(np.std(control_values, ddof=1)),
                            treatment_std=float(np.std(treatment_values, ddof=1)),
                            control_n=len(control_values),
                            treatment_n=len(treatment_values)
                        )
                        
                        scenario_results[metric_name] = result
                        
                    except Exception as e:
                        logger.warning(f"Error in scenario analysis for {scenario_type}/{metric_name}: {e}")
            
            if scenario_results:
                scenario_analysis[scenario_type] = scenario_results
        
        return scenario_analysis
    
    def _determine_overall_winner(self, primary_results: Dict[str, StatisticalResult]) -> Optional[TestGroup]:
        """Determine overall winner based on primary metrics"""
        if not primary_results:
            return None
        
        # Count wins for treatment group
        treatment_wins = 0
        total_significant = 0
        
        for metric_name, result in primary_results.items():
            if result.is_significant:
                total_significant += 1
                if result.treatment_mean > result.control_mean:
                    treatment_wins += 1
        
        # Need majority of significant results to declare winner
        if total_significant == 0:
            return None
        
        win_rate = treatment_wins / total_significant
        
        if win_rate >= 0.7:  # 70% of significant metrics favor treatment
            return TestGroup.TREATMENT
        elif win_rate <= 0.3:  # 70% of significant metrics favor control
            return TestGroup.CONTROL
        else:
            return None  # No clear winner
    
    def _calculate_sample_ratio_balance(self) -> float:
        """Calculate how balanced the sample ratios are"""
        control_count = len(self.samples_by_group[TestGroup.CONTROL])
        treatment_count = len(self.samples_by_group[TestGroup.TREATMENT])
        
        if control_count == 0 or treatment_count == 0:
            return 0.0
        
        ratio = min(control_count, treatment_count) / max(control_count, treatment_count)
        return ratio
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score"""
        if not self.samples:
            return 0.0
        
        # Factors affecting data quality
        outlier_penalty = min(self.outliers_detected / len(self.samples), 0.2)  # Max 20% penalty
        balance_score = self._calculate_sample_ratio_balance()
        
        # Sample size adequacy
        min_required = min(self.config.control_group_size, self.config.treatment_group_size)
        actual_min = min(len(self.samples_by_group[TestGroup.CONTROL]), 
                        len(self.samples_by_group[TestGroup.TREATMENT]))
        size_adequacy = min(actual_min / min_required, 1.0) if min_required > 0 else 0.0
        
        # Error rate
        error_count = sum(1 for s in self.samples if s.error_occurred)
        error_penalty = min(error_count / len(self.samples), 0.1)  # Max 10% penalty
        
        # Calculate overall score
        quality_score = (balance_score * 0.3 + 
                        size_adequacy * 0.4 + 
                        (1.0 - outlier_penalty) * 0.2 + 
                        (1.0 - error_penalty) * 0.1)
        
        return max(quality_score, 0.0)
    
    def _generate_recommendation(self, primary_results: Dict[str, StatisticalResult],
                               overall_winner: Optional[TestGroup], 
                               data_quality_score: float) -> str:
        """Generate recommendation based on test results"""
        recommendations = []
        
        # Data quality check
        if data_quality_score < 0.7:
            recommendations.append(
                f"Data quality score is low ({data_quality_score:.2f}). "
                "Consider re-running the test with improved data collection."
            )
        
        # Overall winner assessment
        if overall_winner == TestGroup.TREATMENT:
            significant_improvements = []
            for metric_name, result in primary_results.items():
                if result.is_significant and result.treatment_mean > result.control_mean:
                    improvement = ((result.treatment_mean - result.control_mean) / result.control_mean) * 100
                    significant_improvements.append(f"{metric_name}: +{improvement:.1f}%")
            
            recommendations.append(
                f"Recommend adopting RL-enhanced system. Significant improvements in: {', '.join(significant_improvements)}"
            )
            
        elif overall_winner == TestGroup.CONTROL:
            recommendations.append(
                "Traditional system performs better. Consider investigating RL system issues "
                "before deployment."
            )
            
        else:
            recommendations.append(
                "No clear winner detected. Consider extending test duration or "
                "investigating specific use cases where each system excels."
            )
        
        # Sample size recommendations
        control_count = len(self.samples_by_group[TestGroup.CONTROL])
        treatment_count = len(self.samples_by_group[TestGroup.TREATMENT])
        
        if control_count < self.config.control_group_size or treatment_count < self.config.treatment_group_size:
            recommendations.append(
                f"Sample sizes below target (Control: {control_count}/{self.config.control_group_size}, "
                f"Treatment: {treatment_count}/{self.config.treatment_group_size}). "
                "Consider extending test duration."
            )
        
        return " ".join(recommendations)
    
    def _save_results(self, result: ABTestResult) -> None:
        """Save test results to file"""
        try:
            # Create results directory
            results_dir = "ab_test_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save detailed results
            result_file = os.path.join(results_dir, f"{result.test_id}.json")
            
            # Convert result to serializable format
            result_dict = asdict(result)
            
            # Convert enum values to strings
            def convert_enums(obj):
                if isinstance(obj, dict):
                    return {k: convert_enums(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_enums(item) for item in obj]
                elif hasattr(obj, 'value'):  # Enum
                    return obj.value
                else:
                    return obj
            
            result_dict = convert_enums(result_dict)
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            # Save summary
            summary_file = os.path.join(results_dir, f"{result.test_id}_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"A/B Test Results Summary\n")
                f.write(f"========================\n\n")
                f.write(f"Test Name: {result.test_name}\n")
                f.write(f"Test ID: {result.test_id}\n")
                f.write(f"Duration: {result.duration_hours:.1f} hours\n")
                f.write(f"Total Samples: {result.total_samples}\n")
                f.write(f"Overall Winner: {result.overall_winner.value if result.overall_winner else 'No clear winner'}\n")
                f.write(f"Data Quality Score: {result.data_quality_score:.2f}\n\n")
                
                f.write(f"Primary Metrics Results:\n")
                f.write(f"------------------------\n")
                for metric_name, stat_result in result.primary_results.items():
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Control Mean: {stat_result.control_mean:.3f}\n")
                    f.write(f"  Treatment Mean: {stat_result.treatment_mean:.3f}\n")
                    f.write(f"  P-value: {stat_result.p_value:.6f}\n")
                    f.write(f"  Significant: {stat_result.is_significant}\n")
                    f.write(f"  Effect Size: {stat_result.effect_size:.3f}\n\n")
                
                f.write(f"Recommendation:\n")
                f.write(f"---------------\n")
                f.write(f"{result.recommendation}\n")
            
            logger.info(f"Saved A/B test results to {result_file}")
            
        except Exception as e:
            logger.error(f"Error saving A/B test results: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current test status"""
        status_info = {
            'test_id': self.test_id,
            'test_name': self.config.test_name,
            'status': self.status.value,
            'start_time': self.start_time,
            'duration_hours': (time.time() - self.start_time) / 3600 if self.start_time else 0,
            'total_samples': len(self.samples),
            'samples_by_group': {
                'control': len(self.samples_by_group[TestGroup.CONTROL]),
                'treatment': len(self.samples_by_group[TestGroup.TREATMENT]),
                'hybrid': len(self.samples_by_group[TestGroup.HYBRID])
            },
            'outliers_detected': self.outliers_detected,
            'early_stopping_checks': len(self.early_stopping_checks)
        }
        
        return status_info
    
    def pause_test(self) -> None:
        """Pause the test"""
        if self.status == TestStatus.RUNNING:
            self.status = TestStatus.PAUSED
            logger.info(f"Paused A/B test {self.config.test_name}")
    
    def resume_test(self) -> None:
        """Resume the test"""
        if self.status == TestStatus.PAUSED:
            self.status = TestStatus.RUNNING
            logger.info(f"Resumed A/B test {self.config.test_name}")
    
    def stop_test(self) -> ABTestResult:
        """Manually stop the test and generate results"""
        if self.status in [TestStatus.RUNNING, TestStatus.PAUSED]:
            logger.info(f"Manually stopping A/B test {self.config.test_name}")
            self._complete_test()
            return self.final_result
        else:
            raise ValueError(f"Cannot stop test in status: {self.status}")
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info(f"A/B test manager cleanup completed for {self.config.test_name}")