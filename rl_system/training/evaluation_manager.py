"""Evaluation Manager for GAAPF RL System

This module manages evaluation and testing of the reinforcement learning
components, providing comprehensive performance assessment and analysis.
"""

import numpy as np
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class EvaluationType(Enum):
    """Types of evaluation"""
    PERFORMANCE = "performance"
    GENERALIZATION = "generalization"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    COLLABORATION = "collaboration"
    USER_SATISFACTION = "user_satisfaction"
    RETENTION = "retention"
    TRANSFER_LEARNING = "transfer_learning"

class MetricType(Enum):
    """Types of metrics"""
    SCALAR = "scalar"
    DISTRIBUTION = "distribution"
    TIME_SERIES = "time_series"
    CATEGORICAL = "categorical"
    CORRELATION = "correlation"

@dataclass
class EvaluationConfig:
    """Configuration for evaluation manager"""
    # Evaluation frequency
    evaluation_frequency: int = 100
    detailed_evaluation_frequency: int = 500
    comprehensive_evaluation_frequency: int = 1000
    
    # Test sets
    test_set_size: int = 1000
    validation_set_size: int = 500
    holdout_set_size: int = 200
    
    # Performance thresholds
    performance_threshold: float = 0.8
    improvement_threshold: float = 0.05
    degradation_threshold: float = 0.1
    
    # Statistical testing
    confidence_level: float = 0.95
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # Evaluation scenarios
    scenario_variety: int = 10
    difficulty_levels: List[float] = None
    user_profiles: List[str] = None
    
    # Resource limits
    max_evaluation_time: float = 3600  # 1 hour
    parallel_evaluations: int = 4
    memory_limit_gb: float = 4.0
    
    # Output settings
    save_detailed_results: bool = True
    generate_plots: bool = True
    export_metrics: bool = True
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        if self.user_profiles is None:
            self.user_profiles = ['beginner', 'intermediate', 'advanced']

@dataclass
class EvaluationMetric:
    """Individual evaluation metric"""
    name: str
    value: Union[float, List[float], Dict[str, float]]
    metric_type: MetricType
    timestamp: float
    
    # Statistical properties
    confidence_interval: Optional[Tuple[float, float]] = None
    standard_error: Optional[float] = None
    sample_size: Optional[int] = None
    
    # Metadata
    description: str = ""
    unit: str = ""
    higher_is_better: bool = True
    
@dataclass
class EvaluationResult:
    """Results from an evaluation session"""
    evaluation_id: str
    evaluation_type: EvaluationType
    timestamp: float
    duration: float
    
    # Metrics
    metrics: Dict[str, EvaluationMetric]
    
    # Summary statistics
    overall_score: float
    performance_grade: str
    
    # Detailed results
    scenario_results: Dict[str, Dict[str, float]]
    agent_results: Dict[str, Dict[str, float]]
    
    # Comparisons
    baseline_comparison: Optional[Dict[str, float]] = None
    previous_comparison: Optional[Dict[str, float]] = None
    
    # Recommendations
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

class EvaluationManager:
    """Manages evaluation and testing of GAAPF RL system"""
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluation manager.
        
        Parameters:
        ----------
        config : EvaluationConfig
            Evaluation configuration
        """
        self.config = config
        
        # Evaluation state
        self.evaluation_count = 0
        self.evaluation_history = []
        self.baseline_results = None
        
        # Test scenarios
        self.test_scenarios = []
        self.validation_scenarios = []
        self.holdout_scenarios = []
        
        # Performance tracking
        self.performance_trends = defaultdict(list)
        self.metric_history = defaultdict(list)
        
        # Statistical tracking
        self.significance_tests = []
        self.correlation_matrix = {}
        
        # Threading for parallel evaluation
        self.executor = ThreadPoolExecutor(max_workers=config.parallel_evaluations)
        self.evaluation_lock = threading.Lock()
        
        # Initialize test scenarios
        self._initialize_test_scenarios()
        
        logger.info(f"Initialized evaluation manager with {len(self.test_scenarios)} test scenarios")
    
    def _initialize_test_scenarios(self) -> None:
        """Initialize test scenarios for evaluation"""
        # Generate diverse test scenarios
        scenario_types = [
            'basic_concept_explanation',
            'coding_task_simple',
            'coding_task_complex',
            'problem_solving',
            'project_guidance',
            'debugging_assistance',
            'code_review',
            'algorithm_explanation',
            'best_practices',
            'troubleshooting'
        ]
        
        # Create test scenarios
        for i in range(self.config.test_set_size):
            scenario = self._generate_test_scenario(i, scenario_types)
            self.test_scenarios.append(scenario)
        
        # Create validation scenarios
        for i in range(self.config.validation_set_size):
            scenario = self._generate_test_scenario(i + self.config.test_set_size, scenario_types)
            self.validation_scenarios.append(scenario)
        
        # Create holdout scenarios (for final evaluation)
        for i in range(self.config.holdout_set_size):
            scenario = self._generate_test_scenario(
                i + self.config.test_set_size + self.config.validation_set_size, 
                scenario_types
            )
            self.holdout_scenarios.append(scenario)
        
        logger.info(f"Generated {len(self.test_scenarios)} test scenarios, "
                   f"{len(self.validation_scenarios)} validation scenarios, "
                   f"{len(self.holdout_scenarios)} holdout scenarios")
    
    def _generate_test_scenario(self, scenario_id: int, scenario_types: List[str]) -> Dict[str, Any]:
        """Generate a test scenario"""
        scenario_type = np.random.choice(scenario_types)
        difficulty = np.random.choice(self.config.difficulty_levels)
        user_profile = np.random.choice(self.config.user_profiles)
        
        # Generate scenario content based on type and difficulty
        scenario_content = self._generate_scenario_content(scenario_type, difficulty, user_profile)
        
        scenario = {
            'scenario_id': f"test_{scenario_id}",
            'type': scenario_type,
            'difficulty': difficulty,
            'user_profile': user_profile,
            'content': scenario_content,
            'expected_outcomes': self._generate_expected_outcomes(scenario_type, difficulty),
            'evaluation_criteria': self._generate_evaluation_criteria(scenario_type),
            'metadata': {
                'created_at': time.time(),
                'complexity_factors': self._calculate_complexity_factors(scenario_type, difficulty),
                'required_capabilities': self._get_required_capabilities(scenario_type)
            }
        }
        
        return scenario
    
    def _generate_scenario_content(self, scenario_type: str, difficulty: float, user_profile: str) -> Dict[str, Any]:
        """Generate content for a test scenario"""
        content_templates = {
            'basic_concept_explanation': {
                'task_description': f"Explain the concept of {{concept}} to a {user_profile} learner",
                'context': "Educational explanation request",
                'expected_interaction_type': "explanation"
            },
            'coding_task_simple': {
                'task_description': f"Help a {user_profile} write a simple {{language}} program that {{task}}",
                'context': "Coding assistance request",
                'expected_interaction_type': "coding_guidance"
            },
            'coding_task_complex': {
                'task_description': f"Assist a {user_profile} with implementing {{algorithm}} in {{language}}",
                'context': "Advanced coding task",
                'expected_interaction_type': "complex_coding"
            },
            'problem_solving': {
                'task_description': f"Help a {user_profile} solve this {{domain}} problem: {{problem}}",
                'context': "Problem-solving assistance",
                'expected_interaction_type': "analytical_guidance"
            },
            'project_guidance': {
                'task_description': f"Guide a {user_profile} through building a {{project_type}} project",
                'context': "Project development assistance",
                'expected_interaction_type': "project_management"
            },
            'debugging_assistance': {
                'task_description': f"Help a {user_profile} debug this {{language}} code with {{error_type}}",
                'context': "Debugging assistance request",
                'expected_interaction_type': "debugging"
            },
            'code_review': {
                'task_description': f"Review this {{language}} code for a {user_profile} and suggest improvements",
                'context': "Code review request",
                'expected_interaction_type': "code_analysis"
            },
            'algorithm_explanation': {
                'task_description': f"Explain how {{algorithm}} works to a {user_profile}",
                'context': "Algorithm learning request",
                'expected_interaction_type': "technical_explanation"
            },
            'best_practices': {
                'task_description': f"Teach a {user_profile} about {{topic}} best practices",
                'context': "Best practices guidance",
                'expected_interaction_type': "advisory"
            },
            'troubleshooting': {
                'task_description': f"Help a {user_profile} troubleshoot {{issue_type}} in their {{context}}",
                'context': "Technical troubleshooting",
                'expected_interaction_type': "problem_diagnosis"
            }
        }
        
        template = content_templates.get(scenario_type, content_templates['basic_concept_explanation'])
        
        # Fill template placeholders based on difficulty
        placeholders = self._get_scenario_placeholders(difficulty, user_profile)
        
        content = {}
        for key, value in template.items():
            try:
                content[key] = value.format(**placeholders)
            except KeyError:
                content[key] = value
        
        # Add difficulty-specific parameters
        content['complexity_level'] = difficulty
        content['estimated_duration'] = self._estimate_scenario_duration(scenario_type, difficulty)
        content['required_expertise'] = self._map_difficulty_to_expertise(difficulty)
        
        return content
    
    def _get_scenario_placeholders(self, difficulty: float, user_profile: str) -> Dict[str, str]:
        """Get placeholders for scenario content generation"""
        if difficulty < 0.3:
            concepts = ['variables', 'loops', 'functions', 'conditionals']
            languages = ['Python', 'JavaScript']
            tasks = ['print hello world', 'calculate sum', 'find maximum']
            algorithms = ['linear search', 'bubble sort']
            domains = ['basic math', 'simple logic']
            problems = ['calculate average', 'count items', 'find duplicates']
        elif difficulty < 0.7:
            concepts = ['classes', 'inheritance', 'data structures', 'algorithms']
            languages = ['Python', 'JavaScript', 'Java', 'C++']
            tasks = ['implement data structure', 'create API', 'build web app']
            algorithms = ['binary search', 'merge sort', 'dynamic programming']
            domains = ['software design', 'data analysis']
            problems = ['optimize performance', 'handle edge cases', 'scale system']
        else:
            concepts = ['design patterns', 'system architecture', 'optimization']
            languages = ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust']
            tasks = ['design distributed system', 'implement ML pipeline', 'optimize algorithm']
            algorithms = ['graph algorithms', 'machine learning', 'distributed consensus']
            domains = ['system design', 'machine learning', 'distributed systems']
            problems = ['design scalable architecture', 'implement fault tolerance', 'optimize for performance']
        
        return {
            'concept': np.random.choice(concepts),
            'language': np.random.choice(languages),
            'task': np.random.choice(tasks),
            'algorithm': np.random.choice(algorithms),
            'domain': np.random.choice(domains),
            'problem': np.random.choice(problems),
            'project_type': np.random.choice(['web application', 'mobile app', 'CLI tool', 'library']),
            'error_type': np.random.choice(['syntax error', 'logic error', 'runtime error']),
            'topic': np.random.choice(['coding standards', 'security', 'performance', 'testing']),
            'issue_type': np.random.choice(['performance issue', 'compatibility problem', 'configuration error']),
            'context': np.random.choice(['development environment', 'production system', 'testing framework'])
        }
    
    def _generate_expected_outcomes(self, scenario_type: str, difficulty: float) -> Dict[str, float]:
        """Generate expected outcomes for a scenario"""
        base_expectations = {
            'task_completion_rate': 0.8,
            'user_satisfaction': 0.7,
            'response_quality': 0.75,
            'interaction_efficiency': 0.7,
            'learning_effectiveness': 0.65
        }
        
        # Adjust expectations based on difficulty
        difficulty_factor = 1.0 - (difficulty * 0.3)  # Harder tasks have lower expected success
        
        adjusted_expectations = {}
        for metric, base_value in base_expectations.items():
            adjusted_value = base_value * difficulty_factor
            adjusted_expectations[metric] = max(adjusted_value, 0.3)  # Minimum threshold
        
        # Scenario-specific adjustments
        if scenario_type in ['coding_task_complex', 'algorithm_explanation']:
            adjusted_expectations['response_quality'] *= 1.1
            adjusted_expectations['learning_effectiveness'] *= 1.1
        elif scenario_type in ['debugging_assistance', 'troubleshooting']:
            adjusted_expectations['interaction_efficiency'] *= 1.2
        
        return adjusted_expectations
    
    def _generate_evaluation_criteria(self, scenario_type: str) -> List[str]:
        """Generate evaluation criteria for a scenario"""
        common_criteria = [
            'response_relevance',
            'response_accuracy',
            'response_completeness',
            'interaction_quality',
            'user_engagement'
        ]
        
        scenario_specific_criteria = {
            'basic_concept_explanation': ['explanation_clarity', 'example_quality', 'pedagogical_effectiveness'],
            'coding_task_simple': ['code_correctness', 'code_style', 'guidance_quality'],
            'coding_task_complex': ['solution_efficiency', 'code_architecture', 'best_practices'],
            'problem_solving': ['analytical_approach', 'solution_creativity', 'step_by_step_guidance'],
            'project_guidance': ['project_structure', 'milestone_planning', 'resource_recommendations'],
            'debugging_assistance': ['error_identification', 'fix_effectiveness', 'prevention_advice'],
            'code_review': ['issue_identification', 'improvement_suggestions', 'code_quality_assessment'],
            'algorithm_explanation': ['conceptual_clarity', 'complexity_analysis', 'implementation_guidance'],
            'best_practices': ['practice_relevance', 'implementation_guidance', 'real_world_examples'],
            'troubleshooting': ['problem_diagnosis', 'solution_effectiveness', 'prevention_strategies']
        }
        
        specific_criteria = scenario_specific_criteria.get(scenario_type, [])
        return common_criteria + specific_criteria
    
    def _calculate_complexity_factors(self, scenario_type: str, difficulty: float) -> Dict[str, float]:
        """Calculate complexity factors for a scenario"""
        base_factors = {
            'cognitive_load': difficulty * 0.8,
            'technical_depth': difficulty * 0.9,
            'interaction_complexity': difficulty * 0.7,
            'domain_specificity': difficulty * 0.6
        }
        
        # Scenario-specific adjustments
        if scenario_type in ['coding_task_complex', 'algorithm_explanation']:
            base_factors['technical_depth'] *= 1.3
        elif scenario_type in ['project_guidance', 'best_practices']:
            base_factors['interaction_complexity'] *= 1.2
        elif scenario_type in ['debugging_assistance', 'troubleshooting']:
            base_factors['cognitive_load'] *= 1.1
        
        return base_factors
    
    def _get_required_capabilities(self, scenario_type: str) -> List[str]:
        """Get required capabilities for a scenario"""
        capability_mapping = {
            'basic_concept_explanation': ['explanation', 'teaching', 'simplification'],
            'coding_task_simple': ['coding', 'guidance', 'syntax_help'],
            'coding_task_complex': ['advanced_coding', 'architecture', 'optimization'],
            'problem_solving': ['analysis', 'problem_decomposition', 'solution_design'],
            'project_guidance': ['project_management', 'planning', 'resource_allocation'],
            'debugging_assistance': ['debugging', 'error_analysis', 'troubleshooting'],
            'code_review': ['code_analysis', 'quality_assessment', 'improvement_suggestions'],
            'algorithm_explanation': ['algorithm_knowledge', 'complexity_analysis', 'teaching'],
            'best_practices': ['domain_expertise', 'standards_knowledge', 'advisory'],
            'troubleshooting': ['problem_diagnosis', 'systematic_analysis', 'solution_implementation']
        }
        
        return capability_mapping.get(scenario_type, ['general'])
    
    def _estimate_scenario_duration(self, scenario_type: str, difficulty: float) -> float:
        """Estimate duration for a scenario in seconds"""
        base_durations = {
            'basic_concept_explanation': 300,  # 5 minutes
            'coding_task_simple': 600,        # 10 minutes
            'coding_task_complex': 1800,      # 30 minutes
            'problem_solving': 900,           # 15 minutes
            'project_guidance': 1200,         # 20 minutes
            'debugging_assistance': 600,      # 10 minutes
            'code_review': 900,               # 15 minutes
            'algorithm_explanation': 600,     # 10 minutes
            'best_practices': 450,            # 7.5 minutes
            'troubleshooting': 750            # 12.5 minutes
        }
        
        base_duration = base_durations.get(scenario_type, 600)
        difficulty_multiplier = 1.0 + (difficulty * 1.5)
        
        return base_duration * difficulty_multiplier
    
    def _map_difficulty_to_expertise(self, difficulty: float) -> str:
        """Map difficulty level to required expertise"""
        if difficulty < 0.2:
            return 'basic'
        elif difficulty < 0.4:
            return 'intermediate'
        elif difficulty < 0.7:
            return 'advanced'
        else:
            return 'expert'
    
    def evaluate_system(self, training_manager, constellation_manager, agents: Dict[str, Any],
                       evaluation_type: EvaluationType = EvaluationType.PERFORMANCE) -> EvaluationResult:
        """Evaluate the RL system comprehensively"""
        evaluation_start_time = time.time()
        evaluation_id = f"eval_{int(evaluation_start_time)}_{evaluation_type.value}"
        
        logger.info(f"Starting {evaluation_type.value} evaluation: {evaluation_id}")
        
        # Select scenarios based on evaluation type
        scenarios = self._select_evaluation_scenarios(evaluation_type)
        
        # Run evaluation
        scenario_results = self._run_scenario_evaluation(scenarios, constellation_manager, agents)
        
        # Calculate metrics
        metrics = self._calculate_evaluation_metrics(scenario_results, evaluation_type)
        
        # Analyze agent performance
        agent_results = self._analyze_agent_performance(scenario_results, agents)
        
        # Calculate overall score and grade
        overall_score = self._calculate_overall_score(metrics)
        performance_grade = self._assign_performance_grade(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, agent_results, evaluation_type)
        
        # Compare with baseline and previous results
        baseline_comparison = self._compare_with_baseline(metrics)
        previous_comparison = self._compare_with_previous(metrics)
        
        # Create evaluation result
        evaluation_result = EvaluationResult(
            evaluation_id=evaluation_id,
            evaluation_type=evaluation_type,
            timestamp=evaluation_start_time,
            duration=time.time() - evaluation_start_time,
            metrics=metrics,
            overall_score=overall_score,
            performance_grade=performance_grade,
            scenario_results=scenario_results,
            agent_results=agent_results,
            baseline_comparison=baseline_comparison,
            previous_comparison=previous_comparison,
            recommendations=recommendations
        )
        
        # Store results
        self.evaluation_history.append(evaluation_result)
        self.evaluation_count += 1
        
        # Update performance trends
        self._update_performance_trends(evaluation_result)
        
        # Save detailed results if configured
        if self.config.save_detailed_results:
            self._save_evaluation_results(evaluation_result)
        
        # Generate plots if configured
        if self.config.generate_plots:
            self._generate_evaluation_plots(evaluation_result)
        
        logger.info(f"Completed evaluation {evaluation_id} in {evaluation_result.duration:.2f}s. "
                   f"Overall score: {overall_score:.3f}, Grade: {performance_grade}")
        
        return evaluation_result
    
    def _select_evaluation_scenarios(self, evaluation_type: EvaluationType) -> List[Dict[str, Any]]:
        """Select scenarios for evaluation based on type"""
        if evaluation_type == EvaluationType.PERFORMANCE:
            # Use validation scenarios for performance evaluation
            return self.validation_scenarios[:100]  # Subset for faster evaluation
        elif evaluation_type == EvaluationType.GENERALIZATION:
            # Use test scenarios for generalization
            return self.test_scenarios[:200]
        elif evaluation_type == EvaluationType.ROBUSTNESS:
            # Use challenging scenarios
            challenging_scenarios = [s for s in self.test_scenarios if s['difficulty'] > 0.7]
            return challenging_scenarios[:100]
        elif evaluation_type == EvaluationType.EFFICIENCY:
            # Use time-sensitive scenarios
            return self.validation_scenarios[:50]  # Smaller set for efficiency testing
        else:
            # Default to validation scenarios
            return self.validation_scenarios[:100]
    
    def _run_scenario_evaluation(self, scenarios: List[Dict[str, Any]], 
                               constellation_manager, agents: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Run evaluation on selected scenarios"""
        scenario_results = {}
        
        for i, scenario in enumerate(scenarios):
            if i % 10 == 0:
                logger.debug(f"Evaluating scenario {i+1}/{len(scenarios)}")
            
            try:
                # Simulate scenario execution
                result = self._execute_evaluation_scenario(scenario, constellation_manager, agents)
                scenario_results[scenario['scenario_id']] = result
            except Exception as e:
                logger.error(f"Error evaluating scenario {scenario['scenario_id']}: {e}")
                # Use default poor performance for failed scenarios
                scenario_results[scenario['scenario_id']] = {
                    'task_completion_rate': 0.0,
                    'user_satisfaction': 0.0,
                    'response_quality': 0.0,
                    'interaction_efficiency': 0.0,
                    'learning_effectiveness': 0.0,
                    'error': str(e)
                }
        
        return scenario_results
    
    def _execute_evaluation_scenario(self, scenario: Dict[str, Any], 
                                   constellation_manager, agents: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single evaluation scenario"""
        scenario_start_time = time.time()
        
        # Extract scenario information
        task_description = scenario['content']['task_description']
        user_context = {
            'user_profile': {
                'experience_level': scenario['user_profile'],
                'learning_style': np.random.choice(['visual', 'auditory', 'kinesthetic']),
                'pace_preference': np.random.choice(['slow', 'medium', 'fast'])
            },
            'engagement_score': np.random.uniform(0.5, 1.0),
            'interaction_count': np.random.randint(1, 10),
            'session_duration': scenario['content']['estimated_duration']
        }
        
        required_capabilities = scenario['metadata']['required_capabilities']
        
        # Form constellation
        selected_agents, formation_metadata = constellation_manager.form_constellation(
            task_description=task_description,
            user_context=user_context,
            required_capabilities=required_capabilities,
            max_agents=3
        )
        
        # Simulate interaction execution
        interaction_result = self._simulate_interaction_execution(
            selected_agents, scenario, user_context
        )
        
        # Calculate performance metrics
        performance_metrics = self._calculate_scenario_performance(
            interaction_result, scenario, time.time() - scenario_start_time
        )
        
        return performance_metrics
    
    def _simulate_interaction_execution(self, selected_agents: List[str], 
                                      scenario: Dict[str, Any],
                                      user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the execution of an interaction"""
        # Base performance factors
        difficulty = scenario['difficulty']
        complexity_factors = scenario['metadata']['complexity_factors']
        
        # Agent capability matching
        required_capabilities = scenario['metadata']['required_capabilities']
        agent_capability_score = self._calculate_agent_capability_match(
            selected_agents, required_capabilities
        )
        
        # User profile matching
        user_profile = scenario['user_profile']
        user_match_score = self._calculate_user_profile_match(user_profile, difficulty)
        
        # Calculate base performance
        base_performance = (agent_capability_score + user_match_score) / 2
        
        # Apply difficulty penalty
        difficulty_penalty = difficulty * 0.3
        adjusted_performance = base_performance * (1.0 - difficulty_penalty)
        
        # Add some randomness
        noise = np.random.normal(0, 0.1)
        final_performance = np.clip(adjusted_performance + noise, 0.0, 1.0)
        
        # Generate specific metrics
        interaction_result = {
            'task_completion': final_performance,
            'user_satisfaction': final_performance * np.random.uniform(0.8, 1.2),
            'response_quality': final_performance * np.random.uniform(0.9, 1.1),
            'interaction_efficiency': final_performance * np.random.uniform(0.7, 1.3),
            'learning_effectiveness': final_performance * np.random.uniform(0.8, 1.2),
            'collaboration_score': len(selected_agents) / 3.0 * final_performance,
            'agent_utilization': np.random.uniform(0.6, 1.0),
            'response_time': np.random.uniform(1.0, 10.0),
            'error_rate': (1.0 - final_performance) * np.random.uniform(0.5, 1.5)
        }
        
        # Clip all values to valid ranges
        for key, value in interaction_result.items():
            if key != 'response_time':
                interaction_result[key] = np.clip(value, 0.0, 1.0)
            else:
                interaction_result[key] = max(value, 0.1)
        
        return interaction_result
    
    def _calculate_agent_capability_match(self, selected_agents: List[str], 
                                        required_capabilities: List[str]) -> float:
        """Calculate how well selected agents match required capabilities"""
        # Simplified capability matching
        # In a real implementation, this would check actual agent capabilities
        
        if not selected_agents or not required_capabilities:
            return 0.5  # Neutral score
        
        # Assume each agent covers some capabilities
        coverage_score = min(len(selected_agents) / len(required_capabilities), 1.0)
        
        # Add some randomness based on agent "quality"
        quality_factor = np.random.uniform(0.7, 1.0)
        
        return coverage_score * quality_factor
    
    def _calculate_user_profile_match(self, user_profile: str, difficulty: float) -> float:
        """Calculate how well the difficulty matches the user profile"""
        profile_difficulty_mapping = {
            'beginner': 0.2,
            'intermediate': 0.5,
            'advanced': 0.8
        }
        
        target_difficulty = profile_difficulty_mapping.get(user_profile, 0.5)
        difficulty_match = 1.0 - abs(difficulty - target_difficulty)
        
        return max(difficulty_match, 0.3)  # Minimum match score
    
    def _calculate_scenario_performance(self, interaction_result: Dict[str, Any],
                                      scenario: Dict[str, Any], duration: float) -> Dict[str, float]:
        """Calculate performance metrics for a scenario"""
        expected_outcomes = scenario['expected_outcomes']
        
        performance_metrics = {}
        
        # Direct metrics from interaction result
        performance_metrics['task_completion_rate'] = interaction_result['task_completion']
        performance_metrics['user_satisfaction'] = interaction_result['user_satisfaction']
        performance_metrics['response_quality'] = interaction_result['response_quality']
        performance_metrics['interaction_efficiency'] = interaction_result['interaction_efficiency']
        performance_metrics['learning_effectiveness'] = interaction_result['learning_effectiveness']
        
        # Derived metrics
        performance_metrics['collaboration_effectiveness'] = interaction_result['collaboration_score']
        performance_metrics['resource_utilization'] = interaction_result['agent_utilization']
        performance_metrics['response_time_score'] = min(10.0 / interaction_result['response_time'], 1.0)
        performance_metrics['error_rate'] = interaction_result['error_rate']
        
        # Performance vs expectations
        expectation_scores = []
        for metric, expected_value in expected_outcomes.items():
            if metric in performance_metrics:
                actual_value = performance_metrics[metric]
                expectation_score = min(actual_value / expected_value, 1.0) if expected_value > 0 else 1.0
                expectation_scores.append(expectation_score)
        
        performance_metrics['expectation_fulfillment'] = np.mean(expectation_scores) if expectation_scores else 0.5
        
        # Overall scenario score
        key_metrics = ['task_completion_rate', 'user_satisfaction', 'response_quality']
        scenario_score = np.mean([performance_metrics[metric] for metric in key_metrics])
        performance_metrics['scenario_score'] = scenario_score
        
        return performance_metrics
    
    def _calculate_evaluation_metrics(self, scenario_results: Dict[str, Dict[str, float]],
                                    evaluation_type: EvaluationType) -> Dict[str, EvaluationMetric]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Extract all metric values
        metric_values = defaultdict(list)
        for scenario_id, results in scenario_results.items():
            for metric_name, value in results.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_values[metric_name].append(value)
        
        # Calculate statistics for each metric
        for metric_name, values in metric_values.items():
            if not values:
                continue
            
            values_array = np.array(values)
            
            # Basic statistics
            mean_value = np.mean(values_array)
            std_value = np.std(values_array)
            median_value = np.median(values_array)
            
            # Confidence interval
            confidence_interval = self._calculate_confidence_interval(values_array)
            
            # Create metric
            metric = EvaluationMetric(
                name=metric_name,
                value=mean_value,
                metric_type=MetricType.SCALAR,
                timestamp=time.time(),
                confidence_interval=confidence_interval,
                standard_error=std_value / np.sqrt(len(values)),
                sample_size=len(values),
                description=f"Average {metric_name} across evaluation scenarios",
                unit="score" if "score" in metric_name or "rate" in metric_name else "value"
            )
            
            metrics[metric_name] = metric
        
        # Add distribution metrics
        key_metrics = ['task_completion_rate', 'user_satisfaction', 'response_quality']
        for metric_name in key_metrics:
            if metric_name in metric_values:
                values = metric_values[metric_name]
                distribution_metric = EvaluationMetric(
                    name=f"{metric_name}_distribution",
                    value=values,
                    metric_type=MetricType.DISTRIBUTION,
                    timestamp=time.time(),
                    sample_size=len(values),
                    description=f"Distribution of {metric_name} values"
                )
                metrics[f"{metric_name}_distribution"] = distribution_metric
        
        # Add evaluation-type specific metrics
        if evaluation_type == EvaluationType.ROBUSTNESS:
            metrics.update(self._calculate_robustness_metrics(scenario_results))
        elif evaluation_type == EvaluationType.EFFICIENCY:
            metrics.update(self._calculate_efficiency_metrics(scenario_results))
        elif evaluation_type == EvaluationType.COLLABORATION:
            metrics.update(self._calculate_collaboration_metrics(scenario_results))
        
        return metrics
    
    def _calculate_confidence_interval(self, values: np.ndarray, confidence: float = None) -> Tuple[float, float]:
        """Calculate confidence interval for values"""
        if confidence is None:
            confidence = self.config.confidence_level
        
        if len(values) < 2:
            return (float(values[0]), float(values[0])) if len(values) == 1 else (0.0, 0.0)
        
        # Use bootstrap for confidence interval
        bootstrap_means = []
        for _ in range(self.config.bootstrap_samples):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return (float(ci_lower), float(ci_upper))
    
    def _calculate_robustness_metrics(self, scenario_results: Dict[str, Dict[str, float]]) -> Dict[str, EvaluationMetric]:
        """Calculate robustness-specific metrics"""
        metrics = {}
        
        # Performance variance across difficulty levels
        difficulty_performance = defaultdict(list)
        for scenario_id, results in scenario_results.items():
            # Would need to track difficulty per scenario
            # For now, use simplified approach
            performance = results.get('scenario_score', 0.0)
            difficulty_performance['overall'].append(performance)
        
        if difficulty_performance['overall']:
            variance = np.var(difficulty_performance['overall'])
            metrics['performance_variance'] = EvaluationMetric(
                name='performance_variance',
                value=variance,
                metric_type=MetricType.SCALAR,
                timestamp=time.time(),
                description="Variance in performance across scenarios",
                higher_is_better=False
            )
        
        return metrics
    
    def _calculate_efficiency_metrics(self, scenario_results: Dict[str, Dict[str, float]]) -> Dict[str, EvaluationMetric]:
        """Calculate efficiency-specific metrics"""
        metrics = {}
        
        # Response time analysis
        response_times = []
        for results in scenario_results.values():
            if 'response_time_score' in results:
                response_times.append(results['response_time_score'])
        
        if response_times:
            avg_response_time = np.mean(response_times)
            metrics['average_response_efficiency'] = EvaluationMetric(
                name='average_response_efficiency',
                value=avg_response_time,
                metric_type=MetricType.SCALAR,
                timestamp=time.time(),
                description="Average response time efficiency",
                unit="efficiency_score"
            )
        
        return metrics
    
    def _calculate_collaboration_metrics(self, scenario_results: Dict[str, Dict[str, float]]) -> Dict[str, EvaluationMetric]:
        """Calculate collaboration-specific metrics"""
        metrics = {}
        
        # Collaboration effectiveness
        collaboration_scores = []
        for results in scenario_results.values():
            if 'collaboration_effectiveness' in results:
                collaboration_scores.append(results['collaboration_effectiveness'])
        
        if collaboration_scores:
            avg_collaboration = np.mean(collaboration_scores)
            metrics['average_collaboration_effectiveness'] = EvaluationMetric(
                name='average_collaboration_effectiveness',
                value=avg_collaboration,
                metric_type=MetricType.SCALAR,
                timestamp=time.time(),
                description="Average collaboration effectiveness",
                unit="effectiveness_score"
            )
        
        return metrics
    
    def _analyze_agent_performance(self, scenario_results: Dict[str, Dict[str, float]],
                                 agents: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze individual agent performance"""
        agent_results = {}
        
        # For each agent, calculate aggregate performance
        for agent_id in agents.keys():
            agent_metrics = {
                'participation_rate': np.random.uniform(0.6, 1.0),  # Simplified
                'average_contribution': np.random.uniform(0.5, 0.9),
                'collaboration_score': np.random.uniform(0.6, 0.95),
                'specialization_effectiveness': np.random.uniform(0.7, 1.0),
                'learning_progress': np.random.uniform(0.0, 0.3)  # Learning rate
            }
            
            agent_results[agent_id] = agent_metrics
        
        return agent_results
    
    def _calculate_overall_score(self, metrics: Dict[str, EvaluationMetric]) -> float:
        """Calculate overall evaluation score"""
        key_metrics = [
            'task_completion_rate',
            'user_satisfaction', 
            'response_quality',
            'interaction_efficiency',
            'learning_effectiveness'
        ]
        
        scores = []
        weights = [0.25, 0.2, 0.2, 0.15, 0.2]  # Weights for key metrics
        
        for i, metric_name in enumerate(key_metrics):
            if metric_name in metrics:
                score = metrics[metric_name].value
                weighted_score = score * weights[i]
                scores.append(weighted_score)
        
        return sum(scores) if scores else 0.0
    
    def _assign_performance_grade(self, overall_score: float) -> str:
        """Assign performance grade based on overall score"""
        if overall_score >= 0.9:
            return 'A+'
        elif overall_score >= 0.85:
            return 'A'
        elif overall_score >= 0.8:
            return 'A-'
        elif overall_score >= 0.75:
            return 'B+'
        elif overall_score >= 0.7:
            return 'B'
        elif overall_score >= 0.65:
            return 'B-'
        elif overall_score >= 0.6:
            return 'C+'
        elif overall_score >= 0.55:
            return 'C'
        elif overall_score >= 0.5:
            return 'C-'
        elif overall_score >= 0.4:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, metrics: Dict[str, EvaluationMetric],
                               agent_results: Dict[str, Dict[str, float]],
                               evaluation_type: EvaluationType) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Performance-based recommendations
        if 'task_completion_rate' in metrics:
            completion_rate = metrics['task_completion_rate'].value
            if completion_rate < 0.7:
                recommendations.append(
                    "Task completion rate is below threshold. Consider improving agent training "
                    "or adjusting task difficulty progression."
                )
        
        if 'user_satisfaction' in metrics:
            satisfaction = metrics['user_satisfaction'].value
            if satisfaction < 0.75:
                recommendations.append(
                    "User satisfaction is low. Focus on improving response quality and "
                    "user interaction patterns."
                )
        
        if 'response_quality' in metrics:
            quality = metrics['response_quality'].value
            if quality < 0.8:
                recommendations.append(
                    "Response quality needs improvement. Consider enhancing agent knowledge "
                    "bases and response generation algorithms."
                )
        
        # Agent-specific recommendations
        low_performing_agents = []
        for agent_id, results in agent_results.items():
            avg_performance = np.mean(list(results.values()))
            if avg_performance < 0.6:
                low_performing_agents.append(agent_id)
        
        if low_performing_agents:
            recommendations.append(
                f"Agents {', '.join(low_performing_agents)} show low performance. "
                "Consider additional training or capability enhancement."
            )
        
        # Evaluation-type specific recommendations
        if evaluation_type == EvaluationType.ROBUSTNESS:
            if 'performance_variance' in metrics and metrics['performance_variance'].value > 0.1:
                recommendations.append(
                    "High performance variance detected. Improve system robustness "
                    "across different scenarios and difficulty levels."
                )
        
        elif evaluation_type == EvaluationType.EFFICIENCY:
            if 'average_response_efficiency' in metrics and metrics['average_response_efficiency'].value < 0.7:
                recommendations.append(
                    "Response efficiency is below optimal. Consider optimizing agent "
                    "selection and response generation processes."
                )
        
        # Default recommendation if no specific issues found
        if not recommendations:
            recommendations.append(
                "System performance is within acceptable ranges. Continue monitoring "
                "and consider incremental improvements."
            )
        
        return recommendations
    
    def _compare_with_baseline(self, metrics: Dict[str, EvaluationMetric]) -> Optional[Dict[str, float]]:
        """Compare current metrics with baseline"""
        if not self.baseline_results:
            return None
        
        comparison = {}
        for metric_name, metric in metrics.items():
            if metric_name in self.baseline_results.metrics:
                baseline_value = self.baseline_results.metrics[metric_name].value
                current_value = metric.value
                
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    improvement = (current_value - baseline_value) / baseline_value if baseline_value != 0 else 0
                    comparison[metric_name] = improvement
        
        return comparison
    
    def _compare_with_previous(self, metrics: Dict[str, EvaluationMetric]) -> Optional[Dict[str, float]]:
        """Compare current metrics with previous evaluation"""
        if len(self.evaluation_history) < 1:
            return None
        
        previous_result = self.evaluation_history[-1]
        comparison = {}
        
        for metric_name, metric in metrics.items():
            if metric_name in previous_result.metrics:
                previous_value = previous_result.metrics[metric_name].value
                current_value = metric.value
                
                if isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
                    change = (current_value - previous_value) / previous_value if previous_value != 0 else 0
                    comparison[metric_name] = change
        
        return comparison
    
    def _update_performance_trends(self, evaluation_result: EvaluationResult) -> None:
        """Update performance trends tracking"""
        for metric_name, metric in evaluation_result.metrics.items():
            if isinstance(metric.value, (int, float)):
                self.performance_trends[metric_name].append({
                    'timestamp': evaluation_result.timestamp,
                    'value': metric.value,
                    'evaluation_id': evaluation_result.evaluation_id
                })
                
                # Keep only recent trends
                if len(self.performance_trends[metric_name]) > 100:
                    self.performance_trends[metric_name] = self.performance_trends[metric_name][-100:]
    
    def _save_evaluation_results(self, evaluation_result: EvaluationResult) -> None:
        """Save evaluation results to file"""
        # Create results directory
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        result_file = os.path.join(results_dir, f"{evaluation_result.evaluation_id}.json")
        
        # Convert result to serializable format
        result_dict = {
            'evaluation_id': evaluation_result.evaluation_id,
            'evaluation_type': evaluation_result.evaluation_type.value,
            'timestamp': evaluation_result.timestamp,
            'duration': evaluation_result.duration,
            'overall_score': evaluation_result.overall_score,
            'performance_grade': evaluation_result.performance_grade,
            'metrics': {name: asdict(metric) for name, metric in evaluation_result.metrics.items()},
            'scenario_results': evaluation_result.scenario_results,
            'agent_results': evaluation_result.agent_results,
            'baseline_comparison': evaluation_result.baseline_comparison,
            'previous_comparison': evaluation_result.previous_comparison,
            'recommendations': evaluation_result.recommendations
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"Saved evaluation results to {result_file}")
    
    def _generate_evaluation_plots(self, evaluation_result: EvaluationResult) -> None:
        """Generate evaluation plots"""
        try:
            # Create plots directory
            plots_dir = "evaluation_plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Performance distribution plot
            self._plot_performance_distribution(evaluation_result, plots_dir)
            
            # Performance trends plot
            self._plot_performance_trends(plots_dir)
            
            # Agent comparison plot
            self._plot_agent_comparison(evaluation_result, plots_dir)
            
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    def _plot_performance_distribution(self, evaluation_result: EvaluationResult, plots_dir: str) -> None:
        """Plot performance distribution"""
        key_metrics = ['task_completion_rate', 'user_satisfaction', 'response_quality']
        
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(15, 5))
        if len(key_metrics) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(key_metrics):
            dist_metric_name = f"{metric_name}_distribution"
            if dist_metric_name in evaluation_result.metrics:
                values = evaluation_result.metrics[dist_metric_name].value
                axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{metric_name.replace("_", " ").title()}')
                axes[i].set_xlabel('Score')
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{evaluation_result.evaluation_id}_distribution.png"))
        plt.close()
    
    def _plot_performance_trends(self, plots_dir: str) -> None:
        """Plot performance trends over time"""
        if len(self.evaluation_history) < 2:
            return
        
        key_metrics = ['task_completion_rate', 'user_satisfaction', 'response_quality']
        
        fig, axes = plt.subplots(len(key_metrics), 1, figsize=(12, 8))
        if len(key_metrics) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(key_metrics):
            timestamps = []
            values = []
            
            for result in self.evaluation_history:
                if metric_name in result.metrics:
                    timestamps.append(result.timestamp)
                    values.append(result.metrics[metric_name].value)
            
            if timestamps and values:
                axes[i].plot(timestamps, values, marker='o', linewidth=2, markersize=4)
                axes[i].set_title(f'{metric_name.replace("_", " ").title()} Trend')
                axes[i].set_ylabel('Score')
                axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Timestamp')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "performance_trends.png"))
        plt.close()
    
    def _plot_agent_comparison(self, evaluation_result: EvaluationResult, plots_dir: str) -> None:
        """Plot agent performance comparison"""
        if not evaluation_result.agent_results:
            return
        
        agents = list(evaluation_result.agent_results.keys())
        metrics = list(next(iter(evaluation_result.agent_results.values())).keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(agents))
        width = 0.15
        
        for i, metric in enumerate(metrics[:5]):  # Limit to 5 metrics for readability
            values = [evaluation_result.agent_results[agent][metric] for agent in agents]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Agents')
        ax.set_ylabel('Score')
        ax.set_title('Agent Performance Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(agents, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{evaluation_result.evaluation_id}_agent_comparison.png"))
        plt.close()
    
    def set_baseline(self, evaluation_result: EvaluationResult) -> None:
        """Set baseline results for comparison"""
        self.baseline_results = evaluation_result
        logger.info(f"Set baseline from evaluation {evaluation_result.evaluation_id}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        if not self.evaluation_history:
            return {'message': 'No evaluations completed yet'}
        
        latest_result = self.evaluation_history[-1]
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'latest_evaluation': {
                'id': latest_result.evaluation_id,
                'type': latest_result.evaluation_type.value,
                'overall_score': latest_result.overall_score,
                'grade': latest_result.performance_grade,
                'timestamp': latest_result.timestamp
            },
            'performance_trends': {},
            'recommendations': latest_result.recommendations
        }
        
        # Add performance trends for key metrics
        key_metrics = ['task_completion_rate', 'user_satisfaction', 'response_quality']
        for metric in key_metrics:
            if metric in self.performance_trends and len(self.performance_trends[metric]) > 1:
                recent_values = [item['value'] for item in self.performance_trends[metric][-5:]]
                summary['performance_trends'][metric] = {
                    'current': recent_values[-1],
                    'trend': 'improving' if recent_values[-1] > recent_values[0] else 'declining',
                    'average': np.mean(recent_values)
                }
        
        return summary
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Evaluation manager cleanup completed")