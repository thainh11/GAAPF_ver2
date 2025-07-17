"""Curriculum Manager for GAAPF RL System

This module manages curriculum learning for the reinforcement learning
components, providing progressive difficulty and adaptive learning strategies.
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)

class CurriculumStage(Enum):
    """Curriculum learning stages"""
    FOUNDATION = "foundation"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class DifficultyMetric(Enum):
    """Metrics for measuring task difficulty"""
    COMPLEXITY = "complexity"
    COGNITIVE_LOAD = "cognitive_load"
    PREREQUISITE_COUNT = "prerequisite_count"
    ABSTRACTION_LEVEL = "abstraction_level"
    PROBLEM_SIZE = "problem_size"
    TIME_CONSTRAINT = "time_constraint"

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    # Stage progression
    num_stages: int = 5
    episodes_per_stage: int = 1000
    min_success_rate: float = 0.7
    progression_window: int = 100
    
    # Difficulty scaling
    initial_difficulty: float = 0.1
    max_difficulty: float = 1.0
    difficulty_increment: float = 0.1
    adaptive_scaling: bool = True
    
    # Performance thresholds
    mastery_threshold: float = 0.8
    struggle_threshold: float = 0.4
    plateau_threshold: float = 0.05
    plateau_window: int = 50
    
    # Adaptation parameters
    difficulty_adaptation_rate: float = 0.1
    complexity_weight: float = 0.3
    performance_weight: float = 0.4
    engagement_weight: float = 0.3
    
    # Task distribution
    task_variety: float = 0.8
    prerequisite_enforcement: bool = True
    spiral_learning: bool = True
    
    # Evaluation
    evaluation_frequency: int = 50
    retention_test_frequency: int = 200
    
@dataclass
class TaskTemplate:
    """Template for generating curriculum tasks"""
    task_id: str
    name: str
    description: str
    stage: CurriculumStage
    base_difficulty: float
    
    # Task characteristics
    concepts: List[str]
    prerequisites: List[str]
    learning_objectives: List[str]
    estimated_duration: float
    
    # Difficulty factors
    complexity_factors: Dict[str, float]
    cognitive_load_factors: Dict[str, float]
    
    # Adaptation parameters
    difficulty_range: Tuple[float, float]
    adaptable_parameters: List[str]
    
    # Metadata
    tags: List[str]
    category: str
    subcategory: Optional[str] = None

@dataclass
class CurriculumMetrics:
    """Metrics for curriculum learning progress"""
    current_stage: CurriculumStage = CurriculumStage.FOUNDATION
    stage_progress: float = 0.0
    overall_progress: float = 0.0
    
    # Performance metrics
    stage_success_rates: Dict[str, List[float]] = None
    concept_mastery: Dict[str, float] = None
    learning_velocity: float = 0.0
    retention_scores: Dict[str, float] = None
    
    # Difficulty metrics
    current_difficulty: float = 0.1
    difficulty_history: List[float] = None
    adaptation_history: List[Dict[str, float]] = None
    
    # Engagement metrics
    engagement_scores: List[float] = None
    struggle_episodes: int = 0
    mastery_episodes: int = 0
    plateau_episodes: int = 0
    
    def __post_init__(self):
        if self.stage_success_rates is None:
            self.stage_success_rates = defaultdict(list)
        if self.concept_mastery is None:
            self.concept_mastery = {}
        if self.retention_scores is None:
            self.retention_scores = {}
        if self.difficulty_history is None:
            self.difficulty_history = []
        if self.adaptation_history is None:
            self.adaptation_history = []
        if self.engagement_scores is None:
            self.engagement_scores = []

class CurriculumManager:
    """Manages curriculum learning for GAAPF RL system"""
    
    def __init__(self, config: CurriculumConfig):
        """
        Initialize curriculum manager.
        
        Parameters:
        ----------
        config : CurriculumConfig
            Curriculum configuration
        """
        self.config = config
        self.metrics = CurriculumMetrics()
        
        # Curriculum state
        self.current_stage = CurriculumStage.FOUNDATION
        self.stage_episode_count = 0
        self.total_episodes = 0
        
        # Task management
        self.task_templates = {}
        self.active_tasks = []
        self.completed_tasks = set()
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.recent_performance = deque(maxlen=self.config.progression_window)
        self.concept_performance = defaultdict(list)
        
        # Difficulty adaptation
        self.current_difficulty = config.initial_difficulty
        self.difficulty_adjustments = deque(maxlen=100)
        
        # Initialize task templates
        self._initialize_task_templates()
        
        logger.info(f"Initialized curriculum manager with {len(self.task_templates)} task templates")
    
    def _initialize_task_templates(self) -> None:
        """Initialize task templates for different stages"""
        # Foundation stage tasks
        foundation_tasks = [
            TaskTemplate(
                task_id="foundation_variables",
                name="Variables and Data Types",
                description="Learn about variables and basic data types",
                stage=CurriculumStage.FOUNDATION,
                base_difficulty=0.1,
                concepts=["variables", "data_types", "assignment"],
                prerequisites=[],
                learning_objectives=["understand variables", "use basic data types"],
                estimated_duration=300,
                complexity_factors={"concept_count": 0.2, "abstraction": 0.1},
                cognitive_load_factors={"memory_load": 0.1, "processing": 0.2},
                difficulty_range=(0.05, 0.3),
                adaptable_parameters=["example_count", "explanation_depth"],
                tags=["beginner", "fundamentals"],
                category="programming_basics"
            ),
            TaskTemplate(
                task_id="foundation_operators",
                name="Basic Operators",
                description="Learn arithmetic and comparison operators",
                stage=CurriculumStage.FOUNDATION,
                base_difficulty=0.15,
                concepts=["arithmetic_operators", "comparison_operators"],
                prerequisites=["variables"],
                learning_objectives=["use arithmetic operators", "make comparisons"],
                estimated_duration=400,
                complexity_factors={"operator_count": 0.3, "precedence": 0.2},
                cognitive_load_factors={"memory_load": 0.2, "processing": 0.3},
                difficulty_range=(0.1, 0.4),
                adaptable_parameters=["operator_variety", "expression_complexity"],
                tags=["beginner", "operators"],
                category="programming_basics"
            )
        ]
        
        # Basic stage tasks
        basic_tasks = [
            TaskTemplate(
                task_id="basic_conditionals",
                name="Conditional Statements",
                description="Learn if-else statements and boolean logic",
                stage=CurriculumStage.BASIC,
                base_difficulty=0.3,
                concepts=["conditionals", "boolean_logic", "control_flow"],
                prerequisites=["variables", "comparison_operators"],
                learning_objectives=["write conditional statements", "understand boolean logic"],
                estimated_duration=600,
                complexity_factors={"nesting_depth": 0.4, "condition_complexity": 0.3},
                cognitive_load_factors={"logic_complexity": 0.4, "branching": 0.3},
                difficulty_range=(0.2, 0.6),
                adaptable_parameters=["nesting_levels", "condition_count"],
                tags=["control_flow", "logic"],
                category="programming_basics"
            ),
            TaskTemplate(
                task_id="basic_loops",
                name="Loop Structures",
                description="Learn for and while loops",
                stage=CurriculumStage.BASIC,
                base_difficulty=0.4,
                concepts=["loops", "iteration", "loop_control"],
                prerequisites=["variables", "conditionals"],
                learning_objectives=["write loops", "control iteration"],
                estimated_duration=800,
                complexity_factors={"loop_nesting": 0.5, "termination_logic": 0.4},
                cognitive_load_factors={"iteration_tracking": 0.5, "state_management": 0.4},
                difficulty_range=(0.3, 0.7),
                adaptable_parameters=["loop_complexity", "nesting_depth"],
                tags=["loops", "iteration"],
                category="programming_basics"
            )
        ]
        
        # Intermediate stage tasks
        intermediate_tasks = [
            TaskTemplate(
                task_id="intermediate_functions",
                name="Functions and Modules",
                description="Learn to write and use functions",
                stage=CurriculumStage.INTERMEDIATE,
                base_difficulty=0.5,
                concepts=["functions", "parameters", "return_values", "scope"],
                prerequisites=["variables", "conditionals", "loops"],
                learning_objectives=["write functions", "understand scope", "use parameters"],
                estimated_duration=1200,
                complexity_factors={"parameter_count": 0.3, "function_complexity": 0.5},
                cognitive_load_factors={"abstraction": 0.6, "scope_management": 0.4},
                difficulty_range=(0.4, 0.8),
                adaptable_parameters=["function_complexity", "parameter_variety"],
                tags=["functions", "abstraction"],
                category="programming_intermediate"
            ),
            TaskTemplate(
                task_id="intermediate_data_structures",
                name="Data Structures",
                description="Learn lists, dictionaries, and sets",
                stage=CurriculumStage.INTERMEDIATE,
                base_difficulty=0.6,
                concepts=["lists", "dictionaries", "sets", "data_manipulation"],
                prerequisites=["variables", "loops", "functions"],
                learning_objectives=["use data structures", "manipulate collections"],
                estimated_duration=1500,
                complexity_factors={"structure_complexity": 0.6, "operations": 0.4},
                cognitive_load_factors={"data_modeling": 0.7, "operation_chaining": 0.5},
                difficulty_range=(0.5, 0.9),
                adaptable_parameters=["structure_size", "operation_complexity"],
                tags=["data_structures", "collections"],
                category="programming_intermediate"
            )
        ]
        
        # Advanced stage tasks
        advanced_tasks = [
            TaskTemplate(
                task_id="advanced_oop",
                name="Object-Oriented Programming",
                description="Learn classes, inheritance, and polymorphism",
                stage=CurriculumStage.ADVANCED,
                base_difficulty=0.7,
                concepts=["classes", "objects", "inheritance", "polymorphism"],
                prerequisites=["functions", "data_structures"],
                learning_objectives=["design classes", "use inheritance", "apply polymorphism"],
                estimated_duration=2000,
                complexity_factors={"class_hierarchy": 0.7, "method_complexity": 0.6},
                cognitive_load_factors={"abstraction": 0.8, "design_thinking": 0.7},
                difficulty_range=(0.6, 1.0),
                adaptable_parameters=["hierarchy_depth", "method_count"],
                tags=["oop", "design"],
                category="programming_advanced"
            ),
            TaskTemplate(
                task_id="advanced_algorithms",
                name="Algorithms and Complexity",
                description="Learn sorting, searching, and algorithm analysis",
                stage=CurriculumStage.ADVANCED,
                base_difficulty=0.8,
                concepts=["algorithms", "sorting", "searching", "complexity_analysis"],
                prerequisites=["functions", "data_structures", "loops"],
                learning_objectives=["implement algorithms", "analyze complexity"],
                estimated_duration=2500,
                complexity_factors={"algorithm_complexity": 0.8, "optimization": 0.7},
                cognitive_load_factors={"algorithmic_thinking": 0.9, "analysis": 0.8},
                difficulty_range=(0.7, 1.0),
                adaptable_parameters=["algorithm_variety", "optimization_level"],
                tags=["algorithms", "complexity"],
                category="programming_advanced"
            )
        ]
        
        # Expert stage tasks
        expert_tasks = [
            TaskTemplate(
                task_id="expert_design_patterns",
                name="Design Patterns",
                description="Learn common design patterns and architectural principles",
                stage=CurriculumStage.EXPERT,
                base_difficulty=0.9,
                concepts=["design_patterns", "architecture", "best_practices"],
                prerequisites=["oop", "algorithms"],
                learning_objectives=["apply design patterns", "design architecture"],
                estimated_duration=3000,
                complexity_factors={"pattern_complexity": 0.9, "integration": 0.8},
                cognitive_load_factors={"design_thinking": 1.0, "abstraction": 0.9},
                difficulty_range=(0.8, 1.0),
                adaptable_parameters=["pattern_count", "integration_complexity"],
                tags=["patterns", "architecture"],
                category="software_engineering"
            )
        ]
        
        # Combine all tasks
        all_tasks = foundation_tasks + basic_tasks + intermediate_tasks + advanced_tasks + expert_tasks
        
        # Store tasks by ID
        for task in all_tasks:
            self.task_templates[task.task_id] = task
        
        logger.info(f"Initialized {len(all_tasks)} task templates across {len(CurriculumStage)} stages")
    
    def get_current_task(self, agent_performance: Dict[str, float],
                        user_context: Dict[str, Any]) -> TaskTemplate:
        """Get current task based on curriculum stage and performance"""
        # Update stage if needed
        self._update_curriculum_stage(agent_performance)
        
        # Get available tasks for current stage
        available_tasks = self._get_available_tasks()
        
        if not available_tasks:
            # Fallback to any task from current stage
            available_tasks = [task for task in self.task_templates.values() 
                             if task.stage == self.current_stage]
        
        if not available_tasks:
            # Ultimate fallback
            available_tasks = list(self.task_templates.values())
        
        # Select task based on performance and preferences
        selected_task = self._select_optimal_task(available_tasks, agent_performance, user_context)
        
        # Adapt task difficulty
        adapted_task = self._adapt_task_difficulty(selected_task, agent_performance)
        
        return adapted_task
    
    def _update_curriculum_stage(self, agent_performance: Dict[str, float]) -> None:
        """Update curriculum stage based on performance"""
        self.stage_episode_count += 1
        self.total_episodes += 1
        
        # Add performance to history
        avg_performance = np.mean(list(agent_performance.values()))
        self.recent_performance.append(avg_performance)
        self.performance_history.append(avg_performance)
        
        # Check for stage progression
        if self._should_progress_stage():
            self._progress_to_next_stage()
        elif self._should_regress_stage():
            self._regress_to_previous_stage()
        
        # Update metrics
        self._update_stage_metrics(agent_performance)
    
    def _should_progress_stage(self) -> bool:
        """Check if should progress to next stage"""
        if len(self.recent_performance) < self.config.progression_window:
            return False
        
        # Check minimum episodes in stage
        if self.stage_episode_count < self.config.episodes_per_stage // 2:
            return False
        
        # Check performance criteria
        recent_avg = np.mean(list(self.recent_performance))
        success_rate = recent_avg
        
        # Check mastery threshold
        mastery_achieved = success_rate >= self.config.mastery_threshold
        
        # Check concept mastery
        current_concepts = self._get_current_stage_concepts()
        concept_mastery = all(
            self.metrics.concept_mastery.get(concept, 0) >= self.config.mastery_threshold
            for concept in current_concepts
        )
        
        return mastery_achieved and concept_mastery
    
    def _should_regress_stage(self) -> bool:
        """Check if should regress to previous stage"""
        if len(self.recent_performance) < self.config.progression_window:
            return False
        
        # Only regress if struggling significantly
        recent_avg = np.mean(list(self.recent_performance))
        return recent_avg < self.config.struggle_threshold
    
    def _progress_to_next_stage(self) -> None:
        """Progress to next curriculum stage"""
        stages = list(CurriculumStage)
        current_index = stages.index(self.current_stage)
        
        if current_index < len(stages) - 1:
            old_stage = self.current_stage
            self.current_stage = stages[current_index + 1]
            self.stage_episode_count = 0
            self.recent_performance.clear()
            
            logger.info(f"Progressed from {old_stage.value} to {self.current_stage.value}")
            
            # Update metrics
            self.metrics.current_stage = self.current_stage
            self._calculate_overall_progress()
    
    def _regress_to_previous_stage(self) -> None:
        """Regress to previous curriculum stage"""
        stages = list(CurriculumStage)
        current_index = stages.index(self.current_stage)
        
        if current_index > 0:
            old_stage = self.current_stage
            self.current_stage = stages[current_index - 1]
            self.stage_episode_count = 0
            self.recent_performance.clear()
            
            logger.info(f"Regressed from {old_stage.value} to {self.current_stage.value}")
            
            # Update metrics
            self.metrics.current_stage = self.current_stage
            self._calculate_overall_progress()
    
    def _get_available_tasks(self) -> List[TaskTemplate]:
        """Get available tasks for current stage"""
        available_tasks = []
        
        for task in self.task_templates.values():
            # Check stage match
            if task.stage != self.current_stage:
                continue
            
            # Check prerequisites
            if self.config.prerequisite_enforcement:
                if not self._check_prerequisites(task):
                    continue
            
            # Check if not recently completed (for variety)
            if self.config.task_variety > 0:
                if self._is_recently_completed(task):
                    if np.random.random() > self.config.task_variety:
                        continue
            
            available_tasks.append(task)
        
        return available_tasks
    
    def _check_prerequisites(self, task: TaskTemplate) -> bool:
        """Check if task prerequisites are met"""
        for prereq in task.prerequisites:
            if self.metrics.concept_mastery.get(prereq, 0) < self.config.min_success_rate:
                return False
        return True
    
    def _is_recently_completed(self, task: TaskTemplate) -> bool:
        """Check if task was recently completed"""
        # Simple implementation - could be more sophisticated
        return task.task_id in self.completed_tasks
    
    def _select_optimal_task(self, available_tasks: List[TaskTemplate],
                           agent_performance: Dict[str, float],
                           user_context: Dict[str, Any]) -> TaskTemplate:
        """Select optimal task from available tasks"""
        if len(available_tasks) == 1:
            return available_tasks[0]
        
        # Calculate task scores
        task_scores = {}
        
        for task in available_tasks:
            score = self._calculate_task_score(task, agent_performance, user_context)
            task_scores[task.task_id] = score
        
        # Select task with highest score
        best_task_id = max(task_scores, key=task_scores.get)
        best_task = next(task for task in available_tasks if task.task_id == best_task_id)
        
        return best_task
    
    def _calculate_task_score(self, task: TaskTemplate,
                            agent_performance: Dict[str, float],
                            user_context: Dict[str, Any]) -> float:
        """Calculate task selection score"""
        score = 0.0
        
        # Performance alignment
        avg_performance = np.mean(list(agent_performance.values()))
        difficulty_match = 1.0 - abs(task.base_difficulty - avg_performance)
        score += difficulty_match * self.config.performance_weight
        
        # Concept coverage
        concept_scores = []
        for concept in task.concepts:
            concept_mastery = self.metrics.concept_mastery.get(concept, 0)
            # Prefer concepts that need work but are not too difficult
            concept_score = 1.0 - abs(concept_mastery - 0.6)
            concept_scores.append(concept_score)
        
        if concept_scores:
            score += np.mean(concept_scores) * self.config.complexity_weight
        
        # User engagement factors
        user_level = user_context.get('user_profile', {}).get('experience_level', 'beginner')
        engagement_score = user_context.get('engagement_score', 0.5)
        
        # Adjust for user level
        level_mapping = {'beginner': 0.2, 'intermediate': 0.5, 'advanced': 0.8}
        target_difficulty = level_mapping.get(user_level, 0.5)
        level_match = 1.0 - abs(task.base_difficulty - target_difficulty)
        
        score += level_match * self.config.engagement_weight * engagement_score
        
        # Variety bonus
        if task.task_id not in self.completed_tasks:
            score += 0.1
        
        return score
    
    def _adapt_task_difficulty(self, task: TaskTemplate,
                             agent_performance: Dict[str, float]) -> TaskTemplate:
        """Adapt task difficulty based on performance"""
        if not self.config.adaptive_scaling:
            return task
        
        # Calculate target difficulty
        avg_performance = np.mean(list(agent_performance.values()))
        
        # Adaptive difficulty adjustment
        if avg_performance > self.config.mastery_threshold:
            # Increase difficulty
            difficulty_adjustment = self.config.difficulty_adaptation_rate
        elif avg_performance < self.config.struggle_threshold:
            # Decrease difficulty
            difficulty_adjustment = -self.config.difficulty_adaptation_rate
        else:
            # Maintain current difficulty
            difficulty_adjustment = 0.0
        
        # Apply adjustment
        new_difficulty = task.base_difficulty + difficulty_adjustment
        new_difficulty = np.clip(new_difficulty, *task.difficulty_range)
        
        # Create adapted task (copy with modified difficulty)
        adapted_task = TaskTemplate(
            task_id=task.task_id,
            name=task.name,
            description=task.description,
            stage=task.stage,
            base_difficulty=new_difficulty,
            concepts=task.concepts.copy(),
            prerequisites=task.prerequisites.copy(),
            learning_objectives=task.learning_objectives.copy(),
            estimated_duration=task.estimated_duration,
            complexity_factors=task.complexity_factors.copy(),
            cognitive_load_factors=task.cognitive_load_factors.copy(),
            difficulty_range=task.difficulty_range,
            adaptable_parameters=task.adaptable_parameters.copy(),
            tags=task.tags.copy(),
            category=task.category,
            subcategory=task.subcategory
        )
        
        # Store difficulty adjustment
        self.difficulty_adjustments.append({
            'task_id': task.task_id,
            'original_difficulty': task.base_difficulty,
            'adapted_difficulty': new_difficulty,
            'adjustment': difficulty_adjustment,
            'performance': avg_performance
        })
        
        return adapted_task
    
    def update_task_performance(self, task_id: str, performance_metrics: Dict[str, float]) -> None:
        """Update performance metrics for a completed task"""
        # Mark task as completed
        self.completed_tasks.add(task_id)
        
        # Update concept mastery
        if task_id in self.task_templates:
            task = self.task_templates[task_id]
            
            for concept in task.concepts:
                # Update concept performance
                self.concept_performance[concept].append(performance_metrics.get('success_rate', 0.0))
                
                # Calculate rolling average for concept mastery
                recent_scores = self.concept_performance[concept][-10:]  # Last 10 attempts
                self.metrics.concept_mastery[concept] = np.mean(recent_scores)
        
        # Update stage success rates
        stage_name = self.current_stage.value
        success_rate = performance_metrics.get('success_rate', 0.0)
        self.metrics.stage_success_rates[stage_name].append(success_rate)
        
        # Update engagement metrics
        engagement = performance_metrics.get('engagement_score', 0.5)
        self.metrics.engagement_scores.append(engagement)
        
        # Classify episode type
        if success_rate >= self.config.mastery_threshold:
            self.metrics.mastery_episodes += 1
        elif success_rate <= self.config.struggle_threshold:
            self.metrics.struggle_episodes += 1
        
        # Check for plateau
        if len(self.recent_performance) >= self.config.plateau_window:
            recent_std = np.std(list(self.recent_performance)[-self.config.plateau_window:])
            if recent_std < self.config.plateau_threshold:
                self.metrics.plateau_episodes += 1
        
        logger.debug(f"Updated performance for task {task_id}: {performance_metrics}")
    
    def _update_stage_metrics(self, agent_performance: Dict[str, float]) -> None:
        """Update stage-specific metrics"""
        # Calculate stage progress
        stages = list(CurriculumStage)
        current_index = stages.index(self.current_stage)
        
        stage_progress = min(self.stage_episode_count / self.config.episodes_per_stage, 1.0)
        self.metrics.stage_progress = stage_progress
        
        # Calculate overall progress
        self._calculate_overall_progress()
        
        # Update difficulty metrics
        avg_performance = np.mean(list(agent_performance.values()))
        self.current_difficulty = self._calculate_current_difficulty(avg_performance)
        self.metrics.current_difficulty = self.current_difficulty
        self.metrics.difficulty_history.append(self.current_difficulty)
        
        # Calculate learning velocity
        if len(self.performance_history) >= 20:
            recent_trend = np.polyfit(range(20), list(self.performance_history)[-20:], 1)[0]
            self.metrics.learning_velocity = recent_trend
    
    def _calculate_overall_progress(self) -> None:
        """Calculate overall curriculum progress"""
        stages = list(CurriculumStage)
        current_index = stages.index(self.current_stage)
        
        # Progress through completed stages
        completed_progress = current_index / len(stages)
        
        # Progress within current stage
        stage_progress = min(self.stage_episode_count / self.config.episodes_per_stage, 1.0)
        current_stage_progress = stage_progress / len(stages)
        
        self.metrics.overall_progress = completed_progress + current_stage_progress
    
    def _calculate_current_difficulty(self, performance: float) -> float:
        """Calculate current effective difficulty"""
        # Base difficulty from current stage
        stage_difficulties = {
            CurriculumStage.FOUNDATION: 0.1,
            CurriculumStage.BASIC: 0.3,
            CurriculumStage.INTERMEDIATE: 0.5,
            CurriculumStage.ADVANCED: 0.7,
            CurriculumStage.EXPERT: 0.9
        }
        
        base_difficulty = stage_difficulties[self.current_stage]
        
        # Adjust based on recent adaptations
        if self.difficulty_adjustments:
            recent_adjustments = list(self.difficulty_adjustments)[-10:]
            avg_adjustment = np.mean([adj['adjustment'] for adj in recent_adjustments])
            base_difficulty += avg_adjustment
        
        return np.clip(base_difficulty, self.config.initial_difficulty, self.config.max_difficulty)
    
    def _get_current_stage_concepts(self) -> List[str]:
        """Get concepts for current stage"""
        stage_concepts = set()
        
        for task in self.task_templates.values():
            if task.stage == self.current_stage:
                stage_concepts.update(task.concepts)
        
        return list(stage_concepts)
    
    def conduct_retention_test(self) -> Dict[str, float]:
        """Conduct retention test for previously learned concepts"""
        retention_scores = {}
        
        # Test concepts from previous stages
        for concept, mastery_score in self.metrics.concept_mastery.items():
            # Simulate retention test (in real implementation, this would be actual testing)
            retention_decay = np.random.uniform(0.8, 1.0)  # Some retention loss
            retention_score = mastery_score * retention_decay
            retention_scores[concept] = retention_score
        
        # Update retention metrics
        self.metrics.retention_scores.update(retention_scores)
        
        logger.info(f"Conducted retention test for {len(retention_scores)} concepts")
        return retention_scores
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get current curriculum status"""
        return {
            'current_stage': self.current_stage.value,
            'stage_progress': self.metrics.stage_progress,
            'overall_progress': self.metrics.overall_progress,
            'stage_episode_count': self.stage_episode_count,
            'total_episodes': self.total_episodes,
            'current_difficulty': self.current_difficulty,
            'concept_mastery_count': len(self.metrics.concept_mastery),
            'mastery_episodes': self.metrics.mastery_episodes,
            'struggle_episodes': self.metrics.struggle_episodes,
            'learning_velocity': self.metrics.learning_velocity
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed curriculum metrics"""
        return asdict(self.metrics)
    
    def save_curriculum_state(self, filepath: str) -> None:
        """Save curriculum state to file"""
        state = {
            'config': asdict(self.config),
            'metrics': asdict(self.metrics),
            'current_stage': self.current_stage.value,
            'stage_episode_count': self.stage_episode_count,
            'total_episodes': self.total_episodes,
            'current_difficulty': self.current_difficulty,
            'completed_tasks': list(self.completed_tasks),
            'difficulty_adjustments': list(self.difficulty_adjustments)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Saved curriculum state to {filepath}")
    
    def load_curriculum_state(self, filepath: str) -> None:
        """Load curriculum state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_stage = CurriculumStage(state['current_stage'])
        self.stage_episode_count = state['stage_episode_count']
        self.total_episodes = state['total_episodes']
        self.current_difficulty = state['current_difficulty']
        self.completed_tasks = set(state['completed_tasks'])
        self.difficulty_adjustments = deque(state['difficulty_adjustments'], maxlen=100)
        
        logger.info(f"Loaded curriculum state from {filepath}")
    
    def reset_curriculum(self) -> None:
        """Reset curriculum to initial state"""
        self.current_stage = CurriculumStage.FOUNDATION
        self.stage_episode_count = 0
        self.total_episodes = 0
        self.current_difficulty = self.config.initial_difficulty
        self.completed_tasks.clear()
        self.difficulty_adjustments.clear()
        self.performance_history.clear()
        self.recent_performance.clear()
        self.concept_performance.clear()
        
        # Reset metrics
        self.metrics = CurriculumMetrics()
        
        logger.info("Reset curriculum to initial state")