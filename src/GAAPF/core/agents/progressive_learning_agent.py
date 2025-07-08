"""
Progressive Learning Agent Module for GAAPF Architecture

This module provides adaptive learning management with progress tracking,
difficulty adjustment, and personalized learning path optimization.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressiveLearningAgent:
    """
    Agent responsible for managing adaptive learning progression and difficulty adjustment.
    
    This agent monitors user performance, adjusts learning parameters dynamically,
    and provides personalized learning experiences based on individual progress.
    """
    
    def __init__(
        self,
        learning_hub,
        progress_tracker,
        is_logging: bool = False
    ):
        """
        Initialize the Progressive Learning Agent.
        
        Args:
            learning_hub: LearningHub instance for core learning operations
            progress_tracker: Progress tracking component
            is_logging: Whether to enable detailed logging
        """
        self.learning_hub = learning_hub
        self.progress_tracker = progress_tracker
        self.is_logging = is_logging
        
        # Learning parameters and thresholds
        self.performance_thresholds = {
            "mastery": 0.9,      # 90% accuracy for mastery
            "proficient": 0.75,  # 75% accuracy for proficiency
            "struggling": 0.5,   # Below 50% indicates difficulty
            "critical": 0.3      # Below 30% requires intervention
        }
        
        self.adjustment_factors = {
            "difficulty_step": 0.1,     # How much to adjust difficulty
            "pacing_factor": 0.2,       # How much to adjust pacing
            "content_depth_step": 0.15, # How much to adjust content depth
            "max_adjustments": 3        # Maximum adjustments per module
        }
        
        if self.is_logging:
            logger.info("Initialized ProgressiveLearningAgent")
    
    async def assess_learning_progress(
        self,
        user_id: str,
        session_data: Dict[str, Any],
        recent_performance: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assess overall learning progress and identify areas for improvement.
        
        Args:
            user_id: User identifier
            session_data: Current session information
            recent_performance: Recent performance data
            
        Returns:
            Comprehensive progress assessment
        """
        try:
            # Calculate performance metrics
            accuracy_scores = [p.get("accuracy", 0.0) for p in recent_performance]
            response_times = [p.get("response_time", 0.0) for p in recent_performance]
            completion_rates = [p.get("completion_rate", 0.0) for p in recent_performance]
            
            # Aggregate metrics
            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            avg_completion_rate = sum(completion_rates) / len(completion_rates) if completion_rates else 0.0
            
            # Determine performance level
            performance_level = self._classify_performance_level(avg_accuracy)
            
            # Calculate learning velocity (progress rate)
            learning_velocity = self._calculate_learning_velocity(recent_performance)
            
            # Identify learning patterns
            learning_patterns = self._analyze_learning_patterns(recent_performance)
            
            # Generate recommendations
            recommendations = await self._generate_learning_recommendations(
                performance_level=performance_level,
                learning_velocity=learning_velocity,
                patterns=learning_patterns,
                session_data=session_data
            )
            
            assessment = {
                "user_id": user_id,
                "assessment_timestamp": datetime.now().isoformat(),
                "performance_metrics": {
                    "average_accuracy": avg_accuracy,
                    "average_response_time": avg_response_time,
                    "average_completion_rate": avg_completion_rate,
                    "performance_level": performance_level,
                    "learning_velocity": learning_velocity
                },
                "learning_patterns": learning_patterns,
                "recommendations": recommendations,
                "next_adjustments": self._determine_next_adjustments(
                    performance_level, learning_patterns
                )
            }
            
            if self.is_logging:
                logger.info(f"Learning progress assessed for user {user_id}: {performance_level}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing learning progress: {str(e)}")
            return {"error": str(e)}
    
    async def adjust_learning_parameters(
        self,
        user_id: str,
        current_parameters: Dict[str, Any],
        assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adjust learning parameters based on performance assessment.
        
        Args:
            user_id: User identifier
            current_parameters: Current learning parameters
            assessment: Performance assessment results
            
        Returns:
            Updated learning parameters
        """
        try:
            recommendations = assessment.get("recommendations", {})
            performance_level = assessment.get("performance_metrics", {}).get("performance_level", "proficient")
            
            # Create adjusted parameters
            adjusted_parameters = current_parameters.copy()
            
            # Adjust difficulty level
            if recommendations.get("adjust_difficulty"):
                new_difficulty = self._adjust_difficulty_level(
                    current_difficulty=current_parameters.get("difficulty_level", "intermediate"),
                    performance_level=performance_level,
                    adjustment_direction=recommendations["difficulty_adjustment"]
                )
                adjusted_parameters["difficulty_level"] = new_difficulty
            
            # Adjust content pacing
            if recommendations.get("adjust_pacing"):
                new_pacing = self._adjust_content_pacing(
                    current_pacing=current_parameters.get("content_pacing", "normal"),
                    learning_velocity=assessment["performance_metrics"]["learning_velocity"],
                    adjustment_direction=recommendations["pacing_adjustment"]
                )
                adjusted_parameters["content_pacing"] = new_pacing
            
            # Adjust content depth
            if recommendations.get("adjust_content_depth"):
                new_depth = self._adjust_content_depth(
                    current_depth=current_parameters.get("content_depth", "standard"),
                    performance_level=performance_level,
                    adjustment_direction=recommendations["depth_adjustment"]
                )
                adjusted_parameters["content_depth"] = new_depth
            
            # Adjust practice frequency
            if recommendations.get("adjust_practice"):
                new_practice_frequency = self._adjust_practice_frequency(
                    current_frequency=current_parameters.get("practice_frequency", "normal"),
                    performance_level=performance_level
                )
                adjusted_parameters["practice_frequency"] = new_practice_frequency
            
            # Add metadata about adjustments
            adjusted_parameters.update({
                "last_adjustment_timestamp": datetime.now().isoformat(),
                "adjustment_reason": recommendations.get("primary_reason", "performance_optimization"),
                "previous_parameters": current_parameters,
                "adjustment_count": current_parameters.get("adjustment_count", 0) + 1
            })
            
            if self.is_logging:
                logger.info(f"Learning parameters adjusted for user {user_id}")
            
            return adjusted_parameters
            
        except Exception as e:
            logger.error(f"Error adjusting learning parameters: {str(e)}")
            return current_parameters
    
    async def provide_adaptive_guidance(
        self,
        user_id: str,
        current_context: Dict[str, Any],
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provide adaptive guidance based on current learning context and performance.
        
        Args:
            user_id: User identifier
            current_context: Current learning context
            performance_data: Recent performance information
            
        Returns:
            Adaptive guidance and suggestions
        """
        try:
            guidance = {
                "user_id": user_id,
                "guidance_timestamp": datetime.now().isoformat(),
                "contextual_tips": [],
                "motivation_message": "",
                "next_steps": [],
                "resources": [],
                "encouragement_level": "neutral"
            }
            
            # Analyze current struggle points
            struggle_areas = self._identify_struggle_areas(performance_data)
            
            # Generate contextual tips
            if struggle_areas:
                guidance["contextual_tips"] = await self._generate_contextual_tips(
                    struggle_areas, current_context
                )
            
            # Determine appropriate motivation level
            performance_trend = self._analyze_performance_trend(performance_data)
            guidance["encouragement_level"] = self._determine_encouragement_level(performance_trend)
            
            # Generate motivational message
            guidance["motivation_message"] = await self._generate_motivation_message(
                performance_trend=performance_trend,
                encouragement_level=guidance["encouragement_level"],
                user_context=current_context
            )
            
            # Suggest next steps
            guidance["next_steps"] = await self._suggest_next_steps(
                current_context=current_context,
                performance_data=performance_data,
                struggle_areas=struggle_areas
            )
            
            # Recommend additional resources
            if struggle_areas or performance_trend == "declining":
                guidance["resources"] = await self._recommend_additional_resources(
                    current_context, struggle_areas
                )
            
            if self.is_logging:
                logger.info(f"Adaptive guidance provided for user {user_id}")
            
            return guidance
            
        except Exception as e:
            logger.error(f"Error providing adaptive guidance: {str(e)}")
            return {"error": str(e)}
    
    def _classify_performance_level(self, accuracy: float) -> str:
        """Classify performance level based on accuracy score."""
        if accuracy >= self.performance_thresholds["mastery"]:
            return "mastery"
        elif accuracy >= self.performance_thresholds["proficient"]:
            return "proficient"
        elif accuracy >= self.performance_thresholds["struggling"]:
            return "struggling"
        else:
            return "critical"
    
    def _calculate_learning_velocity(self, performance_history: List[Dict[str, Any]]) -> float:
        """Calculate the rate of learning progress."""
        if len(performance_history) < 2:
            return 0.0
        
        # Calculate trend in accuracy over time
        accuracies = [p.get("accuracy", 0.0) for p in performance_history]
        
        # Simple linear regression to find slope
        n = len(accuracies)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(accuracies)
        sum_xy = sum(x * y for x, y in zip(x_values, accuracies))
        sum_x_squared = sum(x * x for x in x_values)
        
        if n * sum_x_squared - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x)
        return slope
    
    def _analyze_learning_patterns(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in learning behavior."""
        if not performance_history:
            return {}
        
        patterns = {
            "consistency": self._calculate_consistency(performance_history),
            "improvement_trend": self._calculate_improvement_trend(performance_history),
            "difficulty_adaptation": self._analyze_difficulty_adaptation(performance_history),
            "time_patterns": self._analyze_time_patterns(performance_history)
        }
        
        return patterns
    
    def _calculate_consistency(self, performance_history: List[Dict[str, Any]]) -> float:
        """Calculate consistency in performance."""
        accuracies = [p.get("accuracy", 0.0) for p in performance_history]
        if len(accuracies) < 2:
            return 1.0
        
        mean_accuracy = sum(accuracies) / len(accuracies)
        variance = sum((acc - mean_accuracy) ** 2 for acc in accuracies) / len(accuracies)
        std_dev = math.sqrt(variance)
        
        # Normalize consistency score (lower std_dev = higher consistency)
        consistency = max(0, 1 - (std_dev * 2))  # Scale factor of 2
        return min(1.0, consistency)
    
    def _calculate_improvement_trend(self, performance_history: List[Dict[str, Any]]) -> str:
        """Determine if performance is improving, declining, or stable."""
        if len(performance_history) < 3:
            return "insufficient_data"
        
        recent_scores = [p.get("accuracy", 0.0) for p in performance_history[-5:]]
        early_scores = [p.get("accuracy", 0.0) for p in performance_history[:5]]
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        early_avg = sum(early_scores) / len(early_scores)
        
        improvement = recent_avg - early_avg
        
        if improvement > 0.05:  # 5% improvement threshold
            return "improving"
        elif improvement < -0.05:  # 5% decline threshold
            return "declining"
        else:
            return "stable"
    
    def _analyze_difficulty_adaptation(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well the user adapts to difficulty changes."""
        difficulty_changes = []
        
        for i in range(1, len(performance_history)):
            prev_difficulty = performance_history[i-1].get("difficulty_level", "intermediate")
            curr_difficulty = performance_history[i].get("difficulty_level", "intermediate")
            
            if prev_difficulty != curr_difficulty:
                curr_accuracy = performance_history[i].get("accuracy", 0.0)
                difficulty_changes.append({
                    "from": prev_difficulty,
                    "to": curr_difficulty,
                    "resulting_accuracy": curr_accuracy
                })
        
        return {
            "adaptation_instances": len(difficulty_changes),
            "average_post_change_accuracy": (
                sum(change["resulting_accuracy"] for change in difficulty_changes) / 
                len(difficulty_changes) if difficulty_changes else 0.0
            ),
            "adaptation_quality": "good" if difficulty_changes and 
                                  (sum(change["resulting_accuracy"] for change in difficulty_changes) / 
                                   len(difficulty_changes)) > 0.6 else "needs_improvement"
        }
    
    def _analyze_time_patterns(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze time-based learning patterns."""
        response_times = [p.get("response_time", 0.0) for p in performance_history if p.get("response_time")]
        
        if not response_times:
            return {"pattern": "no_data"}
        
        avg_response_time = sum(response_times) / len(response_times)
        
        # Categorize response time patterns
        if avg_response_time < 30:  # seconds
            time_pattern = "quick_responder"
        elif avg_response_time < 60:
            time_pattern = "normal_pace"
        elif avg_response_time < 120:
            time_pattern = "deliberate_thinker"
        else:
            time_pattern = "slow_processor"
        
        return {
            "pattern": time_pattern,
            "average_response_time": avg_response_time,
            "response_consistency": self._calculate_time_consistency(response_times)
        }
    
    def _calculate_time_consistency(self, response_times: List[float]) -> float:
        """Calculate consistency in response times."""
        if len(response_times) < 2:
            return 1.0
        
        mean_time = sum(response_times) / len(response_times)
        variance = sum((time - mean_time) ** 2 for time in response_times) / len(response_times)
        std_dev = math.sqrt(variance)
        
        # Normalize consistency (coefficient of variation)
        if mean_time == 0:
            return 1.0
        
        cv = std_dev / mean_time
        consistency = max(0, 1 - cv)  # Lower CV = higher consistency
        return min(1.0, consistency)
    
    async def _generate_learning_recommendations(
        self,
        performance_level: str,
        learning_velocity: float,
        patterns: Dict[str, Any],
        session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate specific learning recommendations."""
        recommendations = {
            "adjust_difficulty": False,
            "adjust_pacing": False,
            "adjust_content_depth": False,
            "adjust_practice": False,
            "primary_reason": ""
        }
        
        # Difficulty adjustment recommendations
        if performance_level == "critical":
            recommendations.update({
                "adjust_difficulty": True,
                "difficulty_adjustment": "decrease",
                "primary_reason": "performance_below_threshold"
            })
        elif performance_level == "mastery" and learning_velocity > 0.1:
            recommendations.update({
                "adjust_difficulty": True,
                "difficulty_adjustment": "increase",
                "primary_reason": "ready_for_challenge"
            })
        
        # Pacing adjustment recommendations
        if learning_velocity < -0.05:  # Declining performance
            recommendations.update({
                "adjust_pacing": True,
                "pacing_adjustment": "slow_down",
                "primary_reason": "declining_performance"
            })
        elif learning_velocity > 0.1 and performance_level in ["proficient", "mastery"]:
            recommendations.update({
                "adjust_pacing": True,
                "pacing_adjustment": "speed_up",
                "primary_reason": "rapid_learning"
            })
        
        # Content depth recommendations
        if patterns.get("consistency", 0) < 0.5:  # Inconsistent performance
            recommendations.update({
                "adjust_content_depth": True,
                "depth_adjustment": "simplify",
                "primary_reason": "inconsistent_performance"
            })
        
        # Practice frequency recommendations
        if performance_level in ["struggling", "critical"]:
            recommendations.update({
                "adjust_practice": True,
                "primary_reason": "needs_more_practice"
            })
        
        return recommendations
    
    def _determine_next_adjustments(
        self,
        performance_level: str,
        patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Determine the next set of adjustments to make."""
        adjustments = []
        
        # Priority-based adjustment suggestions
        if performance_level == "critical":
            adjustments.append({
                "type": "immediate",
                "action": "reduce_difficulty",
                "priority": "high",
                "description": "Reduce content difficulty to build confidence"
            })
            adjustments.append({
                "type": "support",
                "action": "increase_practice",
                "priority": "high",
                "description": "Provide additional practice opportunities"
            })
        
        elif performance_level == "struggling":
            adjustments.append({
                "type": "gradual",
                "action": "adjust_pacing",
                "priority": "medium",
                "description": "Slow down content delivery pace"
            })
        
        elif performance_level == "mastery":
            adjustments.append({
                "type": "challenge",
                "action": "increase_difficulty",
                "priority": "medium",
                "description": "Introduce more challenging content"
            })
        
        # Pattern-based adjustments
        if patterns.get("consistency", 1.0) < 0.6:
            adjustments.append({
                "type": "stabilization",
                "action": "focus_on_consistency",
                "priority": "medium",
                "description": "Focus on consistent performance before advancing"
            })
        
        return adjustments
    
    def _adjust_difficulty_level(
        self,
        current_difficulty: str,
        performance_level: str,
        adjustment_direction: str
    ) -> str:
        """Adjust the difficulty level based on performance."""
        difficulty_levels = ["beginner", "intermediate", "advanced", "expert"]
        current_index = difficulty_levels.index(current_difficulty) if current_difficulty in difficulty_levels else 1
        
        if adjustment_direction == "increase" and current_index < len(difficulty_levels) - 1:
            return difficulty_levels[current_index + 1]
        elif adjustment_direction == "decrease" and current_index > 0:
            return difficulty_levels[current_index - 1]
        
        return current_difficulty
    
    def _adjust_content_pacing(
        self,
        current_pacing: str,
        learning_velocity: float,
        adjustment_direction: str
    ) -> str:
        """Adjust content pacing based on learning velocity."""
        pacing_levels = ["slow", "normal", "fast", "accelerated"]
        current_index = pacing_levels.index(current_pacing) if current_pacing in pacing_levels else 1
        
        if adjustment_direction == "speed_up" and current_index < len(pacing_levels) - 1:
            return pacing_levels[current_index + 1]
        elif adjustment_direction == "slow_down" and current_index > 0:
            return pacing_levels[current_index - 1]
        
        return current_pacing
    
    def _adjust_content_depth(
        self,
        current_depth: str,
        performance_level: str,
        adjustment_direction: str
    ) -> str:
        """Adjust content depth based on performance level."""
        depth_levels = ["basic", "standard", "detailed", "comprehensive"]
        current_index = depth_levels.index(current_depth) if current_depth in depth_levels else 1
        
        if adjustment_direction == "increase" and current_index < len(depth_levels) - 1:
            return depth_levels[current_index + 1]
        elif adjustment_direction == "simplify" and current_index > 0:
            return depth_levels[current_index - 1]
        
        return current_depth
    
    def _adjust_practice_frequency(
        self,
        current_frequency: str,
        performance_level: str
    ) -> str:
        """Adjust practice frequency based on performance level."""
        if performance_level in ["critical", "struggling"]:
            return "high"
        elif performance_level == "proficient":
            return "normal"
        else:  # mastery
            return "low"
    
    def _identify_struggle_areas(self, performance_data: Dict[str, Any]) -> List[str]:
        """Identify specific areas where the user is struggling."""
        struggle_areas = []
        
        # Check accuracy by topic/concept
        if "topic_performance" in performance_data:
            for topic, accuracy in performance_data["topic_performance"].items():
                if accuracy < self.performance_thresholds["struggling"]:
                    struggle_areas.append(topic)
        
        # Check response time issues
        if performance_data.get("average_response_time", 0) > 120:  # 2 minutes
            struggle_areas.append("response_time")
        
        # Check completion rate issues
        if performance_data.get("completion_rate", 1.0) < 0.8:
            struggle_areas.append("task_completion")
        
        return struggle_areas
    
    def _analyze_performance_trend(self, performance_data: Dict[str, Any]) -> str:
        """Analyze overall performance trend."""
        recent_accuracy = performance_data.get("recent_accuracy", [])
        
        if len(recent_accuracy) < 3:
            return "stable"
        
        # Check trend over last few attempts
        recent_avg = sum(recent_accuracy[-3:]) / 3
        earlier_avg = sum(recent_accuracy[:-3]) / max(1, len(recent_accuracy) - 3)
        
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _determine_encouragement_level(self, performance_trend: str) -> str:
        """Determine appropriate level of encouragement."""
        if performance_trend == "improving":
            return "positive"
        elif performance_trend == "declining":
            return "supportive"
        else:
            return "neutral"
    
    async def _generate_contextual_tips(
        self,
        struggle_areas: List[str],
        current_context: Dict[str, Any]
    ) -> List[str]:
        """Generate contextual tips for struggle areas."""
        tips = []
        
        for area in struggle_areas:
            if area == "response_time":
                tips.append("Take your time to read questions carefully before answering.")
            elif area == "task_completion":
                tips.append("Try breaking down complex problems into smaller steps.")
            else:
                tips.append(f"Consider reviewing the {area} concepts before moving forward.")
        
        return tips
    
    async def _generate_motivation_message(
        self,
        performance_trend: str,
        encouragement_level: str,
        user_context: Dict[str, Any]
    ) -> str:
        """Generate appropriate motivational message."""
        if encouragement_level == "positive":
            return "Great progress! You're mastering these concepts well. Keep up the excellent work!"
        elif encouragement_level == "supportive":
            return "Learning can be challenging, but you're making important progress. Don't give up!"
        else:
            return "You're doing well. Stay focused and continue learning at your own pace."
    
    async def _suggest_next_steps(
        self,
        current_context: Dict[str, Any],
        performance_data: Dict[str, Any],
        struggle_areas: List[str]
    ) -> List[str]:
        """Suggest appropriate next steps."""
        steps = []
        
        if struggle_areas:
            steps.append("Review the concepts you found challenging")
            steps.append("Practice with additional exercises in those areas")
        else:
            steps.append("Continue to the next module")
            steps.append("Try some advanced practice problems")
        
        return steps
    
    async def _recommend_additional_resources(
        self,
        current_context: Dict[str, Any],
        struggle_areas: List[str]
    ) -> List[Dict[str, str]]:
        """Recommend additional learning resources."""
        resources = []
        
        for area in struggle_areas:
            resources.append({
                "type": "documentation",
                "title": f"Additional reading on {area}",
                "description": f"Comprehensive guide covering {area} concepts"
            })
        
        return resources 