"""
Learning guidance module to help guide users through their learning journey.
"""

import logging
from typing import Dict, List, Optional, Any
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningGuidance:
    """
    Provides guidance for users through their learning journey.
    
    The LearningGuidance class is responsible for:
    1. Suggesting next steps in the learning process
    2. Providing learning path recommendations
    3. Identifying knowledge gaps and suggesting remediation
    4. Adapting guidance based on user progress and goals
    5. Tracking concept mastery and learning progression
    """
    
    def __init__(self, is_logging: bool = False):
        """
        Initialize the learning guidance system.
        
        Parameters:
        ----------
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        self.is_logging = is_logging
        
        if self.is_logging:
            logger.info("Initialized LearningGuidance")
    
    def get_next_steps(self, learning_context: Dict) -> Dict:
        """
        Get recommended next steps for the user based on their learning context.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        Dict
            Next steps recommendations
        """
        # Get relevant context
        framework_id = learning_context.get("framework_id", "")
        current_module = learning_context.get("current_module", "")
        learning_stage = learning_context.get("learning_stage", "exploration")
        framework_config = learning_context.get("framework_config", {})
        user_profile = learning_context.get("user_profile", {})
        interaction_count = learning_context.get("interaction_count", 0)
        
        # Get modules from framework config
        modules = framework_config.get("modules", {})
        
        # Track learning progress
        progress_info = self.track_learning_progress(learning_context)
        
        # Generate comprehensive next steps
        next_steps = {
            "recommendations": [],
            "current_module": current_module,
            "learning_stage": learning_stage,
            "progress": progress_info,
            "suggested_activities": []
        }
        
        # Add stage-specific recommendations
        if learning_stage == "exploration":
            next_steps["recommendations"].extend(self._get_exploration_recommendations(learning_context, modules))
        elif learning_stage == "concept":
            next_steps["recommendations"].extend(self._get_concept_recommendations(learning_context, modules))
        elif learning_stage == "practice":
            next_steps["recommendations"].extend(self._get_practice_recommendations(learning_context, modules))
        elif learning_stage == "assessment":
            next_steps["recommendations"].extend(self._get_assessment_recommendations(learning_context, modules))
        
        # Add module progression recommendations
        progression_recs = self._get_progression_recommendations(learning_context, modules, progress_info)
        next_steps["recommendations"].extend(progression_recs)
        
        # Add concept-specific guidance
        concept_guidance = self._get_concept_guidance(learning_context, modules)
        if concept_guidance:
            next_steps["concept_guidance"] = concept_guidance
        
        # Suggest activities based on interaction count and progress
        activities = self.suggest_practice_exercises(learning_context)
        next_steps["suggested_activities"] = activities
        
        return next_steps
    
    def track_learning_progress(self, learning_context: Dict) -> Dict:
        """
        Track and analyze learning progress.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        Dict
            Progress tracking information
        """
        framework_config = learning_context.get("framework_config", {})
        current_module = learning_context.get("current_module", "")
        user_profile = learning_context.get("user_profile", {})
        interaction_count = learning_context.get("interaction_count", 0)
        
        modules = framework_config.get("modules", {})
        completed_modules = user_profile.get("completed_modules", [])
        
        progress = {
            "total_modules": len(modules),
            "completed_modules": len(completed_modules),
            "current_module_progress": 0.0,
            "overall_progress": 0.0,
            "concepts_learned": [],
            "interaction_engagement": "low"
        }
        
        if modules:
            # Calculate overall progress
            progress["overall_progress"] = len(completed_modules) / len(modules)
            
            # Estimate current module progress based on interactions
            if current_module and interaction_count > 0:
                # Simple heuristic: more interactions suggest more progress
                max_interactions_per_module = 10
                progress["current_module_progress"] = min(interaction_count / max_interactions_per_module, 1.0)
            
            # Determine engagement level
            if interaction_count >= 8:
                progress["interaction_engagement"] = "high"
            elif interaction_count >= 4:
                progress["interaction_engagement"] = "medium"
            
            # Track concepts from completed modules
            for module_id in completed_modules:
                if module_id in modules:
                    module_concepts = modules[module_id].get("concepts", [])
                    progress["concepts_learned"].extend(module_concepts)
        
        return progress
    
    def suggest_practice_exercises(self, learning_context: Dict) -> List[Dict]:
        """
        Suggest practice exercises based on current learning context.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        List[Dict]
            List of suggested practice activities
        """
        current_module = learning_context.get("current_module", "")
        framework_config = learning_context.get("framework_config", {})
        learning_stage = learning_context.get("learning_stage", "exploration")
        
        modules = framework_config.get("modules", {})
        activities = []
        
        if current_module and current_module in modules:
            module_info = modules[current_module]
            concepts = module_info.get("concepts", [])
            complexity = module_info.get("complexity", "basic")
            
            # Suggest concept-specific exercises
            for concept in concepts[:3]:  # Focus on first 3 concepts
                activities.append({
                    "type": "concept_practice",
                    "title": f"Practice {concept}",
                    "description": f"Apply your understanding of {concept} with hands-on exercises",
                    "difficulty": complexity,
                    "estimated_time": "15-30 minutes"
                })
            
            # Suggest stage-appropriate activities
            if learning_stage == "exploration":
                activities.append({
                    "type": "exploration",
                    "title": "Explore Code Examples",
                    "description": f"Look at real-world examples of {current_module} concepts",
                    "difficulty": "beginner",
                    "estimated_time": "20-30 minutes"
                })
            elif learning_stage == "practice":
                activities.append({
                    "type": "coding_exercise",
                    "title": "Build a Mini Project",
                    "description": f"Create a small project using {current_module} concepts",
                    "difficulty": complexity,
                    "estimated_time": "45-60 minutes"
                })
        
        return activities
    
    def recommend_next_concepts(self, learning_context: Dict) -> List[str]:
        """
        Recommend the next concepts to learn.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        List[str]
            List of recommended concepts
        """
        current_module = learning_context.get("current_module", "")
        framework_config = learning_context.get("framework_config", {})
        user_profile = learning_context.get("user_profile", {})
        
        modules = framework_config.get("modules", {})
        completed_modules = user_profile.get("completed_modules", [])
        
        next_concepts = []
        
        if current_module and current_module in modules:
            # Get concepts from current module
            current_concepts = modules[current_module].get("concepts", [])
            next_concepts.extend(current_concepts)
            
            # Look for next module concepts
            module_order = list(modules.keys())
            if current_module in module_order:
                current_index = module_order.index(current_module)
                if current_index < len(module_order) - 1:
                    next_module = module_order[current_index + 1]
                    next_module_concepts = modules[next_module].get("concepts", [])
                    next_concepts.extend(next_module_concepts[:2])  # Add first 2 concepts from next module
        
        return next_concepts
    
    def enhance_response_with_guidance(self, response: Dict, learning_context: Dict) -> Dict:
        """
        Enhance an agent response with comprehensive learning guidance.
        
        Parameters:
        ----------
        response : Dict
            Original agent response
        learning_context : Dict
            Current learning context
            
        Returns:
        -------
        Dict
            Enhanced response with guidance
        """
        # Get comprehensive next steps
        next_steps = self.get_next_steps(learning_context)
        
        # Add guidance to response
        if "content" in response:
            content = response["content"]
            
            # Create enhanced guidance section
            guidance = self._create_guidance_section(next_steps, learning_context)
            
            # Add to response content
            response["content"] = content + guidance
        
        # Add detailed guidance data to response
        response["guidance"] = next_steps
        response["learning_recommendations"] = next_steps["recommendations"]
        response["suggested_activities"] = next_steps.get("suggested_activities", [])
        
        return response
    
    def _get_exploration_recommendations(self, learning_context: Dict, modules: Dict) -> List[Dict]:
        """Get recommendations for exploration stage."""
        current_module = learning_context.get("current_module", "")
        
        recommendations = []
        
        if current_module in modules:
            module_info = modules[current_module]
            concepts = module_info.get("concepts", [])
            
            recommendations.append({
                "type": "explore_concepts",
                "description": f"Explore core concepts: {', '.join(concepts[:3])}",
                "reason": "Understanding these fundamental concepts will build your foundation",
                "priority": "high"
            })
            
            recommendations.append({
                "type": "read_documentation",
                "description": f"Read about {module_info.get('title', current_module)}",
                "reason": "Getting familiar with the documentation will help you understand the framework better",
                "priority": "medium"
            })
        
        return recommendations
    
    def _get_concept_recommendations(self, learning_context: Dict, modules: Dict) -> List[Dict]:
        """Get recommendations for concept learning stage."""
        recommendations = [
            {
                "type": "deep_dive",
                "description": "Deep dive into current concepts with examples",
                "reason": "Thorough understanding will prepare you for practical application",
                "priority": "high"
            },
            {
                "type": "ask_questions",
                "description": "Ask specific questions about concepts you find challenging",
                "reason": "Clarifying doubts now will prevent confusion later",
                "priority": "high"
            }
        ]
        
        return recommendations
    
    def _get_practice_recommendations(self, learning_context: Dict, modules: Dict) -> List[Dict]:
        """Get recommendations for practice stage."""
        recommendations = [
            {
                "type": "hands_on_coding",
                "description": "Start with hands-on coding exercises",
                "reason": "Practice solidifies your understanding and builds muscle memory",
                "priority": "high"
            },
            {
                "type": "build_project",
                "description": "Build a small project using what you've learned",
                "reason": "Projects help you see how concepts work together in real scenarios",
                "priority": "medium"
            }
        ]
        
        return recommendations
    
    def _get_assessment_recommendations(self, learning_context: Dict, modules: Dict) -> List[Dict]:
        """Get recommendations for assessment stage."""
        recommendations = [
            {
                "type": "review_concepts",
                "description": "Review all concepts covered in this module",
                "reason": "Assessment will test your comprehensive understanding",
                "priority": "high"
            },
            {
                "type": "take_quiz",
                "description": "Take practice quizzes to test your knowledge",
                "reason": "Quizzes help identify areas that need more attention",
                "priority": "high"
            }
        ]
        
        return recommendations
    
    def _get_progression_recommendations(self, learning_context: Dict, modules: Dict, progress_info: Dict) -> List[Dict]:
        """Get module progression recommendations."""
        current_module = learning_context.get("current_module", "")
        recommendations = []
        
        # Check if ready for next module
        if progress_info["current_module_progress"] > 0.7:  # 70% progress threshold
            module_order = list(modules.keys())
            if current_module in module_order:
                current_index = module_order.index(current_module)
                if current_index < len(module_order) - 1:
                    next_module = module_order[current_index + 1]
                    next_module_info = modules[next_module]
                    recommendations.append({
                        "type": "next_module",
                        "description": f"Consider moving to: {next_module_info.get('title', next_module)}",
                        "reason": "You've made good progress on the current module",
                        "priority": "medium"
                    })
        
        return recommendations
    
    def _get_concept_guidance(self, learning_context: Dict, modules: Dict) -> Optional[Dict]:
        """Get concept-specific guidance."""
        current_module = learning_context.get("current_module", "")
        
        if current_module and current_module in modules:
            module_info = modules[current_module]
            concepts = module_info.get("concepts", [])
            
            return {
                "current_concepts": concepts,
                "focus_areas": concepts[:2],  # Focus on first 2 concepts
                "prerequisites": module_info.get("prerequisites", [])
            }
        
        return None
    
    def _create_guidance_section(self, next_steps: Dict, learning_context: Dict) -> str:
        """Create a formatted guidance section for the response."""
        guidance = "\n\n**🎯 Learning Path Guidance:**\n"
        
        # Add progress information
        progress = next_steps.get("progress", {})
        if progress:
            engagement = progress.get("interaction_engagement", "low")
            emoji = "🔥" if engagement == "high" else "📈" if engagement == "medium" else "🌱"
            guidance += f"\n{emoji} Your engagement level: {engagement.title()}\n"
        
        # Add recommendations
        recommendations = next_steps.get("recommendations", [])
        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3 recommendations
            priority_emoji = "🚀" if rec.get("priority") == "high" else "📝" if rec.get("priority") == "medium" else "💡"
            guidance += f"\n{i}. {priority_emoji} **{rec['description']}** - {rec['reason']}"
        
        # Add suggested activities
        activities = next_steps.get("suggested_activities", [])
        if activities:
            guidance += f"\n\n**💪 Suggested Activities:**"
            for activity in activities[:2]:  # Show top 2 activities
                guidance += f"\n• {activity['title']} ({activity.get('estimated_time', '15-30 min')})"
        
        return guidance 