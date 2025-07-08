"""
Curriculum Management Module for GAAPF Architecture

This module provides comprehensive curriculum management capabilities including
approval workflows, feedback processing, and curriculum adjustments.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CurriculumManager:
    """
    Manages curriculum approval workflows, feedback processing, and iterative
    curriculum improvements based on user input and preferences.
    """
    
    def __init__(
        self,
        framework_onboarding,
        learning_hub,
        is_logging: bool = False
    ):
        """
        Initialize the CurriculumManager.
        
        Args:
            framework_onboarding: FrameworkOnboarding instance for curriculum generation
            learning_hub: LearningHub instance for overall coordination
            is_logging: Whether to enable detailed logging
        """
        self.framework_onboarding = framework_onboarding
        self.learning_hub = learning_hub
        self.is_logging = is_logging
        
        # Configuration for approval process
        self.approval_config = {
            "max_iterations": 3,
            "timeout_minutes": 30,
            "auto_approve_threshold": 0.9,  # If user satisfaction score >= 90%
            "require_explicit_approval": True
        }
        
        if self.is_logging:
            logger.info("Initialized CurriculumManager")
    
    async def present_curriculum_for_approval(self, curriculum: Dict, user_id: str) -> Dict:
        """
        Present curriculum to user with approval options and gather feedback.
        
        Args:
            curriculum: Generated curriculum to present
            user_id: User ID for tracking approval history
            
        Returns:
            Dictionary containing approval status and user feedback
        """
        if self.is_logging:
            logger.info(f"Presenting curriculum for approval to user {user_id}")
        
        # Create approval presentation format
        presentation = self._format_curriculum_for_presentation(curriculum)
        
        # Track approval session
        approval_session = {
            "session_id": f"approval_{user_id}_{int(datetime.now().timestamp())}",
            "user_id": user_id,
            "curriculum_version": curriculum.get("version", "1.0"),
            "framework_name": curriculum.get("framework_name", "Unknown"),
            "presented_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Save approval session for tracking
        self._save_approval_session(approval_session)
        
        # Return presentation format with approval options
        return {
            "approval_session": approval_session,
            "curriculum_presentation": presentation,
            "approval_options": self._get_approval_options(),
            "feedback_prompts": self._get_feedback_prompts()
        }
    
    async def adjust_curriculum_based_feedback(
        self, 
        curriculum: Dict, 
        feedback: Dict, 
        user_config: Dict
    ) -> Dict:
        """
        Adjust curriculum based on user feedback and preferences.
        
        Args:
            curriculum: Original curriculum to adjust
            feedback: User feedback about the curriculum
            user_config: User configuration and preferences
            
        Returns:
            Adjusted curriculum incorporating user feedback
        """
        if self.is_logging:
            logger.info("Adjusting curriculum based on user feedback")
        
        # Analyze feedback to identify adjustment areas
        adjustment_areas = self._analyze_feedback(feedback)
        
        # Create adjusted curriculum
        adjusted_curriculum = curriculum.copy()
        adjusted_curriculum["adjustment_history"] = adjusted_curriculum.get("adjustment_history", [])
        
        # Apply adjustments based on feedback
        for area, adjustments in adjustment_areas.items():
            if area == "difficulty_level":
                adjusted_curriculum = self._adjust_difficulty_level(
                    adjusted_curriculum, adjustments, user_config
                )
            elif area == "content_focus":
                adjusted_curriculum = self._adjust_content_focus(
                    adjusted_curriculum, adjustments, user_config
                )
            elif area == "module_structure":
                adjusted_curriculum = self._adjust_module_structure(
                    adjusted_curriculum, adjustments, user_config
                )
            elif area == "learning_pace":
                adjusted_curriculum = self._adjust_learning_pace(
                    adjusted_curriculum, adjustments, user_config
                )
        
        # Update version and track changes
        adjusted_curriculum["version"] = str(float(curriculum.get("version", "1.0")) + 0.1)
        adjusted_curriculum["last_adjusted"] = datetime.now().isoformat()
        
        # Record adjustment details
        adjustment_record = {
            "timestamp": datetime.now().isoformat(),
            "feedback_summary": feedback.get("summary", ""),
            "adjustments_made": adjustment_areas,
            "user_satisfaction_before": feedback.get("satisfaction_score", 0),
            "adjustment_type": "user_feedback"
        }
        adjusted_curriculum["adjustment_history"].append(adjustment_record)
        
        if self.is_logging:
            logger.info(f"Curriculum adjusted. New version: {adjusted_curriculum['version']}")
        
        return adjusted_curriculum
    
    async def curriculum_approval_loop(
        self, 
        framework_name: str, 
        user_id: str, 
        user_config: Dict
    ) -> Dict:
        """
        Complete approval loop with iterations until user satisfaction.
        
        Args:
            framework_name: Name of the framework for curriculum generation
            user_id: User ID for tracking
            user_config: User configuration and preferences
            
        Returns:
            Final approved curriculum and approval metadata
        """
        if self.is_logging:
            logger.info(f"Starting curriculum approval loop for {framework_name}")
        
        iteration = 0
        max_iterations = self.approval_config["max_iterations"]
        
        # Generate initial curriculum
        current_curriculum = await self.framework_onboarding.initialize_framework(
            framework_name=framework_name,
            user_id=user_id,
            user_config=user_config,
            initialization_mode=user_config.get("initialization_mode", "quick")
        )
        
        approval_result = None
        
        while iteration < max_iterations:
            iteration += 1
            
            if self.is_logging:
                logger.info(f"Approval iteration {iteration}/{max_iterations}")
            
            # Present curriculum for approval
            presentation_result = await self.present_curriculum_for_approval(
                current_curriculum, user_id
            )
            
            # This would typically involve user interaction through CLI/UI
            # For now, we'll simulate the approval check
            approval_result = await self._simulate_user_approval_interaction(
                presentation_result, user_config
            )
            
            # Check if approved
            if approval_result["approved"]:
                if self.is_logging:
                    logger.info("Curriculum approved by user")
                break
            
            # If not approved and we have feedback, adjust curriculum
            if approval_result.get("feedback"):
                current_curriculum = await self.adjust_curriculum_based_feedback(
                    current_curriculum,
                    approval_result["feedback"], 
                    user_config
                )
            else:
                # No feedback provided, ask for specific feedback
                feedback_request = self._generate_feedback_request(current_curriculum)
                approval_result["feedback_request"] = feedback_request
                break
        
        # Finalize approval process
        final_result = {
            "curriculum": current_curriculum,
            "approval_metadata": {
                "approved": approval_result.get("approved", False),
                "iterations": iteration,
                "final_satisfaction_score": approval_result.get("satisfaction_score", 0),
                "approval_timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "framework_name": framework_name
            }
        }
        
        # Save final approved curriculum
        if approval_result.get("approved"):
            curriculum_path = self.framework_onboarding.save_curriculum(
                current_curriculum, user_id, framework_name
            )
            final_result["curriculum_path"] = curriculum_path
        
        return final_result
    
    def _format_curriculum_for_presentation(self, curriculum: Dict) -> Dict:
        """Format curriculum for user-friendly presentation."""
        presentation = {
            "overview": {
                "framework_name": curriculum.get("framework_name", "Unknown"),
                "estimated_duration": curriculum.get("estimated_duration", "Not specified"),
                "difficulty_level": curriculum.get("difficulty_level", "Not specified"),
                "total_modules": len(curriculum.get("modules", []))
            },
            "modules_summary": [],
            "learning_objectives": curriculum.get("learning_objectives", []),
            "prerequisites": curriculum.get("prerequisites", []),
            "assessment_strategy": curriculum.get("assessment_strategy", {})
        }
        
        # Format modules for presentation
        for i, module in enumerate(curriculum.get("modules", []), 1):
            module_summary = {
                "module_number": i,
                "title": module.get("title", f"Module {i}"),
                "description": module.get("description", "No description"),
                "estimated_time": module.get("estimated_time", "Not specified"),
                "key_concepts": module.get("concepts", [])[:5],  # Show first 5 concepts
                "learning_outcomes": module.get("learning_outcomes", [])[:3]  # Show first 3 outcomes
            }
            presentation["modules_summary"].append(module_summary)
        
        return presentation
    
    def _get_approval_options(self) -> List[Dict]:
        """Get available approval options for user interaction."""
        return [
            {
                "option": "approve",
                "label": "Approve Curriculum",
                "description": "Accept the curriculum as presented and proceed to learning"
            },
            {
                "option": "request_changes",
                "label": "Request Changes",
                "description": "Provide feedback for curriculum adjustments"
            },
            {
                "option": "regenerate",
                "label": "Regenerate Curriculum",
                "description": "Start over with a completely new curriculum"
            },
            {
                "option": "preview_module",
                "label": "Preview Module",
                "description": "Get detailed preview of a specific module"
            }
        ]
    
    def _get_feedback_prompts(self) -> List[Dict]:
        """Get feedback prompts to guide user input."""
        return [
            {
                "category": "difficulty_level",
                "prompt": "Is the difficulty level appropriate for your experience?",
                "options": ["too_easy", "appropriate", "too_difficult"]
            },
            {
                "category": "content_focus",
                "prompt": "Does the content focus match your learning goals?",
                "options": ["too_theoretical", "good_balance", "too_practical"]
            },
            {
                "category": "module_structure",
                "prompt": "How do you feel about the module organization?",
                "options": ["too_many_modules", "good_structure", "too_few_modules"]
            },
            {
                "category": "learning_pace",
                "prompt": "Is the learning pace suitable for you?",
                "options": ["too_fast", "appropriate", "too_slow"]
            }
        ]
    
    def _analyze_feedback(self, feedback: Dict) -> Dict:
        """Analyze user feedback to identify specific adjustment areas."""
        adjustment_areas = {}
        
        # Analyze difficulty feedback
        if feedback.get("difficulty_level"):
            if feedback["difficulty_level"] == "too_easy":
                adjustment_areas["difficulty_level"] = {"action": "increase", "degree": "moderate"}
            elif feedback["difficulty_level"] == "too_difficult":
                adjustment_areas["difficulty_level"] = {"action": "decrease", "degree": "moderate"}
        
        # Analyze content focus feedback
        if feedback.get("content_focus"):
            if feedback["content_focus"] == "too_theoretical":
                adjustment_areas["content_focus"] = {"action": "add_practical", "degree": "significant"}
            elif feedback["content_focus"] == "too_practical":
                adjustment_areas["content_focus"] = {"action": "add_theoretical", "degree": "moderate"}
        
        # Analyze module structure feedback
        if feedback.get("module_structure"):
            if feedback["module_structure"] == "too_many_modules":
                adjustment_areas["module_structure"] = {"action": "consolidate", "degree": "moderate"}
            elif feedback["module_structure"] == "too_few_modules":
                adjustment_areas["module_structure"] = {"action": "expand", "degree": "moderate"}
        
        # Analyze learning pace feedback
        if feedback.get("learning_pace"):
            if feedback["learning_pace"] == "too_fast":
                adjustment_areas["learning_pace"] = {"action": "slow_down", "degree": "moderate"}
            elif feedback["learning_pace"] == "too_slow":
                adjustment_areas["learning_pace"] = {"action": "speed_up", "degree": "moderate"}
        
        return adjustment_areas
    
    def _adjust_difficulty_level(self, curriculum: Dict, adjustments: Dict, user_config: Dict) -> Dict:
        """Adjust curriculum difficulty level based on feedback."""
        action = adjustments.get("action")
        degree = adjustments.get("degree", "moderate")
        
        if action == "increase":
            # Add more advanced topics, complex examples
            for module in curriculum.get("modules", []):
                module["include_advanced_topics"] = True
                module["complexity_level"] = "high"
                if degree == "significant":
                    module["prerequisites"] = module.get("prerequisites", []) + ["advanced_concepts"]
        
        elif action == "decrease":
            # Simplify topics, add more explanations
            for module in curriculum.get("modules", []):
                module["include_detailed_explanations"] = True
                module["complexity_level"] = "basic"
                if degree == "significant":
                    module["include_examples"] = True
                    module["step_by_step_guidance"] = True
        
        return curriculum
    
    def _adjust_content_focus(self, curriculum: Dict, adjustments: Dict, user_config: Dict) -> Dict:
        """Adjust curriculum content focus based on feedback."""
        action = adjustments.get("action")
        
        if action == "add_practical":
            # Add more practical exercises and real-world applications
            for module in curriculum.get("modules", []):
                module["practical_exercises"] = module.get("practical_exercises", []) + [
                    "hands_on_project", "real_world_application", "code_examples"
                ]
                module["focus"] = "practical_application"
        
        elif action == "add_theoretical":
            # Add more theoretical background and concepts
            for module in curriculum.get("modules", []):
                module["theoretical_background"] = True
                module["concept_depth"] = "comprehensive"
                module["focus"] = "theoretical_understanding"
        
        return curriculum
    
    def _adjust_module_structure(self, curriculum: Dict, adjustments: Dict, user_config: Dict) -> Dict:
        """Adjust curriculum module structure based on feedback."""
        action = adjustments.get("action")
        modules = curriculum.get("modules", [])
        
        if action == "consolidate" and len(modules) > 2:
            # Combine modules to reduce total number
            new_modules = []
            i = 0
            while i < len(modules):
                if i + 1 < len(modules):
                    # Combine two modules
                    combined_module = {
                        "title": f"{modules[i]['title']} & {modules[i+1]['title']}",
                        "description": f"Combined module covering {modules[i]['title']} and {modules[i+1]['title']}",
                        "concepts": modules[i].get("concepts", []) + modules[i+1].get("concepts", []),
                        "learning_outcomes": modules[i].get("learning_outcomes", []) + modules[i+1].get("learning_outcomes", []),
                        "estimated_time": f"{int(modules[i].get('estimated_hours', 0)) + int(modules[i+1].get('estimated_hours', 0))} hours"
                    }
                    new_modules.append(combined_module)
                    i += 2
                else:
                    new_modules.append(modules[i])
                    i += 1
            curriculum["modules"] = new_modules
        
        elif action == "expand":
            # Split modules into smaller units
            new_modules = []
            for module in modules:
                concepts = module.get("concepts", [])
                if len(concepts) > 3:
                    # Split into two modules
                    mid_point = len(concepts) // 2
                    module1 = module.copy()
                    module1["title"] = f"{module['title']} - Part 1"
                    module1["concepts"] = concepts[:mid_point]
                    
                    module2 = module.copy()
                    module2["title"] = f"{module['title']} - Part 2"
                    module2["concepts"] = concepts[mid_point:]
                    
                    new_modules.extend([module1, module2])
                else:
                    new_modules.append(module)
            curriculum["modules"] = new_modules
        
        return curriculum
    
    def _adjust_learning_pace(self, curriculum: Dict, adjustments: Dict, user_config: Dict) -> Dict:
        """Adjust curriculum learning pace based on feedback."""
        action = adjustments.get("action")
        
        for module in curriculum.get("modules", []):
            if action == "slow_down":
                # Increase time estimates, add review sessions
                current_hours = int(module.get("estimated_hours", 2))
                module["estimated_hours"] = str(int(current_hours * 1.3))
                module["include_review_sessions"] = True
                module["practice_time"] = "extended"
            
            elif action == "speed_up":
                # Decrease time estimates, focus on essentials
                current_hours = int(module.get("estimated_hours", 2))
                module["estimated_hours"] = str(max(1, int(current_hours * 0.8)))
                module["focus_on_essentials"] = True
                module["condensed_format"] = True
        
        return curriculum
    
    async def _simulate_user_approval_interaction(self, presentation_result: Dict, user_config: Dict) -> Dict:
        """
        Simulate user approval interaction. In real implementation, this would
        involve actual user interface interaction through CLI or web interface.
        """
        # For simulation purposes, auto-approve based on user config
        user_experience = user_config.get("experience_level", "beginner")
        user_goals = user_config.get("goals", [])
        
        # Simulate approval based on curriculum match with user profile
        satisfaction_score = 0.8  # Base satisfaction
        
        # Adjust satisfaction based on experience level match
        curriculum = presentation_result["curriculum_presentation"]
        if user_experience == "beginner" and curriculum["overview"].get("difficulty_level") == "basic":
            satisfaction_score += 0.1
        elif user_experience == "advanced" and curriculum["overview"].get("difficulty_level") == "advanced":
            satisfaction_score += 0.1
        
        # Check if auto-approval threshold is met
        auto_approve = satisfaction_score >= self.approval_config["auto_approve_threshold"]
        
        if auto_approve:
            return {
                "approved": True,
                "satisfaction_score": satisfaction_score,
                "feedback": {"positive_aspects": ["good_structure", "appropriate_difficulty"]}
            }
        else:
            # Simulate feedback for improvement
            return {
                "approved": False,
                "satisfaction_score": satisfaction_score,
                "feedback": {
                    "difficulty_level": "appropriate" if satisfaction_score > 0.7 else "too_difficult",
                    "content_focus": "good_balance",
                    "module_structure": "good_structure",
                    "learning_pace": "appropriate",
                    "suggestions": ["Add more practical examples", "Include more detailed explanations"]
                }
            }
    
    def _generate_feedback_request(self, curriculum: Dict) -> Dict:
        """Generate specific feedback request when user doesn't provide feedback."""
        return {
            "message": "We'd love to improve this curriculum for you. Please provide feedback on:",
            "specific_questions": [
                "What aspects of the curriculum do you find most/least appealing?",
                "Is the difficulty level appropriate for your experience?",
                "Are there any topics you'd like to see added or removed?",
                "How do you prefer to learn - more theory or hands-on practice?"
            ],
            "quick_options": self._get_feedback_prompts()
        }
    
    def _save_approval_session(self, approval_session: Dict):
        """Save approval session for tracking and analytics."""
        sessions_dir = Path("data/approval_sessions")
        sessions_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = sessions_dir / f"{approval_session['session_id']}.json"
        with open(session_file, "w") as f:
            json.dump(approval_session, f, indent=2)
        
        if self.is_logging:
            logger.info(f"Approval session saved: {session_file}")