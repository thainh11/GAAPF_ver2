"""
Profile Manager - User profile and progress tracking for Study Mode
Manages user profiles, learning progress, and personalization data.

This module provides:
1. User profile loading and saving
2. Quiz result tracking
3. Learning progress analytics
4. Personalization data management
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timezone
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfileManager:
    """
    Manages user profiles and learning progress data.
    
    Handles:
    - User profile persistence
    - Quiz results tracking
    - Learning analytics
    - Personalization settings
    """
    
    def __init__(self, profiles_dir: Path = None, is_logging: bool = False):
        """
        Initialize ProfileManager.
        
        Args:
            profiles_dir: Directory to store user profiles
        """
        if profiles_dir is None:
            profiles_dir = Path("user_profiles")
        
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        self.is_logging = is_logging
        
        # Default profile template
        self.default_profile = {
            "user_id": "default",
            "name": "Learner",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_active": datetime.now(timezone.utc).isoformat(),
            "preferences": {
                "study_mode_preferred": True,
                "difficulty_level": "intermediate",
                "learning_style": "balanced",  # visual, auditory, kinesthetic, balanced
                "feedback_style": "encouraging"  # direct, encouraging, detailed
            },
            "learning_progress": {
                "frameworks": {},
                "concepts_mastered": [],
                "concepts_struggling": [],
                "total_interactions": 0,
                "study_sessions": 0
            },
            "quiz_history": {
                "total_quizzes": 0,
                "total_questions": 0,
                "correct_answers": 0,
                "average_score": 0.0,
                "recent_scores": [],
                "topics_mastered": [],
                "topics_need_review": []
            },
            "code_metrics": {
                "code_snippets_generated": 0,
                "tests_passed": 0,
                "bugs_fixed": 0,
                "average_quality": 0.0,
                "recent_generations": [],
                "recent_fixes": []
            },
            "sync_stats": {
                "total_docs_available": 0,
                "last_sync_time": None
            },
            "achievements": [],
            "settings": {
                "notifications": True,
                "progress_tracking": True,
                "adaptive_difficulty": True
            }
        }
        
        logger.info(f"ProfileManager initialized with profiles directory: {self.profiles_dir}")
    
    def load_profile(self, user_id: str = "default") -> Dict[str, Any]:
        """
        Load user profile from disk.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile dictionary
        """
        profile_path = self.profiles_dir / f"{user_id}.json"
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                
                # Update last active timestamp
                profile["last_active"] = datetime.now(timezone.utc).isoformat()
                
                # Ensure profile has all required fields (migration support)
                profile = self._migrate_profile(profile)
                
                logger.info(f"Loaded profile for user: {user_id}")
                return profile
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading profile for {user_id}: {e}")
                return self._create_new_profile(user_id)
        else:
            logger.info(f"Creating new profile for user: {user_id}")
            return self._create_new_profile(user_id)

    def update_sync_stats(
        self,
        profile: Dict[str, Any],
        num_new_docs: int = 0
    ) -> Dict[str, Any]:
        """Update synchronization statistics in the user profile.
        
        Args:
            profile: User profile to update
            num_new_docs: Number of new document chunks upserted during the sync
        
        Returns:
            Updated profile with sync statistics
        """
        sync_stats = profile.setdefault("sync_stats", {
            "total_docs_available": 0,
            "last_sync_time": None
        })
        
        # Increment total docs and set last sync time
        sync_stats["total_docs_available"] = sync_stats.get("total_docs_available", 0) + num_new_docs
        sync_stats["last_sync_time"] = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Updated sync stats: +{num_new_docs} docs, total={sync_stats['total_docs_available']}")
        return profile
    
    def save_profile(self, profile: Dict[str, Any]) -> bool:
        """
        Save user profile to disk.
        
        Args:
            profile: User profile dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        user_id = profile.get("user_id", "default")
        profile_path = self.profiles_dir / f"{user_id}.json"
        
        try:
            # Update last active timestamp
            profile["last_active"] = datetime.now(timezone.utc).isoformat()
            
            # Create backup if profile exists
            if profile_path.exists():
                backup_path = profile_path.with_suffix('.json.backup')
                profile_path.rename(backup_path)
            
            # Save profile
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved profile for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving profile for {user_id}: {e}")
            return False
    
    def _create_new_profile(self, user_id: str) -> Dict[str, Any]:
        """Create a new user profile with default values."""
        profile = self.default_profile.copy()
        profile["user_id"] = user_id
        profile["created_at"] = datetime.now(timezone.utc).isoformat()
        profile["last_active"] = datetime.now(timezone.utc).isoformat()
        
        # Generate a friendly name based on user_id
        if user_id != "default":
            profile["name"] = f"Learner_{user_id[:8]}"
        
        return profile
    
    def _migrate_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old profile format to current format."""
        # Ensure all required fields exist
        migrated = self.default_profile.copy()
        
        # Update with existing data
        migrated.update(profile)
        
        # Ensure nested dictionaries exist
        for key in ["preferences", "learning_progress", "quiz_history", "settings"]:
            if key not in migrated:
                migrated[key] = self.default_profile[key].copy()
            else:
                # Merge with defaults to ensure all fields exist
                default_section = self.default_profile[key].copy()
                default_section.update(migrated[key])
                migrated[key] = default_section
        
        return migrated
    
    def update_quiz_result(
        self, 
        profile: Dict[str, Any], 
        quiz_results: List[Dict[str, Any]],
        framework: str = "programming"
    ) -> Dict[str, Any]:
        """
        Update profile with quiz results.
        
        Args:
            profile: User profile to update
            quiz_results: List of quiz result dictionaries
            framework: Framework being studied
            
        Returns:
            Updated profile
        """
        if not quiz_results:
            return profile
        
        quiz_history = profile.setdefault("quiz_history", self.default_profile["quiz_history"].copy())
        
        # Calculate quiz statistics
        total_questions = len(quiz_results)
        correct_answers = sum(1 for result in quiz_results if result.get("is_correct", False))
        score_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Update totals
        quiz_history["total_quizzes"] += 1
        quiz_history["total_questions"] += total_questions
        quiz_history["correct_answers"] += correct_answers
        
        # Update average score
        quiz_history["average_score"] = (
            quiz_history["correct_answers"] / quiz_history["total_questions"] * 100
            if quiz_history["total_questions"] > 0 else 0
        )
        
        # Track recent scores (keep last 10)
        recent_scores = quiz_history.setdefault("recent_scores", [])
        recent_scores.append({
            "score": score_percentage,
            "framework": framework,
            "date": datetime.now(timezone.utc).isoformat(),
            "questions": total_questions,
            "correct": correct_answers
        })
        
        # Keep only last 10 scores
        quiz_history["recent_scores"] = recent_scores[-10:]
        
        # Update topic mastery
        topics_mastered = quiz_history.setdefault("topics_mastered", [])
        topics_need_review = quiz_history.setdefault("topics_need_review", [])
        
        for result in quiz_results:
            topic = result.get("topic", "unknown")
            if topic == "unknown":
                continue
                
            if result.get("is_correct", False):
                # Add to mastered if not already there
                if topic not in topics_mastered:
                    topics_mastered.append(topic)
                # Remove from review list if present
                if topic in topics_need_review:
                    topics_need_review.remove(topic)
            else:
                # Add to review list if not already there
                if topic not in topics_need_review:
                    topics_need_review.append(topic)
        
        # Update learning progress
        learning_progress = profile.setdefault("learning_progress", self.default_profile["learning_progress"].copy())
        framework_progress = learning_progress.setdefault("frameworks", {}).setdefault(framework, {
            "total_interactions": 0,
            "quizzes_completed": 0,
            "average_score": 0.0,
            "concepts_learned": [],
            "last_activity": datetime.now(timezone.utc).isoformat()
        })
        
        framework_progress["quizzes_completed"] += 1
        framework_progress["last_activity"] = datetime.now(timezone.utc).isoformat()
        
        # Update framework average score
        framework_scores = [
            score["score"] for score in recent_scores 
            if score["framework"] == framework
        ]
        if framework_scores:
            framework_progress["average_score"] = sum(framework_scores) / len(framework_scores)
        
        logger.info(f"Updated quiz results: {correct_answers}/{total_questions} ({score_percentage:.1f}%)")
        
        return profile
    
    def update_interaction(
        self, 
        profile: Dict[str, Any], 
        framework: str = "programming",
        concepts_covered: List[str] = None
    ) -> Dict[str, Any]:
        """
        Update profile with interaction data.
        
        Args:
            profile: User profile to update
            framework: Framework being studied
            concepts_covered: List of concepts covered in interaction
            
        Returns:
            Updated profile
        """
        if concepts_covered is None:
            concepts_covered = []
        
        # Update total interactions
        learning_progress = profile.setdefault("learning_progress", self.default_profile["learning_progress"].copy())
        learning_progress["total_interactions"] += 1
        
        # Update framework-specific progress
        framework_progress = learning_progress.setdefault("frameworks", {}).setdefault(framework, {
            "total_interactions": 0,
            "quizzes_completed": 0,
            "average_score": 0.0,
            "concepts_learned": [],
            "last_activity": datetime.now(timezone.utc).isoformat()
        })
        
        framework_progress["total_interactions"] += 1
        framework_progress["last_activity"] = datetime.now(timezone.utc).isoformat()
        
        # Add new concepts
        concepts_learned = framework_progress.setdefault("concepts_learned", [])
        for concept in concepts_covered:
            if concept not in concepts_learned:
                concepts_learned.append(concept)
        
        return profile
    
    def update_code_metrics(
        self,
        profile: Dict[str, Any],
        operation_type: str,  # "generate" or "fix"
        success: bool = True,
        quality_score: float = 0.0,
        tests_passed: bool = False,
        specification: str = "",
        file_path: str = ""
    ) -> Dict[str, Any]:
        """
        Update profile with code generation/repair metrics.
        
        Args:
            profile: User profile to update
            operation_type: "generate" or "fix"
            success: Whether the operation was successful
            quality_score: Quality score (0-100)
            tests_passed: Whether tests passed
            specification: Original specification/file path
            file_path: Generated/fixed file path
            
        Returns:
            Updated profile
        """
        code_metrics = profile.setdefault("code_metrics", self.default_profile["code_metrics"].copy())
        
        # Update counters
        if operation_type == "generate" and success:
            code_metrics["code_snippets_generated"] += 1
            
            # Track recent generations (keep last 10)
            recent_generations = code_metrics.setdefault("recent_generations", [])
            recent_generations.append({
                "specification": specification,
                "file_path": file_path,
                "quality_score": quality_score,
                "tests_passed": tests_passed,
                "date": datetime.now(timezone.utc).isoformat()
            })
            code_metrics["recent_generations"] = recent_generations[-10:]
            
        elif operation_type == "fix" and success:
            code_metrics["bugs_fixed"] += 1
            
            # Track recent fixes (keep last 10)
            recent_fixes = code_metrics.setdefault("recent_fixes", [])
            recent_fixes.append({
                "file_path": file_path,
                "quality_score": quality_score,
                "tests_passed": tests_passed,
                "date": datetime.now(timezone.utc).isoformat()
            })
            code_metrics["recent_fixes"] = recent_fixes[-10:]
        
        # Update tests passed counter
        if tests_passed:
            code_metrics["tests_passed"] += 1
        
        # Update average quality score
        all_operations = (
            code_metrics.get("recent_generations", []) +
            code_metrics.get("recent_fixes", [])
        )
        
        if all_operations:
            total_quality = sum(op.get("quality_score", 0) for op in all_operations)
            code_metrics["average_quality"] = total_quality / len(all_operations)
        
        if self.is_logging:
            logger.info(f"Updated code metrics: {operation_type}, success={success}, quality={quality_score}")
        
        return profile
    
    def get_learning_analytics(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate learning analytics from profile data.
        
        Args:
            profile: User profile
            
        Returns:
            Analytics dictionary
        """
        quiz_history = profile.get("quiz_history", {})
        learning_progress = profile.get("learning_progress", {})
        
        # Calculate engagement metrics
        total_interactions = learning_progress.get("total_interactions", 0)
        total_quizzes = quiz_history.get("total_quizzes", 0)
        average_score = quiz_history.get("average_score", 0)
        
        # Calculate learning velocity (interactions per day since creation)
        created_at = profile.get("created_at")
        learning_velocity = 0
        if created_at:
            try:
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                days_active = (datetime.now(timezone.utc) - created_date).days + 1
                learning_velocity = total_interactions / days_active if days_active > 0 else 0
            except ValueError:
                pass
        
        # Analyze strengths and weaknesses
        topics_mastered = quiz_history.get("topics_mastered", [])
        topics_need_review = quiz_history.get("topics_need_review", [])
        
        # Recent performance trend
        recent_scores = quiz_history.get("recent_scores", [])
        performance_trend = "stable"
        if len(recent_scores) >= 3:
            recent_avg = sum(score["score"] for score in recent_scores[-3:]) / 3
            older_avg = sum(score["score"] for score in recent_scores[-6:-3]) / 3 if len(recent_scores) >= 6 else recent_avg
            
            if recent_avg > older_avg + 10:
                performance_trend = "improving"
            elif recent_avg < older_avg - 10:
                performance_trend = "declining"
        
        return {
            "total_interactions": total_interactions,
            "total_quizzes": total_quizzes,
            "average_score": round(average_score, 1),
            "learning_velocity": round(learning_velocity, 2),
            "topics_mastered": len(topics_mastered),
            "topics_need_review": len(topics_need_review),
            "performance_trend": performance_trend,
            "engagement_level": self._calculate_engagement_level(total_interactions, total_quizzes),
            "strengths": topics_mastered[:5],  # Top 5 strengths
            "areas_for_improvement": topics_need_review[:5],  # Top 5 areas to work on
            "recent_activity": recent_scores[-5:] if recent_scores else []  # Last 5 quiz results
        }
    
    def _calculate_engagement_level(self, interactions: int, quizzes: int) -> str:
        """Calculate user engagement level based on activity."""
        total_activity = interactions + (quizzes * 2)  # Quizzes count double
        
        if total_activity >= 50:
            return "highly_engaged"
        elif total_activity >= 20:
            return "moderately_engaged"
        elif total_activity >= 5:
            return "getting_started"
        else:
            return "new_user"
    
    def get_personalization_context(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract personalization context from profile for agents.
        
        Args:
            profile: User profile
            
        Returns:
            Context dictionary for personalizing responses
        """
        preferences = profile.get("preferences", {})
        quiz_history = profile.get("quiz_history", {})
        learning_progress = profile.get("learning_progress", {})
        
        return {
            "user_name": profile.get("name", "Learner"),
            "study_mode_preferred": preferences.get("study_mode_preferred", True),
            "difficulty_level": preferences.get("difficulty_level", "intermediate"),
            "learning_style": preferences.get("learning_style", "balanced"),
            "feedback_style": preferences.get("feedback_style", "encouraging"),
            "average_score": quiz_history.get("average_score", 0),
            "topics_mastered": quiz_history.get("topics_mastered", []),
            "topics_need_review": quiz_history.get("topics_need_review", []),
            "total_interactions": learning_progress.get("total_interactions", 0),
            "engagement_level": self._calculate_engagement_level(
                learning_progress.get("total_interactions", 0),
                quiz_history.get("total_quizzes", 0)
            )
        }