import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserProfile:
    """
    User profile models and management system.
    
    The UserProfile class is responsible for:
    1. Storing learning preferences, experience levels, and goals
    2. Managing user profile data persistence
    3. Providing methods for profile updates and retrieval
    """
    
    def __init__(
        self,
        user_profiles_path: Optional[Union[Path, str]] = Path('user_profiles'),
        is_logging: bool = False,
        *args, **kwargs
    ):
        """
        Initialize the UserProfile manager with data path.
        
        Parameters:
        ----------
        user_profiles_path : Path or str
            Path to the directory for storing user profile data
        is_logging : bool
            Flag to enable detailed logging
        """
        # Initialize path
        if isinstance(user_profiles_path, str):
            self.user_profiles_path = Path(user_profiles_path)
        else:
            self.user_profiles_path = user_profiles_path
        
        # Create directory if it doesn't exist
        self.user_profiles_path.mkdir(parents=True, exist_ok=True)
        
        self.is_logging = is_logging
        
        # In-memory cache of profiles
        self.profile_cache = {}
        
        if self.is_logging:
            logger.info(f"UserProfile manager initialized with data at {self.user_profiles_path}")
    
    def create_profile(
        self,
        user_id: str,
        name: str = None,
        email: str = None,
        experience_level: str = "beginner",
        learning_style: Dict = None,
        goals: List[str] = None,
        frameworks: List[str] = None
    ) -> Dict:
        """
        Create a new user profile.
        
        Parameters:
        ----------
        user_id : str
            Unique identifier for the user
        name : str, optional
            User's name
        email : str, optional
            User's email
        experience_level : str, optional
            User's experience level (beginner, intermediate, advanced)
        learning_style : Dict, optional
            User's learning style preferences
        goals : List[str], optional
            User's learning goals
        frameworks : List[str], optional
            Frameworks the user is interested in
            
        Returns:
        -------
        Dict
            The created user profile
        """
        # Set defaults
        if learning_style is None:
            learning_style = {
                "preferred_mode": "balanced",
                "pace": "moderate",
                "interaction_style": "guided",
                "detail_level": "balanced"
            }
        
        if goals is None:
            goals = []
        
        if frameworks is None:
            frameworks = []
        
        # Create profile
        profile = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "experience_level": experience_level,
            "learning_style": learning_style,
            "goals": goals,
            "frameworks": frameworks,
            "progress": {},
            "feedback_history": [],
            "learning_analytics": {
                "total_sessions": 0,
                "total_interactions": 0,
                "average_session_length": 0,
                "preferred_learning_times": [],
                "engagement_patterns": {},
                "learning_velocity": "moderate",
                "concept_mastery_rate": 0.0
            },
            "preferences": {
                "notification_settings": {
                    "progress_updates": True,
                    "learning_reminders": True,
                    "achievement_notifications": True
                },
                "ui_preferences": {
                    "theme": "default",
                    "language": "en",
                    "accessibility_features": []
                }
            },
            "last_session": None,
            "last_session_time": None,
            "created_at": int(time.time()),
            "last_updated": int(time.time())
        }
        
        # Save profile
        self._save_profile(user_id, profile)
        
        # Update cache
        self.profile_cache[user_id] = profile
        
        if self.is_logging:
            logger.info(f"Created profile for user {user_id}")
        
        return profile
    
    def get_profile(self, user_id: str) -> Optional[Dict]:
        """
        Get a user profile by ID.
        
        Parameters:
        ----------
        user_id : str
            Unique identifier for the user
            
        Returns:
        -------
        Optional[Dict]
            The user profile or None if not found
        """
        # Check cache first
        if user_id in self.profile_cache:
            return self.profile_cache[user_id]
        
        # Try to load from file
        profile_path = self.user_profiles_path / f"{user_id}.json"
        
        if not profile_path.exists():
            if self.is_logging:
                logger.warning(f"Profile for user {user_id} not found")
            return None
        
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
                
                # Update cache
                self.profile_cache[user_id] = profile
                
                return profile
        except Exception as e:
            logger.error(f"Error loading profile for user {user_id}: {e}")
            return None
    
    def update_profile(self, user_id: str, updates: Dict) -> Optional[Dict]:
        """
        Update a user profile.
        
        Parameters:
        ----------
        user_id : str
            Unique identifier for the user
        updates : Dict
            Dictionary of profile fields to update
            
        Returns:
        -------
        Optional[Dict]
            The updated profile or None if not found
        """
        # Get current profile
        profile = self.get_profile(user_id)
        
        if profile is None:
            if self.is_logging:
                logger.warning(f"Cannot update profile for user {user_id}: profile not found")
            return None
        
        # Apply updates
        for key, value in updates.items():
            if key in profile:
                # Handle nested dictionaries
                if isinstance(profile[key], dict) and isinstance(value, dict):
                    profile[key].update(value)
                else:
                    profile[key] = value
        
        # Update timestamp
        profile["last_updated"] = int(time.time())
        
        # Save updated profile
        self._save_profile(user_id, profile)
        
        # Update cache
        self.profile_cache[user_id] = profile
        
        if self.is_logging:
            logger.info(f"Updated profile for user {user_id}")
        
        return profile
    
    def update_learning_progress(
        self,
        user_id: str,
        framework_id: str,
        module_id: str = None,
        completion_percentage: float = None,
        mastered_concepts: List[str] = None
    ) -> Optional[Dict]:
        """
        Update learning progress for a user.
        
        Parameters:
        ----------
        user_id : str
            Unique identifier for the user
        framework_id : str
            Framework identifier
        module_id : str, optional
            Module identifier
        completion_percentage : float, optional
            Completion percentage (0-100)
        mastered_concepts : List[str], optional
            List of mastered concepts
            
        Returns:
        -------
        Optional[Dict]
            The updated profile or None if not found
        """
        # Get current profile
        profile = self.get_profile(user_id)
        
        if profile is None:
            return None
        
        # Initialize progress if not present
        if "progress" not in profile:
            profile["progress"] = {}
        
        if framework_id not in profile["progress"]:
            profile["progress"][framework_id] = {}
        
        framework_progress = profile["progress"][framework_id]
        
        # Update module progress if specified
        if module_id:
            if module_id not in framework_progress:
                framework_progress[module_id] = {}
            
            module_progress = framework_progress[module_id]
            
            if completion_percentage is not None:
                module_progress["completion_percentage"] = completion_percentage
            
            if mastered_concepts is not None:
                if "mastered_concepts" not in module_progress:
                    module_progress["mastered_concepts"] = []
                
                # Add new concepts
                for concept in mastered_concepts:
                    if concept not in module_progress["mastered_concepts"]:
                        module_progress["mastered_concepts"].append(concept)
        
        # Update overall framework progress
        if "overall_completion" not in framework_progress:
            framework_progress["overall_completion"] = 0
        
        # Save profile
        profile["last_updated"] = int(time.time())
        self._save_profile(user_id, profile)
        
        # Update cache
        self.profile_cache[user_id] = profile
        
        if self.is_logging:
            logger.info(f"Updated learning progress for user {user_id} in framework {framework_id}")
        
        return profile
    
    def get_all_users(self) -> List[str]:
        """
        Get all user IDs.
        
        Returns:
        -------
        List[str]
            List of all user IDs
        """
        user_ids = []
        
        # Get from cache
        user_ids.extend(self.profile_cache.keys())
        
        # Get from files
        for profile_file in self.user_profiles_path.glob("*.json"):
            user_id = profile_file.stem
            if user_id not in user_ids:
                user_ids.append(user_id)
        
        return user_ids
    
    def store_feedback(self, user_id: str, feedback: Dict) -> Optional[Dict]:
        """
        Store user feedback for learning improvement.
        
        Parameters:
        ----------
        user_id : str
            Unique identifier for the user
        feedback : Dict
            Feedback data containing rating, comments, suggestions, etc.
            
        Returns:
        -------
        Optional[Dict]
            The updated profile or None if not found
        """
        profile = self.get_profile(user_id)
        
        if profile is None:
            if self.is_logging:
                logger.warning(f"Cannot store feedback for user {user_id}: profile not found")
            return None
        
        # Initialize feedback history if not present
        if "feedback_history" not in profile:
            profile["feedback_history"] = []
        
        # Add timestamp and process feedback
        feedback_entry = {
            "timestamp": int(time.time()),
            "session_id": feedback.get("session_id"),
            "feedback_type": feedback.get("type", "general"),
            "rating": feedback.get("rating"),
            "comments": feedback.get("comments", ""),
            "suggestions": feedback.get("suggestions", []),
            "difficulty_rating": feedback.get("difficulty_rating"),
            "pace_rating": feedback.get("pace_rating"),
            "content_relevance": feedback.get("content_relevance"),
            "learning_effectiveness": feedback.get("learning_effectiveness"),
            "areas_for_improvement": feedback.get("areas_for_improvement", []),
            "positive_aspects": feedback.get("positive_aspects", []),
            "context": {
                "current_module": feedback.get("current_module"),
                "learning_stage": feedback.get("learning_stage"),
                "interaction_count": feedback.get("interaction_count", 0)
            }
        }
        
        # Store feedback
        profile["feedback_history"].append(feedback_entry)
        
        # Analyze feedback for profile improvements
        self._analyze_and_apply_feedback(profile, feedback_entry)
        
        # Update profile
        profile["last_updated"] = int(time.time())
        self._save_profile(user_id, profile)
        self.profile_cache[user_id] = profile
        
        if self.is_logging:
            logger.info(f"Stored feedback for user {user_id}")
        
        return profile
    
    def get_feedback_summary(self, user_id: str, limit: int = 10) -> Optional[Dict]:
        """
        Get a summary of user feedback.
        
        Parameters:
        ----------
        user_id : str
            Unique identifier for the user
        limit : int
            Maximum number of recent feedback entries to include
            
        Returns:
        -------
        Optional[Dict]
            Feedback summary or None if profile not found
        """
        profile = self.get_profile(user_id)
        
        if profile is None:
            return None
        
        feedback_history = profile.get("feedback_history", [])
        
        if not feedback_history:
            return {
                "total_feedback_count": 0,
                "recent_feedback": [],
                "average_rating": None,
                "common_themes": []
            }
        
        # Get recent feedback
        recent_feedback = sorted(feedback_history, key=lambda x: x["timestamp"], reverse=True)[:limit]
        
        # Calculate statistics
        ratings = [f["rating"] for f in feedback_history if f.get("rating") is not None]
        average_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Extract common themes
        all_comments = [f.get("comments", "") for f in feedback_history if f.get("comments")]
        all_suggestions = []
        for f in feedback_history:
            all_suggestions.extend(f.get("suggestions", []))
        
        return {
            "total_feedback_count": len(feedback_history),
            "recent_feedback": recent_feedback,
            "average_rating": average_rating,
            "rating_distribution": self._calculate_rating_distribution(ratings),
            "common_suggestions": self._extract_common_themes(all_suggestions),
            "improvement_areas": self._extract_improvement_areas(feedback_history)
        }
    
    def update_learning_analytics(self, user_id: str, session_data: Dict) -> Optional[Dict]:
        """
        Update learning analytics based on session data.
        
        Parameters:
        ----------
        user_id : str
            Unique identifier for the user
        session_data : Dict
            Session data including duration, interactions, etc.
            
        Returns:
        -------
        Optional[Dict]
            The updated profile or None if not found
        """
        profile = self.get_profile(user_id)
        
        if profile is None:
            return None
        
        # Initialize analytics if not present
        if "learning_analytics" not in profile:
            profile["learning_analytics"] = {
                "total_sessions": 0,
                "total_interactions": 0,
                "average_session_length": 0,
                "preferred_learning_times": [],
                "engagement_patterns": {},
                "learning_velocity": "moderate",
                "concept_mastery_rate": 0.0
            }
        
        analytics = profile["learning_analytics"]
        
        # Update session count
        analytics["total_sessions"] += 1
        
        # Update interaction count
        interaction_count = session_data.get("interaction_count", 0)
        analytics["total_interactions"] += interaction_count
        
        # Update average session length
        session_duration = session_data.get("duration_minutes", 0)
        current_avg = analytics.get("average_session_length", 0)
        total_sessions = analytics["total_sessions"]
        analytics["average_session_length"] = ((current_avg * (total_sessions - 1)) + session_duration) / total_sessions
        
        # Track learning times
        session_time = session_data.get("start_time")
        if session_time:
            hour = time.localtime(session_time).tm_hour
            preferred_times = analytics.get("preferred_learning_times", [])
            preferred_times.append(hour)
            analytics["preferred_learning_times"] = preferred_times[-50:]  # Keep last 50 sessions
        
        # Update engagement patterns
        engagement_level = session_data.get("engagement_level", "moderate")
        engagement_patterns = analytics.get("engagement_patterns", {})
        engagement_patterns[engagement_level] = engagement_patterns.get(engagement_level, 0) + 1
        analytics["engagement_patterns"] = engagement_patterns
        
        # Update learning velocity
        concepts_learned = session_data.get("concepts_learned", 0)
        if concepts_learned > 0:
            velocity_score = concepts_learned / max(session_duration, 1)  # concepts per minute
            if velocity_score > 0.5:
                analytics["learning_velocity"] = "fast"
            elif velocity_score > 0.2:
                analytics["learning_velocity"] = "moderate"
            else:
                analytics["learning_velocity"] = "slow"
        
        # Update concept mastery rate
        mastery_assessments = session_data.get("mastery_assessments", [])
        if mastery_assessments:
            mastery_scores = [a.get("score", 0) for a in mastery_assessments]
            analytics["concept_mastery_rate"] = sum(mastery_scores) / len(mastery_scores)
        
        # Save profile
        profile["last_updated"] = int(time.time())
        self._save_profile(user_id, profile)
        self.profile_cache[user_id] = profile
        
        if self.is_logging:
            logger.info(f"Updated learning analytics for user {user_id}")
        
        return profile
    
    def _analyze_and_apply_feedback(self, profile: Dict, feedback: Dict) -> None:
        """
        Analyze feedback and apply improvements to user profile.
        
        Parameters:
        ----------
        profile : Dict
            User profile to update
        feedback : Dict
            Feedback entry to analyze
        """
        # Adjust learning style based on feedback
        if feedback.get("pace_rating"):
            pace_rating = feedback["pace_rating"]
            current_pace = profile["learning_style"].get("pace", "moderate")
            
            if pace_rating < 3 and current_pace != "slow":
                profile["learning_style"]["pace"] = "slow"
            elif pace_rating > 4 and current_pace != "fast":
                profile["learning_style"]["pace"] = "fast"
        
        # Adjust detail level based on feedback
        if feedback.get("difficulty_rating"):
            difficulty = feedback["difficulty_rating"]
            current_detail = profile["learning_style"].get("detail_level", "balanced")
            
            if difficulty < 3 and current_detail != "detailed":
                profile["learning_style"]["detail_level"] = "detailed"
            elif difficulty > 4 and current_detail != "concise":
                profile["learning_style"]["detail_level"] = "concise"
        
        # Update experience level based on consistent high performance
        if feedback.get("learning_effectiveness", 0) >= 4:
            current_level = profile.get("experience_level", "beginner")
            if current_level == "beginner":
                # Check if user consistently performs well
                recent_feedback = profile.get("feedback_history", [])[-5:]
                high_performance_count = sum(1 for f in recent_feedback 
                                            if f.get("learning_effectiveness", 0) >= 4)
                if high_performance_count >= 3:
                    profile["experience_level"] = "intermediate"
    
    def _calculate_rating_distribution(self, ratings: List[float]) -> Dict:
        """
        Calculate distribution of ratings.
        
        Parameters:
        ----------
        ratings : List[float]
            List of rating values
            
        Returns:
        -------
        Dict
            Rating distribution
        """
        if not ratings:
            return {}
        
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for rating in ratings:
            rounded_rating = round(rating)
            if 1 <= rounded_rating <= 5:
                distribution[rounded_rating] += 1
        
        total = len(ratings)
        return {k: v/total for k, v in distribution.items()}
    
    def _extract_common_themes(self, suggestions: List[str]) -> List[str]:
        """
        Extract common themes from suggestions.
        
        Parameters:
        ----------
        suggestions : List[str]
            List of suggestion strings
            
        Returns:
        -------
        List[str]
            Common themes
        """
        if not suggestions:
            return []
        
        # Simple keyword-based theme extraction
        theme_keywords = {
            "pace": ["slow", "fast", "speed", "pace", "quick", "rushed"],
            "examples": ["example", "demo", "sample", "illustration"],
            "explanation": ["explain", "clarify", "detail", "elaborate"],
            "practice": ["practice", "exercise", "hands-on", "coding"],
            "difficulty": ["easy", "hard", "difficult", "simple", "complex"]
        }
        
        theme_counts = {theme: 0 for theme in theme_keywords}
        
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in suggestion_lower for keyword in keywords):
                    theme_counts[theme] += 1
        
        # Return themes with at least 2 mentions
        return [theme for theme, count in theme_counts.items() if count >= 2]
    
    def _extract_improvement_areas(self, feedback_history: List[Dict]) -> List[str]:
        """
        Extract areas for improvement from feedback history.
        
        Parameters:
        ----------
        feedback_history : List[Dict]
            List of feedback entries
            
        Returns:
        -------
        List[str]
            Areas for improvement
        """
        improvement_areas = []
        
        for feedback in feedback_history:
            areas = feedback.get("areas_for_improvement", [])
            improvement_areas.extend(areas)
        
        # Count occurrences and return most common
        area_counts = {}
        for area in improvement_areas:
            area_counts[area] = area_counts.get(area, 0) + 1
        
        # Return areas mentioned at least twice
        return [area for area, count in area_counts.items() if count >= 2]
    
    def _save_profile(self, user_id: str, profile: Dict) -> None:
        """
        Save a user profile to file.
        
        Parameters:
        ----------
        user_id : str
            Unique identifier for the user
        profile : Dict
            The profile data to save
        """
        profile_path = self.user_profiles_path / f"{user_id}.json"
        
        try:
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving profile for user {user_id}: {e}")

# Alias for backwards compatibility
UserProfileManager = UserProfile