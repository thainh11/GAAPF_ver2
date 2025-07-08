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