# Creating the Analytics Engine

"""
Analytics Engine for GAAPF Architecture

This module provides the AnalyticsEngine class that tracks learning
metrics and provides insights for the GAAPF architecture.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import time
import copy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """
    Tracks learning metrics and provides insights.
    
    The AnalyticsEngine class is responsible for:
    1. Tracking user learning metrics
    2. Analyzing learning effectiveness
    3. Providing insights for optimizing learning experiences
    4. Generating reports on learning progress
    """
    
    def __init__(
        self,
        analytics_path: Optional[Union[Path, str]] = Path('data/analytics'),
        is_logging: bool = False,
        *args, **kwargs
    ):
        """
        Initialize the AnalyticsEngine.
        
        Parameters:
        ----------
        analytics_path : Path or str, optional
            Path to the analytics data directory
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        # Initialize path
        self.analytics_path = Path(analytics_path) if isinstance(analytics_path, str) else analytics_path
        
        # Create directory if it doesn't exist
        self.analytics_path.mkdir(parents=True, exist_ok=True)
        
        self.is_logging = is_logging
        
        # Initialize analytics storage
        self.user_analytics = {}
        self.session_analytics = {}
        
        if self.is_logging:
            logger.info(f"AnalyticsEngine initialized with data path at {self.analytics_path}")
    
    def track_interaction(
        self,
        user_id: str,
        session_id: str,
        learning_context: Dict,
        interaction_data: Dict,
        response: Dict
    ) -> None:
        """
        Track a learning interaction.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        session_id : str
            Identifier for the session
        learning_context : Dict
            Current learning context
        interaction_data : Dict
            Data about the interaction
        response : Dict
            Response data
        """
        # Create interaction record
        timestamp = time.time()
        interaction_id = f"{session_id}_{int(timestamp)}"
        
        # Extract framework and module info
        framework_id = learning_context.get("framework_id", "")
        module_id = learning_context.get("current_module", "")
        
        # Extract metrics from interaction
        metrics = {
            "duration": interaction_data.get("duration", 0),
            "query_length": len(interaction_data.get("query", "")),
            "response_length": len(response.get("content", "")),
            "engagement": 0.5,  # Default value
            "learning_value": 0.5  # Default value
        }
        
        # Create interaction record
        interaction_record = {
            "id": interaction_id,
            "user_id": user_id,
            "session_id": session_id,
            "framework_id": framework_id,
            "module_id": module_id,
            "timestamp": timestamp,
            "type": interaction_data.get("type", "general"),
            "metrics": metrics
        }
        
        # Update session analytics
        self._update_session_analytics(session_id, interaction_record)
        
        # Update user analytics
        self._update_user_analytics(user_id, interaction_record)
        
        if self.is_logging:
            logger.info(f"Tracked interaction {interaction_id} for user {user_id} in session {session_id}")
    
    def get_user_analytics(self, user_id: str) -> Dict:
        """
        Get analytics data for a user.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
            
        Returns:
        -------
        Dict
            User analytics data
        """
        # Check if analytics are in memory
        if user_id in self.user_analytics:
            return copy.deepcopy(self.user_analytics[user_id])
        
        # Load from file
        user_analytics_path = self.analytics_path / "users" / f"{user_id}.json"
        if user_analytics_path.exists():
            try:
                with open(user_analytics_path, "r", encoding="utf-8") as f:
                    analytics = json.load(f)
                    self.user_analytics[user_id] = analytics
                    return copy.deepcopy(analytics)
            except Exception as e:
                if self.is_logging:
                    logger.error(f"Error loading user analytics for {user_id}: {e}")
        
        # Return empty analytics
        return {
            "user_id": user_id,
            "total_sessions": 0,
            "total_interactions": 0,
            "framework_analytics": {}
        }
    
    def get_session_analytics(self, session_id: str) -> Dict:
        """
        Get analytics data for a session.
        
        Parameters:
        ----------
        session_id : str
            Identifier for the session
            
        Returns:
        -------
        Dict
            Session analytics data
        """
        # Check if analytics are in memory
        if session_id in self.session_analytics:
            return copy.deepcopy(self.session_analytics[session_id])
        
        # Load from file
        session_analytics_path = self.analytics_path / "sessions" / f"{session_id}.json"
        if session_analytics_path.exists():
            try:
                with open(session_analytics_path, "r", encoding="utf-8") as f:
                    analytics = json.load(f)
                    self.session_analytics[session_id] = analytics
                    return copy.deepcopy(analytics)
            except Exception as e:
                if self.is_logging:
                    logger.error(f"Error loading session analytics for {session_id}: {e}")
        
        # Return empty analytics
        return {
            "session_id": session_id,
            "total_interactions": 0,
            "duration": 0
        }
    
    def _update_session_analytics(self, session_id: str, interaction_record: Dict) -> None:
        """
        Update session analytics with an interaction.
        
        Parameters:
        ----------
        session_id : str
            Identifier for the session
        interaction_record : Dict
            Interaction record
        """
        # Get current session analytics
        if session_id in self.session_analytics:
            session_analytics = self.session_analytics[session_id]
        else:
            # Initialize session analytics
            session_analytics = {
                "session_id": session_id,
                "user_id": interaction_record["user_id"],
                "framework_id": interaction_record["framework_id"],
                "module_id": interaction_record["module_id"],
                "start_time": interaction_record["timestamp"],
                "end_time": interaction_record["timestamp"],
                "total_interactions": 0,
                "metrics": {
                    "engagement": 0,
                    "learning_value": 0
                }
            }
        
        # Update session analytics
        session_analytics["total_interactions"] += 1
        session_analytics["end_time"] = interaction_record["timestamp"]
        
        # Update metrics
        metrics = session_analytics["metrics"]
        interaction_metrics = interaction_record["metrics"]
        
        # Update engagement (rolling average)
        metrics["engagement"] = (
            (metrics["engagement"] * (session_analytics["total_interactions"] - 1) +
             interaction_metrics["engagement"]) / session_analytics["total_interactions"]
        )
        
        # Update learning value (rolling average)
        metrics["learning_value"] = (
            (metrics["learning_value"] * (session_analytics["total_interactions"] - 1) +
             interaction_metrics["learning_value"]) / session_analytics["total_interactions"]
        )
        
        # Store updated session analytics
        self.session_analytics[session_id] = session_analytics
        
        # Save to file
        self._save_session_analytics(session_id, session_analytics)
    
    def _update_user_analytics(self, user_id: str, interaction_record: Dict) -> None:
        """
        Update user analytics with an interaction.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        interaction_record : Dict
            Interaction record
        """
        # Get current user analytics
        if user_id in self.user_analytics:
            user_analytics = self.user_analytics[user_id]
        else:
            # Initialize user analytics
            user_analytics = {
                "user_id": user_id,
                "total_sessions": 0,
                "total_interactions": 0,
                "first_interaction": interaction_record["timestamp"],
                "last_interaction": interaction_record["timestamp"],
                "framework_analytics": {},
                "metrics": {
                    "engagement": 0,
                    "learning_value": 0
                }
            }
        
        # Update user analytics
        user_analytics["total_interactions"] += 1
        user_analytics["last_interaction"] = interaction_record["timestamp"]
        
        # Get framework ID
        framework_id = interaction_record["framework_id"]
        
        # Initialize framework analytics if not exists
        if framework_id not in user_analytics["framework_analytics"]:
            user_analytics["framework_analytics"][framework_id] = {
                "total_interactions": 0,
                "modules": {},
                "metrics": {
                    "engagement": 0,
                    "learning_value": 0
                }
            }
        
        # Update framework analytics
        framework_analytics = user_analytics["framework_analytics"][framework_id]
        framework_analytics["total_interactions"] += 1
        
        # Get module ID
        module_id = interaction_record["module_id"]
        
        # Initialize module analytics if not exists
        if module_id and module_id not in framework_analytics["modules"]:
            framework_analytics["modules"][module_id] = {
                "total_interactions": 0,
                "metrics": {
                    "engagement": 0,
                    "learning_value": 0
                }
            }
        
        # Update module analytics
        if module_id:
            module_analytics = framework_analytics["modules"][module_id]
            module_analytics["total_interactions"] += 1
            
            # Update module metrics
            module_metrics = module_analytics["metrics"]
            interaction_metrics = interaction_record["metrics"]
            
            # Update engagement (rolling average)
            module_metrics["engagement"] = (
                (module_metrics["engagement"] * (module_analytics["total_interactions"] - 1) +
                 interaction_metrics["engagement"]) / module_analytics["total_interactions"]
            )
            
            # Update learning value (rolling average)
            module_metrics["learning_value"] = (
                (module_metrics["learning_value"] * (module_analytics["total_interactions"] - 1) +
                 interaction_metrics["learning_value"]) / module_analytics["total_interactions"]
            )
        
        # Update framework metrics
        framework_metrics = framework_analytics["metrics"]
        interaction_metrics = interaction_record["metrics"]
        
        # Update engagement (rolling average)
        framework_metrics["engagement"] = (
            (framework_metrics["engagement"] * (framework_analytics["total_interactions"] - 1) +
             interaction_metrics["engagement"]) / framework_analytics["total_interactions"]
        )
        
        # Update learning value (rolling average)
        framework_metrics["learning_value"] = (
            (framework_metrics["learning_value"] * (framework_analytics["total_interactions"] - 1) +
             interaction_metrics["learning_value"]) / framework_analytics["total_interactions"]
        )
        
        # Update overall user metrics
        user_metrics = user_analytics["metrics"]
        
        # Update engagement (rolling average)
        user_metrics["engagement"] = (
            (user_metrics["engagement"] * (user_analytics["total_interactions"] - 1) +
             interaction_metrics["engagement"]) / user_analytics["total_interactions"]
        )
        
        # Update learning value (rolling average)
        user_metrics["learning_value"] = (
            (user_metrics["learning_value"] * (user_analytics["total_interactions"] - 1) +
             interaction_metrics["learning_value"]) / user_analytics["total_interactions"]
        )
        
        # Store updated user analytics
        self.user_analytics[user_id] = user_analytics
        
        # Save to file
        self._save_user_analytics(user_id, user_analytics)
    
    def _save_session_analytics(self, session_id: str, session_analytics: Dict) -> None:
        """
        Save session analytics to file.
        
        Parameters:
        ----------
        session_id : str
            Identifier for the session
        session_analytics : Dict
            Session analytics data
        """
        # Create sessions directory if it doesn't exist
        sessions_dir = self.analytics_path / "sessions"
        sessions_dir.mkdir(exist_ok=True)
        
        # Save to file
        session_path = sessions_dir / f"{session_id}.json"
        try:
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session_analytics, f, indent=2)
        except Exception as e:
            if self.is_logging:
                logger.error(f"Error saving session analytics for {session_id}: {e}")
    
    def _save_user_analytics(self, user_id: str, user_analytics: Dict) -> None:
        """
        Save user analytics to file.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        user_analytics : Dict
            User analytics data
        """
        # Create users directory if it doesn't exist
        users_dir = self.analytics_path / "users"
        users_dir.mkdir(exist_ok=True)
        
        # Save to file
        user_path = users_dir / f"{user_id}.json"
        try:
            with open(user_path, "w", encoding="utf-8") as f:
                json.dump(user_analytics, f, indent=2)
        except Exception as e:
            if self.is_logging:
                logger.error(f"Error saving user analytics for {user_id}: {e}")
