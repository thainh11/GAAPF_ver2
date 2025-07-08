"""
Session Manager for GAAPF Architecture

This module provides persistent session management to maintain conversation
continuity across application restarts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages persistent session storage and retrieval for conversation continuity.
    """
    
    def __init__(
        self,
        sessions_path: Optional[Path] = None,
        conversations_path: Optional[Path] = None,
        is_logging: bool = False
    ):
        """
        Initialize the session manager.
        
        Parameters:
        ----------
        sessions_path : Path, optional
            Path to store session data files
        conversations_path : Path, optional  
            Path to store conversation history files
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        self.sessions_path = sessions_path or Path('data/sessions')
        self.conversations_path = conversations_path or Path('data/conversations')
        self.is_logging = is_logging
        
        # Create directories if they don't exist
        self.sessions_path.mkdir(parents=True, exist_ok=True)
        self.conversations_path.mkdir(parents=True, exist_ok=True)
        
        if self.is_logging:
            logger.info("SessionManager initialized")
    
    def get_session_id(self, user_id: str, framework_id: str) -> str:
        """
        Generate deterministic session ID for user+framework combination.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        framework_id : str
            Identifier for the framework
            
        Returns:
        -------
        str
            Session ID
        """
        return f"{user_id}_{framework_id}"
    
    def session_exists(self, user_id: str, framework_id: str) -> bool:
        """
        Check if a session exists for the given user and framework.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        framework_id : str
            Identifier for the framework
            
        Returns:
        -------
        bool
            True if session exists, False otherwise
        """
        session_id = self.get_session_id(user_id, framework_id)
        session_file = self.sessions_path / f"session_{session_id}.json"
        return session_file.exists()
    
    def save_session_state(self, session_data: Dict) -> bool:
        """
        Save session state to persistent storage.
        
        Parameters:
        ----------
        session_data : Dict
            Session data to save
            
        Returns:
        -------
        bool
            True if saved successfully, False otherwise
        """
        try:
            session_id = session_data.get("session_id")
            if not session_id:
                logger.error("No session_id in session data")
                return False
            
            session_file = self.sessions_path / f"session_{session_id}.json"
            
            # Add timestamp
            session_data["last_updated"] = datetime.now().isoformat()
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=4, ensure_ascii=False)
            
            if self.is_logging:
                logger.info(f"Saved session state for {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session state: {e}")
            return False
    
    def load_session_state(self, user_id: str, framework_id: str) -> Optional[Dict]:
        """
        Load session state from persistent storage.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        framework_id : str
            Identifier for the framework
            
        Returns:
        -------
        Dict or None
            Session data if found, None otherwise
        """
        try:
            session_id = self.get_session_id(user_id, framework_id)
            session_file = self.sessions_path / f"session_{session_id}.json"
            
            if not session_file.exists():
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            if self.is_logging:
                logger.info(f"Loaded session state for {session_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"Error loading session state: {e}")
            return None
    
    def save_conversation_history(self, session_id: str, messages: List[Dict]) -> bool:
        """
        Save conversation history for a session.
        
        Parameters:
        ----------
        session_id : str
            Identifier for the session
        messages : List[Dict]
            List of conversation messages
            
        Returns:
        -------
        bool
            True if saved successfully, False otherwise
        """
        try:
            conversation_file = self.conversations_path / f"conversation_{session_id}.json"
            
            conversation_data = {
                "session_id": session_id,
                "messages": messages,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=4, ensure_ascii=False)
            
            if self.is_logging:
                logger.info(f"Saved conversation history for {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
            return False
    
    def load_conversation_history(self, session_id: str) -> List[Dict]:
        """
        Load conversation history for a session.
        
        Parameters:
        ----------
        session_id : str
            Identifier for the session
            
        Returns:
        -------
        List[Dict]
            List of conversation messages
        """
        try:
            conversation_file = self.conversations_path / f"conversation_{session_id}.json"
            
            if not conversation_file.exists():
                return []
            
            with open(conversation_file, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            messages = conversation_data.get("messages", [])
            
            if self.is_logging:
                logger.info(f"Loaded {len(messages)} messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            return []
    
    def delete_session(self, user_id: str, framework_id: str) -> bool:
        """
        Delete session and conversation data.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        framework_id : str
            Identifier for the framework
            
        Returns:
        -------
        bool
            True if deleted successfully, False otherwise
        """
        try:
            session_id = self.get_session_id(user_id, framework_id)
            session_file = self.sessions_path / f"session_{session_id}.json"
            conversation_file = self.conversations_path / f"conversation_{session_id}.json"
            
            success = True
            
            if session_file.exists():
                session_file.unlink()
                if self.is_logging:
                    logger.info(f"Deleted session file for {session_id}")
            
            if conversation_file.exists():
                conversation_file.unlink()
                if self.is_logging:
                    logger.info(f"Deleted conversation file for {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """
        Get all sessions for a user.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
            
        Returns:
        -------
        List[Dict]
            List of session information
        """
        try:
            sessions = []
            for session_file in self.sessions_path.glob(f"session_{user_id}_*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    sessions.append(session_data)
                except Exception as e:
                    logger.error(f"Error reading session file {session_file}: {e}")
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return [] 