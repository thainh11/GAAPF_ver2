"""Configuration management for GAAPF.

This module provides centralized configuration management for the GAAPF system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

class Settings:
    """Centralized configuration management for GAAPF."""
    
    def __init__(self):
        """Initialize settings with default values."""
        self.base_dir = Path(__file__).parent.parent.parent.parent
        self._load_settings()
    
    def _load_settings(self):
        """Load settings from environment variables and config files."""
        # Database and storage paths
        self.memory_base_path = self.base_dir / "memory"
        self.user_profiles_path = self.base_dir / "user_profiles"
        self.frameworks_path = self.base_dir / "frameworks"
        self.data_path = self.base_dir / "data"
        
        # LLM Configuration
        self.llm_config = {
            "model_name": os.getenv("GAAPF_LLM_MODEL", "gpt-3.5-turbo"),
            "temperature": float(os.getenv("GAAPF_LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("GAAPF_LLM_MAX_TOKENS", "2048")),
            "timeout": int(os.getenv("GAAPF_LLM_TIMEOUT", "30"))
        }
        
        # Agent Configuration
        self.agent_config = {
            "max_handoffs": int(os.getenv("GAAPF_MAX_HANDOFFS", "3")),
            "agent_timeout": int(os.getenv("GAAPF_AGENT_TIMEOUT", "60")),
            "memory_limit": int(os.getenv("GAAPF_MEMORY_LIMIT", "1000"))
        }
        
        # Learning Configuration
        self.learning_config = {
            "interaction_threshold_low": int(os.getenv("GAAPF_INTERACTION_THRESHOLD_LOW", "3")),
            "interaction_threshold_high": int(os.getenv("GAAPF_INTERACTION_THRESHOLD_HIGH", "8")),
            "session_timeout": int(os.getenv("GAAPF_SESSION_TIMEOUT", "3600")),
            "auto_save_interval": int(os.getenv("GAAPF_AUTO_SAVE_INTERVAL", "300"))
        }
        
        # Logging Configuration
        self.logging_config = {
            "level": os.getenv("GAAPF_LOG_LEVEL", "INFO"),
            "file_path": self.base_dir / "logs" / "gaapf.log",
            "max_file_size": int(os.getenv("GAAPF_LOG_MAX_SIZE", "10485760")),  # 10MB
            "backup_count": int(os.getenv("GAAPF_LOG_BACKUP_COUNT", "5"))
        }
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.memory_base_path,
            self.user_profiles_path,
            self.data_path,
            self.logging_config["file_path"].parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_memory_path(self, agent_type: str, user_id: str = "default") -> Path:
        """Get memory file path for a specific agent and user."""
        sanitized_agent_type = agent_type.strip().lower().replace(" ", "_")
        filename = f"{sanitized_agent_type}_memory.json"
        return self.memory_base_path / filename
    
    def get_user_profile_path(self, user_id: str) -> Path:
        """Get user profile file path."""
        return self.user_profiles_path / f"{user_id}.json"
    
    def get_framework_path(self, framework_name: str) -> Path:
        """Get framework configuration file path."""
        return self.frameworks_path / f"{framework_name}.json"
    
    def update_setting(self, section: str, key: str, value: Any):
        """Update a specific setting."""
        if hasattr(self, section):
            config = getattr(self, section)
            if isinstance(config, dict):
                config[key] = value
    
    def get_setting(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        if hasattr(self, section):
            config = getattr(self, section)
            if isinstance(config, dict):
                return config.get(key, default)
        return default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "llm_config": self.llm_config,
            "agent_config": self.agent_config,
            "learning_config": self.learning_config,
            "logging_config": {
                **self.logging_config,
                "file_path": str(self.logging_config["file_path"])
            },
            "paths": {
                "memory_base_path": str(self.memory_base_path),
                "user_profiles_path": str(self.user_profiles_path),
                "frameworks_path": str(self.frameworks_path),
                "data_path": str(self.data_path)
            }
        }

# Global settings instance
settings = Settings()