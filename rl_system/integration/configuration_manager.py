"""Configuration Manager for RL System Integration

This module provides centralized configuration management for the RL system,
handling settings, validation, and dynamic configuration updates.
"""

import os
import json
import yaml
import logging
import time
from typing import Dict, List, Any, Optional, Union, Type, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"

class ConfigScope(Enum):
    """Configuration scopes"""
    GLOBAL = "global"
    USER = "user"
    SESSION = "session"
    AGENT = "agent"
    TEMPORARY = "temporary"

class ValidationLevel(Enum):
    """Configuration validation levels"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    CUSTOM = "custom"

@dataclass
class ConfigMetadata:
    """Metadata for configuration entries"""
    key: str
    scope: ConfigScope
    data_type: str
    description: str = ""
    default_value: Any = None
    required: bool = False
    validation_rules: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    tags: List[str] = field(default_factory=list)

@dataclass
class ConfigEntry:
    """Configuration entry with metadata"""
    metadata: ConfigMetadata
    value: Any
    source: str = "default"
    locked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'metadata': asdict(self.metadata),
            'value': self.value,
            'source': self.source,
            'locked': self.locked
        }

class ConfigValidator(ABC):
    """Abstract base class for configuration validators"""
    
    @abstractmethod
    def validate(self, key: str, value: Any, metadata: ConfigMetadata) -> bool:
        """Validate configuration value"""
        pass
    
    @abstractmethod
    def get_error_message(self) -> str:
        """Get validation error message"""
        pass

class TypeValidator(ConfigValidator):
    """Type-based validator"""
    
    def __init__(self, expected_type: Type):
        self.expected_type = expected_type
        self.error_message = ""
    
    def validate(self, key: str, value: Any, metadata: ConfigMetadata) -> bool:
        """Validate type"""
        if not isinstance(value, self.expected_type):
            self.error_message = f"Expected {self.expected_type.__name__}, got {type(value).__name__}"
            return False
        return True
    
    def get_error_message(self) -> str:
        return self.error_message

class RangeValidator(ConfigValidator):
    """Range-based validator for numeric values"""
    
    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None):
        self.min_value = min_value
        self.max_value = max_value
        self.error_message = ""
    
    def validate(self, key: str, value: Any, metadata: ConfigMetadata) -> bool:
        """Validate range"""
        if not isinstance(value, (int, float)):
            self.error_message = "Value must be numeric for range validation"
            return False
        
        if self.min_value is not None and value < self.min_value:
            self.error_message = f"Value {value} is below minimum {self.min_value}"
            return False
        
        if self.max_value is not None and value > self.max_value:
            self.error_message = f"Value {value} is above maximum {self.max_value}"
            return False
        
        return True
    
    def get_error_message(self) -> str:
        return self.error_message

class ChoiceValidator(ConfigValidator):
    """Choice-based validator"""
    
    def __init__(self, choices: List[Any]):
        self.choices = choices
        self.error_message = ""
    
    def validate(self, key: str, value: Any, metadata: ConfigMetadata) -> bool:
        """Validate choice"""
        if value not in self.choices:
            self.error_message = f"Value {value} not in allowed choices: {self.choices}"
            return False
        return True
    
    def get_error_message(self) -> str:
        return self.error_message

class RegexValidator(ConfigValidator):
    """Regex-based validator"""
    
    def __init__(self, pattern: str):
        import re
        self.pattern = re.compile(pattern)
        self.pattern_str = pattern
        self.error_message = ""
    
    def validate(self, key: str, value: Any, metadata: ConfigMetadata) -> bool:
        """Validate regex pattern"""
        if not isinstance(value, str):
            self.error_message = "Value must be string for regex validation"
            return False
        
        if not self.pattern.match(value):
            self.error_message = f"Value '{value}' does not match pattern '{self.pattern_str}'"
            return False
        
        return True
    
    def get_error_message(self) -> str:
        return self.error_message

class ConfigurationManager:
    """Centralized configuration manager for RL system"""
    
    def __init__(self, config_dir: str = "config", validation_level: ValidationLevel = ValidationLevel.BASIC):
        """
        Initialize configuration manager.
        
        Parameters:
        ----------
        config_dir : str
            Directory for configuration files
        validation_level : ValidationLevel
            Level of configuration validation
        """
        self.config_dir = Path(config_dir)
        self.validation_level = validation_level
        
        # Configuration storage
        self.configurations: Dict[str, ConfigEntry] = {}
        self.metadata_registry: Dict[str, ConfigMetadata] = {}
        self.validators: Dict[str, List[ConfigValidator]] = {}
        
        # Change tracking
        self.change_listeners: List[Callable[[str, Any, Any], None]] = []
        self.change_history: List[Dict[str, Any]] = []
        
        # File watching
        self.file_watchers: Dict[str, float] = {}  # file_path -> last_modified
        self.auto_reload = True
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Create config directory
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default configurations
        self._initialize_default_configurations()
        
        logger.info(f"Initialized configuration manager with directory: {self.config_dir}")
    
    def _initialize_default_configurations(self) -> None:
        """Initialize default configuration schema"""
        # RL System Core Settings
        self.register_config(
            "rl.enabled",
            ConfigScope.GLOBAL,
            bool,
            "Enable RL system",
            default_value=True,
            validators=[TypeValidator(bool)]
        )
        
        self.register_config(
            "rl.algorithm",
            ConfigScope.GLOBAL,
            str,
            "RL algorithm to use",
            default_value="maddpg",
            validators=[
                TypeValidator(str),
                ChoiceValidator(["maddpg", "dqn", "policy_gradient", "ppo", "trpo"])
            ]
        )
        
        self.register_config(
            "rl.learning_rate",
            ConfigScope.GLOBAL,
            float,
            "Learning rate for RL algorithms",
            default_value=0.001,
            validators=[
                TypeValidator(float),
                RangeValidator(min_value=1e-6, max_value=1.0)
            ]
        )
        
        self.register_config(
            "rl.exploration_rate",
            ConfigScope.GLOBAL,
            float,
            "Exploration rate (epsilon)",
            default_value=0.1,
            validators=[
                TypeValidator(float),
                RangeValidator(min_value=0.0, max_value=1.0)
            ]
        )
        
        self.register_config(
            "rl.discount_factor",
            ConfigScope.GLOBAL,
            float,
            "Discount factor (gamma)",
            default_value=0.99,
            validators=[
                TypeValidator(float),
                RangeValidator(min_value=0.0, max_value=1.0)
            ]
        )
        
        # Training Settings
        self.register_config(
            "training.enabled",
            ConfigScope.GLOBAL,
            bool,
            "Enable training",
            default_value=True,
            validators=[TypeValidator(bool)]
        )
        
        self.register_config(
            "training.batch_size",
            ConfigScope.GLOBAL,
            int,
            "Training batch size",
            default_value=32,
            validators=[
                TypeValidator(int),
                RangeValidator(min_value=1, max_value=1024)
            ]
        )
        
        self.register_config(
            "training.max_episodes",
            ConfigScope.GLOBAL,
            int,
            "Maximum training episodes",
            default_value=1000,
            validators=[
                TypeValidator(int),
                RangeValidator(min_value=1, max_value=100000)
            ]
        )
        
        self.register_config(
            "training.save_interval",
            ConfigScope.GLOBAL,
            int,
            "Model save interval (episodes)",
            default_value=100,
            validators=[
                TypeValidator(int),
                RangeValidator(min_value=1, max_value=10000)
            ]
        )
        
        # Integration Settings
        self.register_config(
            "integration.mode",
            ConfigScope.GLOBAL,
            str,
            "Integration mode",
            default_value="shadow",
            validators=[
                TypeValidator(str),
                ChoiceValidator(["disabled", "shadow", "gradual", "full", "testing"])
            ]
        )
        
        self.register_config(
            "integration.rollout_percentage",
            ConfigScope.GLOBAL,
            float,
            "Gradual rollout percentage",
            default_value=10.0,
            validators=[
                TypeValidator(float),
                RangeValidator(min_value=0.0, max_value=100.0)
            ]
        )
        
        # Performance Settings
        self.register_config(
            "performance.max_concurrent_requests",
            ConfigScope.GLOBAL,
            int,
            "Maximum concurrent requests",
            default_value=10,
            validators=[
                TypeValidator(int),
                RangeValidator(min_value=1, max_value=1000)
            ]
        )
        
        self.register_config(
            "performance.request_timeout",
            ConfigScope.GLOBAL,
            float,
            "Request timeout in seconds",
            default_value=30.0,
            validators=[
                TypeValidator(float),
                RangeValidator(min_value=1.0, max_value=300.0)
            ]
        )
        
        # Monitoring Settings
        self.register_config(
            "monitoring.enabled",
            ConfigScope.GLOBAL,
            bool,
            "Enable monitoring",
            default_value=True,
            validators=[TypeValidator(bool)]
        )
        
        self.register_config(
            "monitoring.metrics_collection_interval",
            ConfigScope.GLOBAL,
            int,
            "Metrics collection interval in seconds",
            default_value=60,
            validators=[
                TypeValidator(int),
                RangeValidator(min_value=1, max_value=3600)
            ]
        )
        
        # Storage Settings
        self.register_config(
            "storage.data_path",
            ConfigScope.GLOBAL,
            str,
            "Data storage path",
            default_value="rl_data",
            validators=[TypeValidator(str)]
        )
        
        self.register_config(
            "storage.max_experience_buffer_size",
            ConfigScope.GLOBAL,
            int,
            "Maximum experience buffer size",
            default_value=10000,
            validators=[
                TypeValidator(int),
                RangeValidator(min_value=100, max_value=1000000)
            ]
        )
        
        # Safety Settings
        self.register_config(
            "safety.enable_circuit_breaker",
            ConfigScope.GLOBAL,
            bool,
            "Enable circuit breaker",
            default_value=True,
            validators=[TypeValidator(bool)]
        )
        
        self.register_config(
            "safety.max_error_rate",
            ConfigScope.GLOBAL,
            float,
            "Maximum error rate threshold",
            default_value=0.05,
            validators=[
                TypeValidator(float),
                RangeValidator(min_value=0.0, max_value=1.0)
            ]
        )
        
        logger.debug("Initialized default configuration schema")
    
    def register_config(self, key: str, scope: ConfigScope, data_type: Type,
                       description: str = "", default_value: Any = None,
                       required: bool = False, validators: Optional[List[ConfigValidator]] = None,
                       tags: Optional[List[str]] = None) -> None:
        """Register a configuration entry"""
        with self.lock:
            metadata = ConfigMetadata(
                key=key,
                scope=scope,
                data_type=data_type.__name__,
                description=description,
                default_value=default_value,
                required=required,
                tags=tags or []
            )
            
            self.metadata_registry[key] = metadata
            
            if validators:
                self.validators[key] = validators
            
            # Set default value if not already set
            if key not in self.configurations and default_value is not None:
                self.set_config(key, default_value, source="default")
            
            logger.debug(f"Registered configuration: {key}")
    
    def set_config(self, key: str, value: Any, source: str = "manual",
                  validate: bool = True, force: bool = False) -> bool:
        """Set configuration value"""
        with self.lock:
            # Check if configuration exists
            if key not in self.metadata_registry:
                logger.warning(f"Setting unregistered configuration: {key}")
                # Auto-register with basic metadata
                self.register_config(
                    key, ConfigScope.GLOBAL, type(value),
                    f"Auto-registered configuration for {key}",
                    default_value=value
                )
            
            metadata = self.metadata_registry[key]
            
            # Check if locked
            if key in self.configurations and self.configurations[key].locked and not force:
                logger.warning(f"Configuration {key} is locked")
                return False
            
            # Validate if required
            if validate and self.validation_level != ValidationLevel.NONE:
                if not self._validate_config(key, value, metadata):
                    return False
            
            # Store old value for change tracking
            old_value = self.configurations[key].value if key in self.configurations else None
            
            # Create or update configuration entry
            entry = ConfigEntry(
                metadata=metadata,
                value=value,
                source=source
            )
            
            self.configurations[key] = entry
            metadata.updated_at = time.time()
            metadata.version += 1
            
            # Track change
            self._track_change(key, old_value, value, source)
            
            # Notify listeners
            self._notify_change_listeners(key, old_value, value)
            
            logger.debug(f"Set configuration {key} = {value} (source: {source})")
            return True
    
    def get_config(self, key: str, default: Any = None, scope: Optional[ConfigScope] = None) -> Any:
        """Get configuration value"""
        with self.lock:
            if key in self.configurations:
                entry = self.configurations[key]
                
                # Check scope if specified
                if scope and entry.metadata.scope != scope:
                    return default
                
                return entry.value
            
            # Return default from metadata if available
            if key in self.metadata_registry:
                return self.metadata_registry[key].default_value
            
            return default
    
    def get_config_entry(self, key: str) -> Optional[ConfigEntry]:
        """Get full configuration entry"""
        with self.lock:
            return self.configurations.get(key)
    
    def has_config(self, key: str) -> bool:
        """Check if configuration exists"""
        with self.lock:
            return key in self.configurations
    
    def delete_config(self, key: str, force: bool = False) -> bool:
        """Delete configuration"""
        with self.lock:
            if key not in self.configurations:
                return False
            
            entry = self.configurations[key]
            
            # Check if locked
            if entry.locked and not force:
                logger.warning(f"Configuration {key} is locked")
                return False
            
            # Check if required
            if key in self.metadata_registry and self.metadata_registry[key].required:
                logger.warning(f"Cannot delete required configuration: {key}")
                return False
            
            old_value = entry.value
            del self.configurations[key]
            
            # Track change
            self._track_change(key, old_value, None, "deleted")
            
            # Notify listeners
            self._notify_change_listeners(key, old_value, None)
            
            logger.debug(f"Deleted configuration: {key}")
            return True
    
    def lock_config(self, key: str) -> bool:
        """Lock configuration to prevent changes"""
        with self.lock:
            if key in self.configurations:
                self.configurations[key].locked = True
                logger.debug(f"Locked configuration: {key}")
                return True
            return False
    
    def unlock_config(self, key: str) -> bool:
        """Unlock configuration"""
        with self.lock:
            if key in self.configurations:
                self.configurations[key].locked = False
                logger.debug(f"Unlocked configuration: {key}")
                return True
            return False
    
    def _validate_config(self, key: str, value: Any, metadata: ConfigMetadata) -> bool:
        """Validate configuration value"""
        if self.validation_level == ValidationLevel.NONE:
            return True
        
        # Basic type validation
        if self.validation_level in [ValidationLevel.BASIC, ValidationLevel.STRICT]:
            expected_type = eval(metadata.data_type) if metadata.data_type else type(value)
            if not isinstance(value, expected_type):
                logger.error(f"Type validation failed for {key}: expected {metadata.data_type}, got {type(value).__name__}")
                return False
        
        # Custom validators
        if key in self.validators:
            for validator in self.validators[key]:
                if not validator.validate(key, value, metadata):
                    logger.error(f"Validation failed for {key}: {validator.get_error_message()}")
                    return False
        
        return True
    
    def _track_change(self, key: str, old_value: Any, new_value: Any, source: str) -> None:
        """Track configuration change"""
        change_record = {
            'timestamp': time.time(),
            'key': key,
            'old_value': old_value,
            'new_value': new_value,
            'source': source
        }
        
        self.change_history.append(change_record)
        
        # Keep only last 1000 changes
        if len(self.change_history) > 1000:
            self.change_history = self.change_history[-1000:]
    
    def _notify_change_listeners(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify change listeners"""
        for listener in self.change_listeners:
            try:
                listener(key, old_value, new_value)
            except Exception as e:
                logger.warning(f"Change listener failed: {e}")
    
    def add_change_listener(self, listener: Callable[[str, Any, Any], None]) -> None:
        """Add configuration change listener"""
        self.change_listeners.append(listener)
        logger.debug(f"Added change listener: {listener.__name__}")
    
    def remove_change_listener(self, listener: Callable[[str, Any, Any], None]) -> None:
        """Remove configuration change listener"""
        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
            logger.debug(f"Removed change listener: {listener.__name__}")
    
    def load_from_file(self, file_path: str, format_type: Optional[ConfigFormat] = None,
                      scope: ConfigScope = ConfigScope.GLOBAL, source: str = "file") -> bool:
        """Load configuration from file"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"Configuration file not found: {file_path}")
                return False
            
            # Auto-detect format if not specified
            if format_type is None:
                if file_path.suffix.lower() == '.json':
                    format_type = ConfigFormat.JSON
                elif file_path.suffix.lower() in ['.yml', '.yaml']:
                    format_type = ConfigFormat.YAML
                else:
                    format_type = ConfigFormat.JSON
            
            # Load data
            with open(file_path, 'r', encoding='utf-8') as f:
                if format_type == ConfigFormat.JSON:
                    data = json.load(f)
                elif format_type == ConfigFormat.YAML:
                    data = yaml.safe_load(f)
                else:
                    logger.error(f"Unsupported format: {format_type}")
                    return False
            
            # Set configurations
            loaded_count = 0
            for key, value in self._flatten_dict(data).items():
                if self.set_config(key, value, source=f"{source}:{file_path.name}"):
                    loaded_count += 1
            
            # Track file for auto-reload
            if self.auto_reload:
                self.file_watchers[str(file_path)] = file_path.stat().st_mtime
            
            logger.info(f"Loaded {loaded_count} configurations from {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            return False
    
    def save_to_file(self, file_path: str, format_type: Optional[ConfigFormat] = None,
                    scope: Optional[ConfigScope] = None, include_metadata: bool = False) -> bool:
        """Save configuration to file"""
        try:
            file_path = Path(file_path)
            
            # Auto-detect format if not specified
            if format_type is None:
                if file_path.suffix.lower() == '.json':
                    format_type = ConfigFormat.JSON
                elif file_path.suffix.lower() in ['.yml', '.yaml']:
                    format_type = ConfigFormat.YAML
                else:
                    format_type = ConfigFormat.JSON
            
            # Collect configurations
            data = {}
            with self.lock:
                for key, entry in self.configurations.items():
                    if scope is None or entry.metadata.scope == scope:
                        if include_metadata:
                            data[key] = entry.to_dict()
                        else:
                            data[key] = entry.value
            
            # Convert flat dict to nested if needed
            if not include_metadata:
                data = self._unflatten_dict(data)
            
            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save data
            with open(file_path, 'w', encoding='utf-8') as f:
                if format_type == ConfigFormat.JSON:
                    json.dump(data, f, indent=2, default=str)
                elif format_type == ConfigFormat.YAML:
                    yaml.dump(data, f, default_flow_style=False)
                else:
                    logger.error(f"Unsupported format: {format_type}")
                    return False
            
            logger.info(f"Saved {len(data)} configurations to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            return False
    
    def _flatten_dict(self, data: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)
    
    def _unflatten_dict(self, data: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
        """Unflatten dictionary"""
        result = {}
        for key, value in data.items():
            keys = key.split(sep)
            current = result
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        return result
    
    def check_file_changes(self) -> List[str]:
        """Check for file changes and reload if necessary"""
        changed_files = []
        
        for file_path, last_modified in list(self.file_watchers.items()):
            try:
                current_modified = Path(file_path).stat().st_mtime
                if current_modified > last_modified:
                    logger.info(f"Configuration file changed: {file_path}")
                    if self.load_from_file(file_path, source="auto-reload"):
                        self.file_watchers[file_path] = current_modified
                        changed_files.append(file_path)
            except Exception as e:
                logger.warning(f"Error checking file {file_path}: {e}")
        
        return changed_files
    
    def get_configurations_by_scope(self, scope: ConfigScope) -> Dict[str, Any]:
        """Get all configurations for a specific scope"""
        with self.lock:
            return {
                key: entry.value
                for key, entry in self.configurations.items()
                if entry.metadata.scope == scope
            }
    
    def get_configurations_by_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Get configurations that have any of the specified tags"""
        with self.lock:
            result = {}
            for key, entry in self.configurations.items():
                if any(tag in entry.metadata.tags for tag in tags):
                    result[key] = entry.value
            return result
    
    def search_configurations(self, pattern: str) -> Dict[str, Any]:
        """Search configurations by key pattern"""
        import re
        regex = re.compile(pattern, re.IGNORECASE)
        
        with self.lock:
            return {
                key: entry.value
                for key, entry in self.configurations.items()
                if regex.search(key)
            }
    
    def get_change_history(self, key: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        if key:
            history = [change for change in self.change_history if change['key'] == key]
        else:
            history = self.change_history
        
        return history[-limit:] if limit > 0 else history
    
    def export_configuration(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Export all configuration data"""
        with self.lock:
            if include_metadata:
                return {
                    'configurations': {key: entry.to_dict() for key, entry in self.configurations.items()},
                    'metadata_registry': {key: asdict(metadata) for key, metadata in self.metadata_registry.items()},
                    'export_timestamp': time.time()
                }
            else:
                return {
                    key: entry.value for key, entry in self.configurations.items()
                }
    
    def import_configuration(self, data: Dict[str, Any], overwrite: bool = False) -> bool:
        """Import configuration data"""
        try:
            imported_count = 0
            
            # Handle full export format
            if 'configurations' in data:
                config_data = data['configurations']
                for key, entry_data in config_data.items():
                    if not overwrite and key in self.configurations:
                        continue
                    
                    if self.set_config(key, entry_data['value'], source="import"):
                        imported_count += 1
            else:
                # Handle simple key-value format
                for key, value in data.items():
                    if not overwrite and key in self.configurations:
                        continue
                    
                    if self.set_config(key, value, source="import"):
                        imported_count += 1
            
            logger.info(f"Imported {imported_count} configurations")
            return True
        
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False
    
    def validate_all_configurations(self) -> Dict[str, List[str]]:
        """Validate all configurations and return errors"""
        errors = {}
        
        with self.lock:
            for key, entry in self.configurations.items():
                validation_errors = []
                
                if not self._validate_config(key, entry.value, entry.metadata):
                    if key in self.validators:
                        for validator in self.validators[key]:
                            if not validator.validate(key, entry.value, entry.metadata):
                                validation_errors.append(validator.get_error_message())
                
                if validation_errors:
                    errors[key] = validation_errors
        
        return errors
    
    def get_status(self) -> Dict[str, Any]:
        """Get configuration manager status"""
        with self.lock:
            return {
                'total_configurations': len(self.configurations),
                'registered_metadata': len(self.metadata_registry),
                'validators': len(self.validators),
                'change_listeners': len(self.change_listeners),
                'file_watchers': len(self.file_watchers),
                'change_history_size': len(self.change_history),
                'validation_level': self.validation_level.value,
                'auto_reload': self.auto_reload,
                'config_dir': str(self.config_dir),
                'scopes': {
                    scope.value: len([
                        entry for entry in self.configurations.values()
                        if entry.metadata.scope == scope
                    ])
                    for scope in ConfigScope
                }
            }
    
    def cleanup(self) -> None:
        """Cleanup configuration manager"""
        logger.info("Cleaning up configuration manager")
        
        with self.lock:
            self.configurations.clear()
            self.metadata_registry.clear()
            self.validators.clear()
            self.change_listeners.clear()
            self.change_history.clear()
            self.file_watchers.clear()
        
        logger.info("Configuration manager cleanup complete")