"""Custom exceptions for GAAPF.

This module defines custom exception classes for better error handling
and more specific error reporting throughout the GAAPF system.
"""

import time
import traceback
from enum import Enum
from typing import Dict, List, Optional, Any, Callable


class ErrorSeverity(Enum):
    """Error severity levels for better error categorization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better error classification."""
    SYSTEM = "system"
    USER_INPUT = "user_input"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    LOGIC = "logic"
    EXTERNAL_SERVICE = "external_service"


class RecoveryStrategy(Enum):
    """Recovery strategies for different types of errors."""
    RETRY = "retry"
    FALLBACK = "fallback"
    IGNORE = "ignore"
    ESCALATE = "escalate"
    USER_INTERVENTION = "user_intervention"

class GaapfException(Exception):
    """Base exception class for all GAAPF-related errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = None, 
        details: dict = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.ESCALATE,
        retry_count: int = 0,
        max_retries: int = 3,
        recovery_function: Optional[Callable] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
            severity: Error severity level
            category: Error category for classification
            recovery_strategy: Suggested recovery strategy
            retry_count: Current retry attempt count
            max_retries: Maximum number of retry attempts
            recovery_function: Optional recovery function to call
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.severity = severity
        self.category = category
        self.recovery_strategy = recovery_strategy
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.recovery_function = recovery_function
        self.timestamp = time.time()
        self.traceback_info = traceback.format_exc()
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "severity": self.severity.value,
            "category": self.category.value,
            "recovery_strategy": self.recovery_strategy.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timestamp": self.timestamp,
            "traceback": self.traceback_info
        }
    
    def can_retry(self) -> bool:
        """Check if this error can be retried."""
        return (
            self.recovery_strategy == RecoveryStrategy.RETRY and 
            self.retry_count < self.max_retries
        )
    
    def should_escalate(self) -> bool:
        """Check if this error should be escalated."""
        return (
            self.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] or
            self.recovery_strategy == RecoveryStrategy.ESCALATE or
            (self.recovery_strategy == RecoveryStrategy.RETRY and not self.can_retry())
        )
    
    def get_user_friendly_message(self) -> str:
        """Get a user-friendly version of the error message."""
        if self.severity == ErrorSeverity.LOW:
            return f"Minor issue: {self.message}"
        elif self.severity == ErrorSeverity.MEDIUM:
            return f"Issue encountered: {self.message}"
        elif self.severity == ErrorSeverity.HIGH:
            return f"Important error: {self.message}"
        else:  # CRITICAL
            return f"Critical error: {self.message}. Please contact support."
    
    def attempt_recovery(self) -> bool:
        """Attempt to recover from this error using the recovery function."""
        if self.recovery_function and callable(self.recovery_function):
            try:
                self.recovery_function(self)
                return True
            except Exception as e:
                self.details["recovery_error"] = str(e)
                return False
        return False

class AgentException(GaapfException):
    """Base exception for agent-related errors."""
    pass

class AgentCreationError(AgentException):
    """Raised when an agent cannot be created or initialized."""
    
    def __init__(self, agent_type: str, reason: str, original_error: Exception = None):
        message = f"Failed to create agent '{agent_type}': {reason}"
        details = {
            "agent_type": agent_type,
            "reason": reason,
            "original_error": str(original_error) if original_error else None
        }
        super().__init__(message, "AGENT_CREATION_FAILED", details)
        self.agent_type = agent_type
        self.original_error = original_error

class AgentNotFoundError(AgentException):
    """Raised when a requested agent is not found in the constellation."""
    
    def __init__(self, agent_type: str, available_agents: list = None):
        message = f"Agent '{agent_type}' not found"
        if available_agents:
            message += f". Available agents: {', '.join(available_agents)}"
        details = {
            "agent_type": agent_type,
            "available_agents": available_agents or []
        }
        super().__init__(message, "AGENT_NOT_FOUND", details)
        self.agent_type = agent_type
        self.available_agents = available_agents

class AgentTimeoutError(AgentException):
    """Raised when an agent operation times out."""
    
    def __init__(self, agent_type: str, timeout_seconds: int, operation: str = None):
        message = f"Agent '{agent_type}' timed out after {timeout_seconds} seconds"
        if operation:
            message += f" during {operation}"
        details = {
            "agent_type": agent_type,
            "timeout_seconds": timeout_seconds,
            "operation": operation
        }
        super().__init__(message, "AGENT_TIMEOUT", details)
        self.agent_type = agent_type
        self.timeout_seconds = timeout_seconds
        self.operation = operation

class ConstellationException(GaapfException):
    """Base exception for constellation-related errors."""
    pass

class ConstellationCreationError(ConstellationException):
    """Raised when a constellation cannot be created."""
    
    def __init__(self, constellation_type: str, reason: str):
        message = f"Failed to create constellation '{constellation_type}': {reason}"
        details = {
            "constellation_type": constellation_type,
            "reason": reason
        }
        super().__init__(message, "CONSTELLATION_CREATION_FAILED", details)
        self.constellation_type = constellation_type

class HandoffException(ConstellationException):
    """Raised when agent handoff fails."""
    
    def __init__(self, from_agent: str, to_agent: str, reason: str):
        message = f"Handoff from '{from_agent}' to '{to_agent}' failed: {reason}"
        details = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "reason": reason
        }
        super().__init__(message, "HANDOFF_FAILED", details)
        self.from_agent = from_agent
        self.to_agent = to_agent

class MaxHandoffsExceededError(ConstellationException):
    """Raised when maximum number of handoffs is exceeded."""
    
    def __init__(self, max_handoffs: int, current_handoffs: int):
        message = f"Maximum handoffs ({max_handoffs}) exceeded. Current: {current_handoffs}"
        details = {
            "max_handoffs": max_handoffs,
            "current_handoffs": current_handoffs
        }
        super().__init__(message, "MAX_HANDOFFS_EXCEEDED", details)
        self.max_handoffs = max_handoffs
        self.current_handoffs = current_handoffs

class LearningException(GaapfException):
    """Base exception for learning-related errors."""
    pass

class SessionException(LearningException):
    """Base exception for session-related errors."""
    pass

class SessionNotFoundError(SessionException):
    """Raised when a learning session is not found."""
    
    def __init__(self, session_id: str, user_id: str = None):
        message = f"Session '{session_id}' not found"
        if user_id:
            message += f" for user '{user_id}'"
        details = {
            "session_id": session_id,
            "user_id": user_id
        }
        super().__init__(message, "SESSION_NOT_FOUND", details)
        self.session_id = session_id
        self.user_id = user_id

class SessionExpiredError(SessionException):
    """Raised when a learning session has expired."""
    
    def __init__(self, session_id: str, expired_at: str):
        message = f"Session '{session_id}' expired at {expired_at}"
        details = {
            "session_id": session_id,
            "expired_at": expired_at
        }
        super().__init__(message, "SESSION_EXPIRED", details)
        self.session_id = session_id
        self.expired_at = expired_at

class UserProfileException(LearningException):
    """Base exception for user profile-related errors."""
    pass

class UserProfileNotFoundError(UserProfileException):
    """Raised when a user profile is not found."""
    
    def __init__(self, user_id: str):
        message = f"User profile '{user_id}' not found"
        details = {"user_id": user_id}
        super().__init__(message, "USER_PROFILE_NOT_FOUND", details)
        self.user_id = user_id

class UserProfileValidationError(UserProfileException):
    """Raised when user profile data is invalid."""
    
    def __init__(self, user_id: str, validation_errors: list):
        message = f"User profile '{user_id}' validation failed: {', '.join(validation_errors)}"
        details = {
            "user_id": user_id,
            "validation_errors": validation_errors
        }
        super().__init__(message, "USER_PROFILE_VALIDATION_FAILED", details)
        self.user_id = user_id
        self.validation_errors = validation_errors

class FrameworkException(GaapfException):
    """Base exception for framework-related errors."""
    pass

class FrameworkNotFoundError(FrameworkException):
    """Raised when a framework configuration is not found."""
    
    def __init__(self, framework_name: str, available_frameworks: list = None):
        message = f"Framework '{framework_name}' not found"
        if available_frameworks:
            message += f". Available frameworks: {', '.join(available_frameworks)}"
        details = {
            "framework_name": framework_name,
            "available_frameworks": available_frameworks or []
        }
        super().__init__(message, "FRAMEWORK_NOT_FOUND", details)
        self.framework_name = framework_name
        self.available_frameworks = available_frameworks

class FrameworkValidationError(FrameworkException):
    """Raised when framework configuration is invalid."""
    
    def __init__(self, framework_name: str, validation_errors: list):
        message = f"Framework '{framework_name}' validation failed: {', '.join(validation_errors)}"
        details = {
            "framework_name": framework_name,
            "validation_errors": validation_errors
        }
        super().__init__(message, "FRAMEWORK_VALIDATION_FAILED", details)
        self.framework_name = framework_name
        self.validation_errors = validation_errors

class LLMException(GaapfException):
    """Base exception for LLM-related errors."""
    pass

class LLMTimeoutError(LLMException):
    """Raised when LLM request times out."""
    
    def __init__(self, timeout_seconds: int, operation: str = None):
        message = f"LLM request timed out after {timeout_seconds} seconds"
        if operation:
            message += f" during {operation}"
        details = {
            "timeout_seconds": timeout_seconds,
            "operation": operation
        }
        super().__init__(message, "LLM_TIMEOUT", details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation

class LLMRateLimitError(LLMException):
    """Raised when LLM rate limit is exceeded."""
    
    def __init__(self, retry_after: int = None):
        message = "LLM rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        details = {"retry_after": retry_after}
        super().__init__(message, "LLM_RATE_LIMIT_EXCEEDED", details)
        self.retry_after = retry_after

class LLMInvalidResponseError(LLMException):
    """Raised when LLM returns an invalid or unexpected response."""
    
    def __init__(self, response: str, expected_format: str = None):
        message = "LLM returned invalid response"
        if expected_format:
            message += f". Expected format: {expected_format}"
        details = {
            "response": response[:500] if response else None,  # Truncate long responses
            "expected_format": expected_format
        }
        super().__init__(message, "LLM_INVALID_RESPONSE", details)
        self.response = response
        self.expected_format = expected_format

class MemoryException(GaapfException):
    """Base exception for memory-related errors."""
    pass

class MemoryLoadError(MemoryException):
    """Raised when memory cannot be loaded."""
    
    def __init__(self, memory_path: str, reason: str):
        message = f"Failed to load memory from '{memory_path}': {reason}"
        details = {
            "memory_path": memory_path,
            "reason": reason
        }
        super().__init__(message, "MEMORY_LOAD_FAILED", details)
        self.memory_path = memory_path

class MemorySaveError(MemoryException):
    """Raised when memory cannot be saved."""
    
    def __init__(self, memory_path: str, reason: str):
        message = f"Failed to save memory to '{memory_path}': {reason}"
        details = {
            "memory_path": memory_path,
            "reason": reason
        }
        super().__init__(message, "MEMORY_SAVE_FAILED", details)
        self.memory_path = memory_path

class MemoryCorruptedError(MemoryException):
    """Raised when memory data is corrupted or invalid."""
    
    def __init__(self, memory_path: str, corruption_details: str = None):
        message = f"Memory data corrupted in '{memory_path}'"
        if corruption_details:
            message += f": {corruption_details}"
        details = {
            "memory_path": memory_path,
            "corruption_details": corruption_details
        }
        super().__init__(message, "MEMORY_CORRUPTED", details)
        self.memory_path = memory_path
        self.corruption_details = corruption_details


class ErrorHandler:
    """Centralized error handling utility class."""
    
    def __init__(self):
        self.error_history: List[GaapfException] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_attempts: Dict[str, int] = {}
        self.max_history_size = 1000
    
    def handle_error(
        self, 
        error: GaapfException, 
        context: Dict[str, Any] = None,
        auto_recover: bool = True
    ) -> bool:
        """Handle an error with automatic recovery attempts.
        
        Args:
            error: The error to handle
            context: Additional context information
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            True if error was recovered, False otherwise
        """
        # Add context to error details
        if context:
            error.details.update({"context": context})
        
        # Track error
        self._track_error(error)
        
        # Attempt recovery if enabled
        if auto_recover and error.can_retry():
            return self._attempt_recovery(error)
        
        return False
    
    def _track_error(self, error: GaapfException) -> None:
        """Track error in history and statistics."""
        # Add to history
        self.error_history.append(error)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
        
        # Update error counts
        error_type = error.__class__.__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def _attempt_recovery(self, error: GaapfException) -> bool:
        """Attempt to recover from an error."""
        error_key = f"{error.__class__.__name__}_{error.error_code}"
        
        # Track recovery attempts
        self.recovery_attempts[error_key] = self.recovery_attempts.get(error_key, 0) + 1
        
        # Increment retry count
        error.retry_count += 1
        
        # Attempt recovery
        if error.attempt_recovery():
            return True
        
        # If recovery failed and we can still retry, implement default strategies
        if error.can_retry():
            return self._apply_default_recovery_strategy(error)
        
        return False
    
    def _apply_default_recovery_strategy(self, error: GaapfException) -> bool:
        """Apply default recovery strategies based on error type and category."""
        if error.category == ErrorCategory.NETWORK:
            # For network errors, wait and retry
            time.sleep(min(2 ** error.retry_count, 30))  # Exponential backoff
            return True
        
        elif error.category == ErrorCategory.RESOURCE:
            # For resource errors, try to free up resources
            import gc
            gc.collect()
            return True
        
        elif error.category == ErrorCategory.EXTERNAL_SERVICE:
            # For external service errors, wait longer
            time.sleep(min(5 * error.retry_count, 60))
            return True
        
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {"total_errors": 0, "error_rate": 0.0}
        
        # Calculate error rates by severity
        severity_counts = {}
        category_counts = {}
        recent_errors = []
        
        # Analyze recent errors (last hour)
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        for error in self.error_history:
            if error.timestamp >= one_hour_ago:
                recent_errors.append(error)
            
            severity = error.severity.value
            category = error.category.value
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_errors": total_errors,
            "recent_errors_count": len(recent_errors),
            "error_rate_per_hour": len(recent_errors),
            "error_counts_by_type": dict(self.error_counts),
            "error_counts_by_severity": severity_counts,
            "error_counts_by_category": category_counts,
            "recovery_attempts": dict(self.recovery_attempts),
            "most_common_errors": self._get_most_common_errors(5)
        }
    
    def _get_most_common_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most common error types."""
        sorted_errors = sorted(
            self.error_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [
            {"error_type": error_type, "count": count}
            for error_type, count in sorted_errors[:limit]
        ]
    
    def clear_error_history(self) -> None:
        """Clear error history and statistics."""
        self.error_history.clear()
        self.error_counts.clear()
        self.recovery_attempts.clear()
    
    def get_recent_errors(
        self, 
        hours: int = 1, 
        severity_filter: Optional[ErrorSeverity] = None
    ) -> List[GaapfException]:
        """Get recent errors within specified time frame."""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        recent_errors = [
            error for error in self.error_history 
            if error.timestamp >= cutoff_time
        ]
        
        if severity_filter:
            recent_errors = [
                error for error in recent_errors 
                if error.severity == severity_filter
            ]
        
        return recent_errors


# Global error handler instance
global_error_handler = ErrorHandler()