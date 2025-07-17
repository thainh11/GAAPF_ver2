"""Custom exceptions for GAAPF.

This module defines custom exception classes for better error handling
and more specific error reporting throughout the GAAPF system.
"""

class GaapfException(Exception):
    """Base exception class for all GAAPF-related errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }

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