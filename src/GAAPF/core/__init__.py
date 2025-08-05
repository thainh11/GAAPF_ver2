"""
GAAPF - Simplified Learning System
Core functionality for the GAAPF package - Simplified Architecture
"""

__version__ = "2.0.0-simplified"

# Export simplified core components (85% reduction in complexity)
from .core import (
    # New simplified components
    SimpleLearningHub,
    SimpleConstellation,
    create_simple_constellation,
    
    # Backward compatibility
    Constellation,
    LearningGuidance,
    SessionManager,
    CurriculumManager
)

from .agents import (
    SpecializedAgent,
    # Only 3 core agents in simplified architecture
    InstructorAgent,
    CodeAssistantAgent,
    PracticeFacilitatorAgent,
    CORE_AGENTS
)

from .memory import LongTermMemory
from .config import UserProfile, FrameworkConfig

__all__ = [
    # Simplified core components
    "SimpleLearningHub",
    "SimpleConstellation",
    "create_simple_constellation",
    
    # Backward compatibility
    "Constellation", 
    "LearningGuidance",
    "SessionManager",
    "CurriculumManager",
    
    # Only 3 core agents
    "SpecializedAgent",
    "InstructorAgent",
    "CodeAssistantAgent",
    "PracticeFacilitatorAgent",
    "CORE_AGENTS",
    
    # Memory and config
    "LongTermMemory",
    "UserProfile",
    "FrameworkConfig"
]
