"""
GAAPF - Simplified Learning System
Core functionality for the GAAPF package - Simplified Architecture
"""

__version__ = "2.0.0-simplified"

# Export simplified core components (85% reduction in complexity)
# Avoid star-importing heavy submodules at import time. Provide light re-exports lazily.
try:
    from .core import (
        # New simplified components
        SimpleLearningHub,
        SimpleConstellation,
        create_simple_constellation,
        
        # Backward compatibility (kept but may import optional pieces)
        Constellation,
        LearningGuidance,
        SessionManager,
        CurriculumManager
    )
except Exception:
    # Minimal graceful degradation – modules can be imported directly where needed
    SimpleLearningHub = None
    SimpleConstellation = None
    create_simple_constellation = None
    Constellation = None
    LearningGuidance = None
    SessionManager = None
    CurriculumManager = None

try:
    from .agents import (
        SpecializedAgent,
        # Only 3 core agents in simplified architecture
        InstructorAgent,
        CodeAssistantAgent,
        PracticeFacilitatorAgent,
        CORE_AGENTS
    )
except Exception:
    SpecializedAgent = None
    InstructorAgent = None
    CodeAssistantAgent = None
    PracticeFacilitatorAgent = None
    CORE_AGENTS = {}

try:
    from .memory import LongTermMemory
except Exception:
    LongTermMemory = None

try:
    from .config import UserProfile, FrameworkConfig
except Exception:
    UserProfile = None
    FrameworkConfig = None

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
