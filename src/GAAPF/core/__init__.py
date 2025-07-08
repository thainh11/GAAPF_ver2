"""Core functionality for the GAAPF package."""

__version__ = "0.1.0"

# Export main core components
from .core import (
    LearningHub,
    Constellation,
    TemporalState,
    KnowledgeGraph,
    AnalyticsEngine
)

from .agents import (
    SpecializedAgent,
    AssessmentAgent,
    CodeAssistantAgent,
    DocumentationExpertAgent,
    InstructorAgent,
    KnowledgeSynthesizerAgent,
    MentorAgent,
    MotivationalCoachAgent,
    PracticeFacilitatorAgent,
    ProgressTrackerAgent,
    ProjectGuideAgent,
    ResearchAssistantAgent,
    TroubleshooterAgent
)

from .memory import LongTermMemory
from .config import UserProfile, FrameworkConfig

__all__ = [
    # Core components
    "LearningHub",
    "Constellation", 
    "TemporalState",
    "KnowledgeGraph",
    "AnalyticsEngine",
    
    # Agents
    "SpecializedAgent",
    "AssessmentAgent",
    "CodeAssistantAgent",
    "DocumentationExpertAgent", 
    "InstructorAgent",
    "KnowledgeSynthesizerAgent",
    "MentorAgent",
    "MotivationalCoachAgent",
    "PracticeFacilitatorAgent",
    "ProgressTrackerAgent",
    "ProjectGuideAgent",
    "ResearchAssistantAgent",
    "TroubleshooterAgent",
    
    # Memory and config
    "LongTermMemory",
    "UserProfile",
    "FrameworkConfig"
]
