# Simplified GAAPF Core Components
# Reduced from 20+ complex modules to 5 essential components

from .simple_hub import SimpleLearningHub
from .simple_constellation import SimpleConstellation, create_simple_constellation
from .constellation import Constellation  # Keep for backward compatibility
from .constellation_types import (
    get_constellation_type,
    get_recommended_constellation_types,
    get_all_constellation_types,
    get_agent_roles_for_constellation
)

# Keep some existing components that are still useful
from .learning_guidance import LearningGuidance
from .session_manager import SessionManager
from .curriculum_manager import CurriculumManager

__all__ = [
    # New simplified components
    "SimpleLearningHub",
    "SimpleConstellation", 
    "create_simple_constellation",
    
    # Backward compatibility
    "Constellation",
    "get_constellation_type",
    "get_recommended_constellation_types",
    "get_all_constellation_types",
    "get_agent_roles_for_constellation",
    
    # Useful existing components
    "LearningGuidance",
    "SessionManager",
    "CurriculumManager"
] 