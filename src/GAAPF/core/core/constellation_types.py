"""
Constellation Types for GAAPF Architecture

This module defines the 5 constellation types used in the GAAPF architecture:
1. Learning Constellation
2. Practice Constellation
3. Assessment Constellation
4. Project Constellation
5. Troubleshooting Constellation

Each constellation type specifies the composition of specialized agents
and their roles within that constellation.
"""

import logging
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the 5 constellation types
CONSTELLATION_TYPES = {
    "learning": {
        "name": "Learning Constellation",
        "description": "Focused on knowledge acquisition and understanding",
        "primary_goal": "Help users understand and learn concepts effectively",
        "secondary_goals": [
            "Provide clear explanations",
            "Connect concepts to prior knowledge",
            "Synthesize information from multiple sources"
        ],
        "agents": [
            {
                "type": "instructor",
                "role": "primary",
                "description": "Provides core instruction and explanations"
            },
            {
                "type": "code_assistant",
                "role": "secondary",
                "description": "Provides code examples and implementation guidance"
            },
            {
                "type": "documentation_expert",
                "role": "secondary",
                "description": "Provides reference information and documentation"
            },
            {
                "type": "knowledge_synthesizer",
                "role": "secondary",
                "description": "Connects concepts and builds mental models"
            },
            {
                "type": "mentor",
                "role": "support",
                "description": "Provides guidance and learning support"
            },
            {
                "type": "research_assistant",
                "role": "support",
                "description": "Finds additional learning resources"
            }
        ]
    },
    "practice": {
        "name": "Practice Constellation",
        "description": "Focused on skill development through practice",
        "primary_goal": "Help users develop practical skills through exercises and examples",
        "secondary_goals": [
            "Provide targeted practice opportunities",
            "Give feedback on practice attempts",
            "Scaffold learning through guided practice"
        ],
        "agents": [
            {
                "type": "practice_facilitator",
                "role": "primary",
                "description": "Creates and guides practice activities"
            },
            {
                "type": "code_assistant",
                "role": "primary",
                "description": "Provides code examples and implementation guidance"
            },
            {
                "type": "instructor",
                "role": "secondary",
                "description": "Explains concepts related to practice"
            },
            {
                "type": "mentor",
                "role": "support",
                "description": "Provides encouragement and guidance"
            },
            {
                "type": "motivational_coach",
                "role": "support",
                "description": "Maintains motivation during practice"
            }
        ]
    },
    "assessment": {
        "name": "Assessment Constellation",
        "description": "Focused on evaluating knowledge and providing feedback",
        "primary_goal": "Assess user knowledge and provide constructive feedback",
        "secondary_goals": [
            "Identify knowledge gaps",
            "Provide targeted recommendations",
            "Track learning progress"
        ],
        "agents": [
            {
                "type": "assessment",
                "role": "primary",
                "description": "Creates and evaluates assessments"
            },
            {
                "type": "progress_tracker",
                "role": "secondary",
                "description": "Tracks learning progress over time"
            },
            {
                "type": "instructor",
                "role": "secondary",
                "description": "Explains concepts related to assessment results"
            },
            {
                "type": "mentor",
                "role": "support",
                "description": "Provides guidance based on assessment results"
            },
            {
                "type": "motivational_coach",
                "role": "support",
                "description": "Maintains motivation during assessment"
            }
        ]
    },
    "project": {
        "name": "Project Constellation",
        "description": "Focused on project-based learning and application",
        "primary_goal": "Guide users through project-based learning experiences",
        "secondary_goals": [
            "Support application of knowledge",
            "Guide project planning and execution",
            "Provide feedback on project work"
        ],
        "agents": [
            {
                "type": "project_guide",
                "role": "primary",
                "description": "Guides project planning and execution"
            },
            {
                "type": "code_assistant",
                "role": "primary",
                "description": "Provides implementation assistance"
            },
            {
                "type": "documentation_expert",
                "role": "secondary",
                "description": "Provides reference information"
            },
            {
                "type": "troubleshooter",
                "role": "secondary",
                "description": "Helps resolve project issues"
            },
            {
                "type": "knowledge_synthesizer",
                "role": "support",
                "description": "Connects project concepts"
            }
        ]
    },
    "troubleshooting": {
        "name": "Troubleshooting Constellation",
        "description": "Focused on error resolution and debugging",
        "primary_goal": "Help users resolve errors and debug problems",
        "secondary_goals": [
            "Identify root causes of issues",
            "Suggest effective solutions",
            "Explain error resolution for learning"
        ],
        "agents": [
            {
                "type": "troubleshooter",
                "role": "primary",
                "description": "Identifies and resolves errors"
            },
            {
                "type": "code_assistant",
                "role": "primary",
                "description": "Provides implementation assistance"
            },
            {
                "type": "documentation_expert",
                "role": "secondary",
                "description": "Provides reference information"
            },
            {
                "type": "instructor",
                "role": "secondary",
                "description": "Explains concepts related to errors"
            },
            {
                "type": "mentor",
                "role": "support",
                "description": "Provides encouragement during debugging"
            }
        ]
    }
}

def get_constellation_type(constellation_type: str) -> Optional[Dict]:
    """
    Get a constellation type configuration.
    
    Parameters:
    ----------
    constellation_type : str
        Type of constellation
        
    Returns:
    -------
    Optional[Dict]
        Constellation type configuration or None if not found
    """
    return CONSTELLATION_TYPES.get(constellation_type)

def get_recommended_constellation_types(learning_context: Dict) -> List[str]:
    """
    Get recommended constellation types based on learning context.
    
    Parameters:
    ----------
    learning_context : Dict
        Current learning context
        
    Returns:
    -------
    List[str]
        List of recommended constellation types in order of relevance
    """
    # Get learning stage and activity
    learning_stage = learning_context.get("learning_stage", "exploration")
    current_activity = learning_context.get("current_activity", "introduction")
    
    # Map learning stages to constellation types
    stage_to_constellation = {
        "exploration": ["learning"],
        "concept": ["learning"],
        "practice": ["practice"],
        "assessment": ["assessment"],
        "project": ["project"],
        "troubleshooting": ["troubleshooting"]
    }
    
    # Map activities to constellation types
    activity_to_constellation = {
        "introduction": ["learning"],
        "concept_learning": ["learning"],
        "coding": ["practice", "project"],
        "practice": ["practice"],
        "assessment": ["assessment"],
        "project": ["project"],
        "debugging": ["troubleshooting"]
    }
    
    # Get recommendations from activity (higher priority)
    recommendations = []
    if current_activity in activity_to_constellation:
        recommendations.extend(activity_to_constellation[current_activity])
    
    # Add recommendations from stage if not already included
    if learning_stage in stage_to_constellation:
        for constellation_type in stage_to_constellation[learning_stage]:
            if constellation_type not in recommendations:
                recommendations.append(constellation_type)
    
    # Add default if no recommendations
    if not recommendations:
        recommendations = ["learning"]
    
    return recommendations

def get_all_constellation_types() -> Dict[str, Dict[str, Any]]:
    """
    Get all constellation type configurations.
    
    Returns:
    -------
    Dict[str, Dict[str, Any]]
        All constellation type configurations
    """
    return CONSTELLATION_TYPES

def get_agent_roles_for_constellation(constellation_type: str) -> Dict[str, str]:
    """
    Get a mapping of agent types to their roles in a constellation.
    
    Parameters:
    ----------
    constellation_type : str
        Name of the constellation type
        
    Returns:
    -------
    Dict[str, str]
        Mapping of agent types to roles
    """
    constellation = get_constellation_type(constellation_type)
    if not constellation:
        return {}
    
    roles = {}
    for agent in constellation.get("agents", []):
        roles[agent["type"]] = agent["role"]
    
    return roles 