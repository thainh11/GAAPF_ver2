"""Enhanced Progress Tracker Agent with milestone system and BKT integration.

This agent tracks user progress with advanced features including:
- Milestone-based progress tracking
- Bayesian Knowledge Tracing integration
- Progress tree visualization
- Adaptive difficulty adjustment
- Achievement integration
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path

try:
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.offline as pyo
except ImportError:
    logging.warning("NetworkX or Plotly not available. Some features may be limited.")
    nx = None
    go = None
    pyo = None

from . import SpecializedAgent
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from ...prompts.progress_tracker import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Milestone:
    """Represents a learning milestone."""
    
    def __init__(self, milestone_id: str, name: str, description: str, 
                 prerequisites: List[str] = None, skills: List[str] = None,
                 difficulty: float = 0.5, estimated_time: int = 30):
        self.milestone_id = milestone_id
        self.name = name
        self.description = description
        self.prerequisites = prerequisites or []
        self.skills = skills or []
        self.difficulty = difficulty
        self.estimated_time = estimated_time  # in minutes
        self.completed = False
        self.completion_date = None
        self.attempts = 0
        self.best_score = 0.0

class ProgressNode:
    """Represents a node in the progress tree."""
    
    def __init__(self, node_id: str, content_type: str, title: str,
                 parent_id: str = None, children: List[str] = None):
        self.node_id = node_id
        self.content_type = content_type  # 'concept', 'exercise', 'project', 'assessment'
        self.title = title
        self.parent_id = parent_id
        self.children = children or []
        self.status = 'not_started'  # 'not_started', 'in_progress', 'completed', 'mastered'
        self.progress_percentage = 0.0
        self.last_accessed = None
        self.time_spent = 0  # in minutes
        self.difficulty_rating = None
        self.user_rating = None

class ProgressTrackerAgent(SpecializedAgent):
    """
    Enhanced specialized agent focused on comprehensive learning progress tracking.
    
    The ProgressTrackerAgent is responsible for:
    1. Tracking and analyzing user learning progress with milestones
    2. Identifying knowledge gaps using Bayesian Knowledge Tracing
    3. Recommending next steps with adaptive difficulty
    4. Providing insights on learning patterns and effectiveness
    5. Managing achievement systems and progress visualization
    """
    
    # Class attributes for agent registry support
    DESCRIPTION = "Expert in monitoring learning progress with advanced analytics"
    CAPABILITIES = [
        "progress_tracking",
        "gap_analysis",
        "learning_insights",
        "next_step_recommendations",
        "pattern_analysis",
        "milestone_tracking",
        "bayesian_knowledge_tracing",
        "progress_visualization",
        "adaptive_difficulty",
        "achievement_management"
    ]
    PRIORITY = 6  # Medium-high priority for progress tracking
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[Union[str, BaseTool]] = [],
        memory_path: Optional[Path] = None,
        config: Dict = None,
        is_logging: bool = False,
        *args, **kwargs
    ):
        """
        Initialize the Enhanced ProgressTrackerAgent.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model to use for this agent
        tools : List[Union[str, BaseTool]], optional
            Tools available to this agent
        memory_path : Path, optional
            Path to agent memory file
        config : Dict, optional
            Agent-specific configuration
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        # Set default config if not provided
        if config is None:
            config = {
                "tracking_detail": "comprehensive",
                "visualization": True,
                "focus_on_improvement": True,
                "track_time_spent": True,
                "provide_comparisons": True,
                "enable_milestones": True,
                "enable_bkt": True,
                "enable_achievements": True,
                "adaptive_difficulty": True
            }
        
        # Set default tools if not provided
        if not tools:
            tools = [
                "websearch_tools"
            ]
        
        # Initialize the base specialized agent
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            config=config,
            agent_type="progress_tracker",
            description="Expert in monitoring learning progress with advanced analytics",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        # Enhanced progress tracking components
        self.milestones: Dict[str, Milestone] = {}
        self.progress_tree: Dict[str, ProgressNode] = {}
        self.learning_sessions = []
        self.skill_assessments = {}
        self.learning_goals = {}
        self.user_preferences = {}
        
        # Progress tree graph (if NetworkX available)
        self.progress_graph = nx.DiGraph() if nx else None
        
        # Data persistence
        self.data_dir = Path("data/progress")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_progress_data()
        
        if self.is_logging:
            logger.info(f"Initialized Enhanced ProgressTrackerAgent with config: {self.config}")

    def _load_progress_data(self):
        """Load existing progress data from files."""
        try:
            # Load milestones
            milestones_file = self.data_dir / "milestones.json"
            if milestones_file.exists():
                with open(milestones_file, 'r') as f:
                    milestone_data = json.load(f)
                    for mid, data in milestone_data.items():
                        milestone = Milestone(
                            milestone_id=data['milestone_id'],
                            name=data['name'],
                            description=data['description'],
                            prerequisites=data.get('prerequisites', []),
                            skills=data.get('skills', []),
                            difficulty=data.get('difficulty', 0.5),
                            estimated_time=data.get('estimated_time', 30)
                        )
                        milestone.completed = data.get('completed', False)
                        milestone.completion_date = data.get('completion_date')
                        milestone.attempts = data.get('attempts', 0)
                        milestone.best_score = data.get('best_score', 0.0)
                        self.milestones[mid] = milestone
            
            # Load progress tree
            tree_file = self.data_dir / "progress_tree.json"
            if tree_file.exists():
                with open(tree_file, 'r') as f:
                    tree_data = json.load(f)
                    for nid, data in tree_data.items():
                        node = ProgressNode(
                            node_id=data['node_id'],
                            content_type=data['content_type'],
                            title=data['title'],
                            parent_id=data.get('parent_id'),
                            children=data.get('children', [])
                        )
                        node.status = data.get('status', 'not_started')
                        node.progress_percentage = data.get('progress_percentage', 0.0)
                        node.last_accessed = data.get('last_accessed')
                        node.time_spent = data.get('time_spent', 0)
                        node.difficulty_rating = data.get('difficulty_rating')
                        node.user_rating = data.get('user_rating')
                        self.progress_tree[nid] = node
                        
                        # Add to graph if available
                        if self.progress_graph is not None:
                            self.progress_graph.add_node(nid, **data)
                            if node.parent_id:
                                self.progress_graph.add_edge(node.parent_id, nid)
            
            logger.info("Progress data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading progress data: {e}")
    
    def _save_progress_data(self):
        """Save progress data to files."""
        try:
            # Save milestones
            milestones_data = {}
            for mid, milestone in self.milestones.items():
                milestones_data[mid] = {
                    'milestone_id': milestone.milestone_id,
                    'name': milestone.name,
                    'description': milestone.description,
                    'prerequisites': milestone.prerequisites,
                    'skills': milestone.skills,
                    'difficulty': milestone.difficulty,
                    'estimated_time': milestone.estimated_time,
                    'completed': milestone.completed,
                    'completion_date': milestone.completion_date,
                    'attempts': milestone.attempts,
                    'best_score': milestone.best_score
                }
            
            with open(self.data_dir / "milestones.json", 'w') as f:
                json.dump(milestones_data, f, indent=2)
            
            # Save progress tree
            tree_data = {}
            for nid, node in self.progress_tree.items():
                tree_data[nid] = {
                    'node_id': node.node_id,
                    'content_type': node.content_type,
                    'title': node.title,
                    'parent_id': node.parent_id,
                    'children': node.children,
                    'status': node.status,
                    'progress_percentage': node.progress_percentage,
                    'last_accessed': node.last_accessed,
                    'time_spent': node.time_spent,
                    'difficulty_rating': node.difficulty_rating,
                    'user_rating': node.user_rating
                }
            
            with open(self.data_dir / "progress_tree.json", 'w') as f:
                json.dump(tree_data, f, indent=2)
            
            logger.info("Progress data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving progress data: {e}")
    
    def _generate_system_prompt(self) -> str:
        """
        Generate a system prompt for this agent.
        
        Returns:
        -------
        str
            System prompt for the agent
        """
        return generate_system_prompt(self.config)
