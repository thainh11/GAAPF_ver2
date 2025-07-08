"""
Framework Onboarding Module for GAAPF Architecture

This module integrates the Framework Information Collection into the onboarding flow,
allowing users to select a framework to learn and initializing the necessary information.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from ..tools.framework_collector import FrameworkCollector, initialize_framework_knowledge
from ..memory.long_term_memory import LongTermMemory
from .knowledge_graph import KnowledgeGraph

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of supported frameworks
SUPPORTED_FRAMEWORKS = [
    "LangChain",
    "LangGraph",
    "Microsoft AutoGen",
    "CrewAI",
    "Haystack by Deepset",
    "Hugging Face SmolAgents",
    "OpenAI Agents Python"
]

class FrameworkOnboarding:
    """
    Framework onboarding class that handles the selection and initialization
    of framework information during the user onboarding process.
    """
    
    def __init__(
        self,
        memory: LongTermMemory,
        knowledge_graph: KnowledgeGraph,
        cache_dir: Optional[Path] = Path('data/framework_cache'),
        is_logging: bool = False,
        tavily_api_key: Optional[str] = None
    ):
        """
        Initialize the framework onboarding process.
        
        Args:
            memory: LongTermMemory instance for storing collected information
            knowledge_graph: KnowledgeGraph instance for storing concept relationships
            cache_dir: Directory to cache framework information
            is_logging: Whether to enable detailed logging
            tavily_api_key: API key for Tavily if using direct client
        """
        self.memory = memory
        self.knowledge_graph = knowledge_graph
        self.cache_dir = cache_dir
        self.is_logging = is_logging
        self.tavily_api_key = tavily_api_key
        
        # Initialize framework collector
        self.collector = FrameworkCollector(
            memory=memory,
            cache_dir=cache_dir,
            is_logging=is_logging,
            tavily_api_key=tavily_api_key
        )
        
        if self.is_logging:
            logger.info("Initialized FrameworkOnboarding")
    
    def get_supported_frameworks(self) -> List[str]:
        """
        Get the list of supported frameworks.
        
        Returns:
            List of supported framework names
        """
        return SUPPORTED_FRAMEWORKS
    
    async def initialize_framework(
        self,
        framework_name: str,
        user_id: str,
        user_config: Dict[str, Any],
        initialization_mode: str = "quick",
        is_background_collection: bool = True
    ) -> Dict:
        """
        Initialize the selected framework by collecting information and creating a curriculum.
        
        Args:
            framework_name: Name of the framework to initialize
            user_id: User ID to associate with the collected information
            user_config: User configuration including experience level, goals, etc.
            initialization_mode: Mode of initialization ("quick", "comprehensive", "custom")
            is_background_collection: Whether to run comprehensive collection in the background
            
        Returns:
            Dictionary containing the initial curriculum
        """
        if framework_name not in SUPPORTED_FRAMEWORKS:
            raise ValueError(f"Framework '{framework_name}' is not supported. Choose from: {', '.join(SUPPORTED_FRAMEWORKS)}")
        
        # Initialize based on selected mode
        is_quick_init = initialization_mode in ["quick", "custom"]
        
        curriculum = await initialize_framework_knowledge(
            framework_name=framework_name,
            user_id=user_id,
            knowledge_graph=self.knowledge_graph,
            memory=self.memory,
            is_quick_init=is_quick_init
        )
        
        # Adjust curriculum based on user configuration and initialization mode
        adjusted_curriculum = self._adjust_curriculum_for_user(curriculum, user_config, initialization_mode)
        
        # Start comprehensive collection in the background if requested and not already comprehensive
        if is_background_collection and initialization_mode != "comprehensive":
            asyncio.create_task(self._run_background_collection(framework_name, user_id))
        
        return adjusted_curriculum
    
    def _adjust_curriculum_for_user(self, curriculum: Dict, user_config: Dict, initialization_mode: str = "quick") -> Dict:
        """
        Adjust the curriculum based on user configuration and initialization mode.
        
        Args:
            curriculum: Initial curriculum generated from framework information
            user_config: User configuration including experience level, goals, etc.
            initialization_mode: Mode of initialization affecting content depth
            
        Returns:
            Adjusted curriculum
        """
        # Create a copy of the curriculum to modify
        adjusted = curriculum.copy()
        
        # Apply initialization mode adjustments
        if initialization_mode == "comprehensive":
            adjusted["detail_level"] = "comprehensive"
            adjusted["include_advanced_topics"] = True
            adjusted["include_practical_examples"] = True
            adjusted["include_theoretical_background"] = True
        elif initialization_mode == "quick":
            adjusted["detail_level"] = "essential"
            adjusted["focus"] = "core_concepts"
        elif initialization_mode == "custom":
            adjusted["detail_level"] = "balanced"
            adjusted["customizable"] = True
        
        # Get user experience level
        experience_level = user_config.get("experience_level", "beginner")
        
        # Adjust based on experience level
        if experience_level == "beginner":
            # For beginners, focus more on introduction and basics
            adjusted["modules"][0]["priority"] = "high"
            adjusted["modules"][1]["priority"] = "medium"
            adjusted["modules"][2]["priority"] = "low"
            
            # Add more explanation resources
            for module in adjusted["modules"]:
                module["include_detailed_explanations"] = True
                
        elif experience_level == "intermediate":
            # For intermediate users, balance introduction and core concepts
            adjusted["modules"][0]["priority"] = "medium"
            adjusted["modules"][1]["priority"] = "high"
            adjusted["modules"][2]["priority"] = "medium"
            
        elif experience_level == "advanced":
            # For advanced users, focus on core concepts and practical application
            adjusted["modules"][0]["priority"] = "low"
            adjusted["modules"][1]["priority"] = "medium"
            adjusted["modules"][2]["priority"] = "high"
            
            # Add more advanced resources
            for module in adjusted["modules"]:
                module["include_advanced_topics"] = True
        
        # Adjust based on user goals
        if "goals" in user_config:
            if "build_production_app" in user_config["goals"]:
                # Add more practical application resources
                adjusted["modules"][2]["priority"] = "high"
                adjusted["modules"][2]["focus"] = "production_readiness"
                
            if "research" in user_config["goals"]:
                # Add more theoretical resources
                adjusted["modules"][1]["priority"] = "high"
                adjusted["modules"][1]["focus"] = "theoretical_understanding"
        
        return adjusted
    
    async def _run_background_collection(self, framework_name: str, user_id: str):
        """
        Run comprehensive framework information collection in the background.
        
        Args:
            framework_name: Name of the framework to collect information about
            user_id: User ID to associate with the collected information
        """
        logger.info(f"Starting background collection for {framework_name}")
        
        try:
            # Run comprehensive collection
            await initialize_framework_knowledge(
                framework_name=framework_name,
                user_id=user_id,
                knowledge_graph=self.knowledge_graph,
                memory=self.memory,
                is_quick_init=False
            )
            
            logger.info(f"Background collection completed for {framework_name}")
            
        except Exception as e:
            logger.error(f"Error in background collection for {framework_name}: {e}")
    
    def save_curriculum(self, curriculum: Dict, user_id: str, framework_name: str) -> str:
        """
        Save the curriculum to a file.
        
        Args:
            curriculum: Curriculum to save
            user_id: User ID
            framework_name: Name of the framework
            
        Returns:
            Path to the saved curriculum file
        """
        # Create directory if it doesn't exist
        curriculum_dir = Path(f"data/curriculums/{user_id}")
        curriculum_dir.mkdir(parents=True, exist_ok=True)
        
        # Save curriculum
        file_path = curriculum_dir / f"{framework_name.lower().replace(' ', '_')}_curriculum.json"
        with open(file_path, "w") as f:
            json.dump(curriculum, f, indent=2)
        
        return str(file_path)
    
    def load_curriculum(self, user_id: str, framework_name: str) -> Optional[Dict]:
        """
        Load a saved curriculum.
        
        Args:
            user_id: User ID
            framework_name: Name of the framework
            
        Returns:
            Loaded curriculum or None if not found
        """
        file_path = Path(f"data/curriculums/{user_id}/{framework_name.lower().replace(' ', '_')}_curriculum.json")
        
        if not file_path.exists():
            return None
        
        with open(file_path, "r") as f:
            return json.load(f) 