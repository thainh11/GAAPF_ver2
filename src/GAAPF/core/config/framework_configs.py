import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameworkConfig:
    """
    Framework-specific configuration management.
    
    The FrameworkConfig class is responsible for:
    1. Defining modules and learning paths for each framework
    2. Managing framework configuration data
    3. Providing methods for framework configuration retrieval and updates
    """
    
    def __init__(
        self,
        frameworks_path: Optional[Union[Path, str]] = Path('frameworks'),
        is_logging: bool = False,
        *args, **kwargs
    ):
        """
        Initialize the FrameworkConfig manager with data path.
        
        Parameters:
        ----------
        frameworks_path : Path or str
            Path to the directory for storing framework configurations
        is_logging : bool
            Flag to enable detailed logging
        """
        # Initialize path
        if isinstance(frameworks_path, str):
            self.frameworks_path = Path(frameworks_path)
        else:
            self.frameworks_path = frameworks_path
        
        # Create directory if it doesn't exist
        self.frameworks_path.mkdir(parents=True, exist_ok=True)
        
        self.is_logging = is_logging
        
        # In-memory cache of framework configs
        self.framework_cache = {}
        
        if self.is_logging:
            logger.info(f"FrameworkConfig manager initialized with data at {self.frameworks_path}")
    
    def create_framework(
        self,
        framework_id: str,
        name: str,
        description: str,
        version: str = "1.0.0",
        modules: Dict = None,
        learning_paths: Dict = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Create a new framework configuration.
        
        Parameters:
        ----------
        framework_id : str
            Unique identifier for the framework
        name : str
            Framework name
        description : str
            Framework description
        version : str, optional
            Framework version
        modules : Dict, optional
            Dictionary of modules with their configurations
        learning_paths : Dict, optional
            Dictionary of learning paths through the modules
        metadata : Dict, optional
            Additional metadata about the framework
            
        Returns:
        -------
        Dict
            The created framework configuration
        """
        # Set defaults
        if modules is None:
            modules = {
                "introduction": {
                    "title": "Introduction",
                    "description": "Introduction to the framework",
                    "complexity": "basic",
                    "estimated_duration": 30,  # minutes
                    "concepts": ["overview", "installation"],
                    "prerequisites": []
                }
            }
        
        if learning_paths is None:
            learning_paths = {
                "default": {
                    "name": "Default Path",
                    "description": "Standard learning path",
                    "modules": ["introduction"],
                    "target_experience": "beginner"
                }
            }
        
        if metadata is None:
            metadata = {
                "tags": [],
                "category": "general",
                "official_website": "",
                "repository": ""
            }
        
        # Create framework config
        framework = {
            "id": framework_id,
            "name": name,
            "description": description,
            "version": version,
            "modules": modules,
            "learning_paths": learning_paths,
            "metadata": metadata,
            "starting_module": "introduction",
            "created_at": self._get_current_timestamp(),
            "updated_at": self._get_current_timestamp()
        }
        
        # Save framework config
        self._save_framework(framework_id, framework)
        
        # Update cache
        self.framework_cache[framework_id] = framework
        
        if self.is_logging:
            logger.info(f"Created framework configuration for {framework_id}")
        
        return framework
    
    def get_framework(self, framework_id: str) -> Optional[Dict]:
        """
        Get a framework configuration by ID.
        
        Parameters:
        ----------
        framework_id : str
            Unique identifier for the framework
            
        Returns:
        -------
        Optional[Dict]
            The framework configuration or None if not found
        """
        # Check cache first
        if framework_id in self.framework_cache:
            return self.framework_cache[framework_id]
        
        # Try to load from file
        framework_path = self.frameworks_path / f"{framework_id}.json"
        
        if not framework_path.exists():
            if self.is_logging:
                logger.warning(f"Framework configuration for {framework_id} not found")
            return None
        
        try:
            with open(framework_path, "r", encoding="utf-8") as f:
                framework = json.load(f)
                
                # Update cache
                self.framework_cache[framework_id] = framework
                
                return framework
        except Exception as e:
            logger.error(f"Error loading framework configuration for {framework_id}: {e}")
            return None
    
    def update_framework(self, framework_id: str, updates: Dict) -> Optional[Dict]:
        """
        Update a framework configuration.
        
        Parameters:
        ----------
        framework_id : str
            Unique identifier for the framework
        updates : Dict
            Dictionary of framework fields to update
            
        Returns:
        -------
        Optional[Dict]
            The updated framework configuration or None if not found
        """
        # Get current framework config
        framework = self.get_framework(framework_id)
        
        if framework is None:
            if self.is_logging:
                logger.warning(f"Cannot update framework {framework_id}: configuration not found")
            return None
        
        # Apply updates
        for key, value in updates.items():
            if key in framework:
                # Handle nested dictionaries
                if isinstance(framework[key], dict) and isinstance(value, dict):
                    framework[key].update(value)
                else:
                    framework[key] = value
        
        # Update timestamp
        framework["updated_at"] = self._get_current_timestamp()
        
        # Save updated framework config
        self._save_framework(framework_id, framework)
        
        # Update cache
        self.framework_cache[framework_id] = framework
        
        if self.is_logging:
            logger.info(f"Updated framework configuration for {framework_id}")
        
        return framework
    
    def add_module(
        self,
        framework_id: str,
        module_id: str,
        title: str,
        description: str,
        complexity: str = "basic",
        estimated_duration: int = 30,
        concepts: List[str] = None,
        prerequisites: List[str] = None,
        content: Dict = None
    ) -> Optional[Dict]:
        """
        Add a module to a framework configuration.
        
        Parameters:
        ----------
        framework_id : str
            Unique identifier for the framework
        module_id : str
            Unique identifier for the module
        title : str
            Module title
        description : str
            Module description
        complexity : str, optional
            Module complexity (basic, intermediate, advanced)
        estimated_duration : int, optional
            Estimated duration in minutes
        concepts : List[str], optional
            List of concepts covered in this module
        prerequisites : List[str], optional
            List of prerequisite module IDs
        content : Dict, optional
            Module content structure
            
        Returns:
        -------
        Optional[Dict]
            The updated framework configuration or None if not found
        """
        # Get current framework config
        framework = self.get_framework(framework_id)
        
        if framework is None:
            if self.is_logging:
                logger.warning(f"Cannot add module to framework {framework_id}: configuration not found")
            return None
        
        # Set defaults
        if concepts is None:
            concepts = []
        
        if prerequisites is None:
            prerequisites = []
        
        if content is None:
            content = {
                "sections": [
                    {
                        "title": "Introduction",
                        "type": "text",
                        "content": f"Introduction to {title}"
                    }
                ]
            }
        
        # Create module
        module = {
            "title": title,
            "description": description,
            "complexity": complexity,
            "estimated_duration": estimated_duration,
            "concepts": concepts,
            "prerequisites": prerequisites,
            "content": content
        }
        
        # Add module to framework
        if "modules" not in framework:
            framework["modules"] = {}
        
        framework["modules"][module_id] = module
        
        # Update timestamp
        framework["updated_at"] = self._get_current_timestamp()
        
        # Save updated framework config
        self._save_framework(framework_id, framework)
        
        # Update cache
        self.framework_cache[framework_id] = framework
        
        if self.is_logging:
            logger.info(f"Added module {module_id} to framework {framework_id}")
        
        return framework
    
    def add_learning_path(
        self,
        framework_id: str,
        path_id: str,
        name: str,
        description: str,
        modules: List[str],
        target_experience: str = "beginner"
    ) -> Optional[Dict]:
        """
        Add a learning path to a framework configuration.
        
        Parameters:
        ----------
        framework_id : str
            Unique identifier for the framework
        path_id : str
            Unique identifier for the learning path
        name : str
            Learning path name
        description : str
            Learning path description
        modules : List[str]
            Ordered list of module IDs in this path
        target_experience : str, optional
            Target experience level for this path
            
        Returns:
        -------
        Optional[Dict]
            The updated framework configuration or None if not found
        """
        # Get current framework config
        framework = self.get_framework(framework_id)
        
        if framework is None:
            if self.is_logging:
                logger.warning(f"Cannot add learning path to framework {framework_id}: configuration not found")
            return None
        
        # Create learning path
        learning_path = {
            "name": name,
            "description": description,
            "modules": modules,
            "target_experience": target_experience
        }
        
        # Add learning path to framework
        if "learning_paths" not in framework:
            framework["learning_paths"] = {}
        
        framework["learning_paths"][path_id] = learning_path
        
        # Update timestamp
        framework["updated_at"] = self._get_current_timestamp()
        
        # Save updated framework config
        self._save_framework(framework_id, framework)
        
        # Update cache
        self.framework_cache[framework_id] = framework
        
        if self.is_logging:
            logger.info(f"Added learning path {path_id} to framework {framework_id}")
        
        return framework
    
    def get_all_frameworks(self) -> List[Dict]:
        """
        Get a list of all framework configurations.
        
        Returns:
        -------
        List[Dict]
            List of framework configurations
        """
        frameworks = []
        
        for framework_path in self.frameworks_path.glob("*.json"):
            framework_id = framework_path.stem
            framework = self.get_framework(framework_id)
            if framework:
                frameworks.append(framework)
        
        return frameworks
    
    def get_module(self, framework_id: str, module_id: str) -> Optional[Dict]:
        """
        Get a specific module from a framework.
        
        Parameters:
        ----------
        framework_id : str
            Unique identifier for the framework
        module_id : str
            Unique identifier for the module
            
        Returns:
        -------
        Optional[Dict]
            The module configuration or None if not found
        """
        # Get framework config
        framework = self.get_framework(framework_id)
        
        if framework is None:
            return None
        
        # Get module
        modules = framework.get("modules", {})
        return modules.get(module_id)
    
    def get_learning_path(self, framework_id: str, path_id: str) -> Optional[Dict]:
        """
        Get a specific learning path from a framework.
        
        Parameters:
        ----------
        framework_id : str
            Unique identifier for the framework
        path_id : str
            Unique identifier for the learning path
            
        Returns:
        -------
        Optional[Dict]
            The learning path configuration or None if not found
        """
        # Get framework config
        framework = self.get_framework(framework_id)
        
        if framework is None:
            return None
        
        # Get learning path
        learning_paths = framework.get("learning_paths", {})
        return learning_paths.get(path_id)
    
    def _save_framework(self, framework_id: str, framework: Dict) -> None:
        """
        Save a framework configuration to file.
        
        Parameters:
        ----------
        framework_id : str
            Unique identifier for the framework
        framework : Dict
            Framework configuration data to save
        """
        framework_path = self.frameworks_path / f"{framework_id}.json"
        
        try:
            with open(framework_path, "w", encoding="utf-8") as f:
                json.dump(framework, f, indent=4)
                
            if self.is_logging:
                logger.info(f"Saved framework configuration for {framework_id}")
        except Exception as e:
            logger.error(f"Error saving framework configuration for {framework_id}: {e}")
    
    def _get_current_timestamp(self) -> int:
        """
        Get the current timestamp.
        
        Returns:
        -------
        int
            Current timestamp
        """
        import time
        return int(time.time()) 