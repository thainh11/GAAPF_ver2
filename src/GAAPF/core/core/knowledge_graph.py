"""
Knowledge Graph Manager for GAAPF Architecture

This module provides the KnowledgeGraph class that manages concept
relationships and learning paths for the GAAPF architecture.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Tuple
import json
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Manages concept relationships and learning paths.
    
    The KnowledgeGraph class is responsible for:
    1. Tracking relationships between concepts
    2. Structuring framework knowledge for better learning sequences
    3. Providing concept recommendations based on learning context
    4. Supporting knowledge synthesis across modules
    """
    
    def __init__(
        self,
        graph_path: Optional[Union[Path, str]] = Path('data/knowledge_graph.json'),
        is_logging: bool = False,
        *args, **kwargs
    ):
        """
        Initialize the KnowledgeGraph.
        
        Parameters:
        ----------
        graph_path : Path or str, optional
            Path to the knowledge graph file
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        # Initialize path
        self.graph_path = Path(graph_path) if isinstance(graph_path, str) else graph_path
        
        # Create parent directory if it doesn't exist
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.is_logging = is_logging
        
        # Initialize graph structure
        self.graph = self._load_graph()
        
        # Initialize concept index
        self.concept_index = self._build_concept_index()
        
        if self.is_logging:
            logger.info(f"KnowledgeGraph initialized with {len(self.graph['concepts'])} concepts")
    
    def update_from_interaction(
        self,
        user_id: str,
        learning_context: Dict,
        interaction_data: Dict,
        response: Dict
    ) -> None:
        """
        Update the knowledge graph based on an interaction.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        learning_context : Dict
            Current learning context
        interaction_data : Dict
            Data about the interaction
        response : Dict
            Response data
        """
        # Extract concepts from interaction and response
        interaction_concepts = self._extract_concepts(interaction_data.get("query", ""))
        response_concepts = self._extract_concepts(response.get("content", ""))
        
        # Get current module and framework
        current_module = learning_context.get("current_module", "")
        framework_id = learning_context.get("framework_id", "")
        
        # Update concept relationships
        for concept1 in interaction_concepts:
            # Add concept if not exists
            self._add_concept_if_not_exists(concept1, framework_id, current_module)
            
            # Connect to response concepts
            for concept2 in response_concepts:
                if concept1 != concept2:
                    # Add concept if not exists
                    self._add_concept_if_not_exists(concept2, framework_id, current_module)
                    
                    # Add relationship
                    self._add_relationship(concept1, concept2, "related")
        
        # Update user knowledge
        self._update_user_knowledge(user_id, interaction_concepts + response_concepts)
        
        # Save graph
        self._save_graph()
        
        if self.is_logging:
            logger.info(f"Updated knowledge graph with {len(interaction_concepts)} interaction concepts and {len(response_concepts)} response concepts")
    
    def get_related_concepts(self, concept: str, limit: int = 5) -> List[Dict]:
        """
        Get concepts related to a given concept.
        
        Parameters:
        ----------
        concept : str
            Concept to find related concepts for
        limit : int, optional
            Maximum number of related concepts to return
            
        Returns:
        -------
        List[Dict]
            List of related concepts with relationship information
        """
        # Check if concept exists
        if concept not in self.concept_index:
            return []
        
        # Get concept ID
        concept_id = self.concept_index[concept]
        
        # Get relationships
        relationships = []
        for rel in self.graph["relationships"]:
            if rel["source"] == concept_id:
                target_id = rel["target"]
                target_concept = self._get_concept_by_id(target_id)
                if target_concept:
                    relationships.append({
                        "concept": target_concept["name"],
                        "type": rel["type"],
                        "strength": rel.get("strength", 1.0),
                        "framework": target_concept.get("framework", ""),
                        "module": target_concept.get("module", "")
                    })
            elif rel["target"] == concept_id:
                source_id = rel["source"]
                source_concept = self._get_concept_by_id(source_id)
                if source_concept:
                    relationships.append({
                        "concept": source_concept["name"],
                        "type": rel["type"],
                        "strength": rel.get("strength", 1.0),
                        "framework": source_concept.get("framework", ""),
                        "module": source_concept.get("module", "")
                    })
        
        # Sort by strength and limit
        relationships.sort(key=lambda x: x["strength"], reverse=True)
        return relationships[:limit]
    
    def get_learning_path(
        self,
        framework_id: str,
        module_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get a recommended learning path for a framework or module.
        
        Parameters:
        ----------
        framework_id : str
            Identifier for the framework
        module_id : str, optional
            Identifier for a specific module
        user_id : str, optional
            Identifier for the user to personalize the path
            
        Returns:
        -------
        List[Dict]
            Ordered list of concepts forming a learning path
        """
        # Filter concepts by framework and module
        filtered_concepts = []
        for concept in self.graph["concepts"]:
            if concept.get("framework") == framework_id:
                if not module_id or concept.get("module") == module_id:
                    filtered_concepts.append(concept)
        
        # If no concepts found, return empty list
        if not filtered_concepts:
            return []
        
        # Get user knowledge if available
        user_knowledge = {}
        if user_id:
            for user_concept in self.graph.get("user_knowledge", {}).get(user_id, []):
                user_knowledge[user_concept["concept_id"]] = user_concept["mastery"]
        
        # Sort concepts by prerequisites
        sorted_concepts = self._topological_sort(filtered_concepts)
        
        # Build learning path
        learning_path = []
        for concept in sorted_concepts:
            # Calculate priority based on user knowledge
            priority = 1.0
            if user_id and concept["id"] in user_knowledge:
                # Lower priority for concepts the user already knows
                priority = 1.0 - user_knowledge[concept["id"]]
            
            learning_path.append({
                "concept": concept["name"],
                "description": concept.get("description", ""),
                "module": concept.get("module", ""),
                "prerequisites": [
                    self._get_concept_by_id(prereq_id)["name"]
                    for prereq_id in concept.get("prerequisites", [])
                    if self._get_concept_by_id(prereq_id)
                ],
                "priority": priority
            })
        
        return learning_path
    
    def get_concept_mastery(self, user_id: str, concept: str) -> float:
        """
        Get a user's mastery level for a concept.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        concept : str
            Concept to check mastery for
            
        Returns:
        -------
        float
            Mastery level (0-1) or 0 if not found
        """
        # Check if concept exists
        if concept not in self.concept_index:
            return 0.0
        
        # Get concept ID
        concept_id = self.concept_index[concept]
        
        # Check user knowledge
        user_knowledge = self.graph.get("user_knowledge", {}).get(user_id, [])
        for item in user_knowledge:
            if item["concept_id"] == concept_id:
                return item["mastery"]
        
        # Not found
        return 0.0
    
    def _load_graph(self) -> Dict:
        """
        Load the knowledge graph from file or create a new one.
        
        Returns:
        -------
        Dict
            Knowledge graph data
        """
        # Check if file exists
        if self.graph_path.exists():
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                if self.is_logging:
                    logger.error(f"Error loading knowledge graph: {e}")
        
        # Create new graph
        return {
            "concepts": [],
            "relationships": [],
            "user_knowledge": {}
        }
    
    def _save_graph(self) -> None:
        """Save the knowledge graph to file."""
        try:
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(self.graph, f, indent=2)
            
            if self.is_logging:
                logger.info(f"Knowledge graph saved to {self.graph_path}")
        except Exception as e:
            if self.is_logging:
                logger.error(f"Error saving knowledge graph: {e}")
    
    def _build_concept_index(self) -> Dict[str, str]:
        """
        Build an index of concept names to IDs.
        
        Returns:
        -------
        Dict[str, str]
            Mapping of concept names to IDs
        """
        index = {}
        for concept in self.graph["concepts"]:
            index[concept["name"]] = concept["id"]
        return index
    
    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract concepts from text.
        
        Parameters:
        ----------
        text : str
            Text to extract concepts from
            
        Returns:
        -------
        List[str]
            List of extracted concepts
        """
        # Simple extraction based on known concepts
        extracted = []
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check for each known concept
        for concept_name in self.concept_index.keys():
            if concept_name.lower() in text_lower:
                extracted.append(concept_name)
        
        return extracted
    
    def _add_concept_if_not_exists(self, concept: str, framework_id: str, module_id: str) -> str:
        """
        Add a concept to the graph if it doesn't exist.
        
        Parameters:
        ----------
        concept : str
            Concept name
        framework_id : str
            Identifier for the framework
        module_id : str
            Identifier for the module
            
        Returns:
        -------
        str
            Concept ID
        """
        # Check if concept exists
        if concept in self.concept_index:
            return self.concept_index[concept]
        
        # Generate ID
        concept_id = f"c_{len(self.graph['concepts'])}"
        
        # Add concept
        self.graph["concepts"].append({
            "id": concept_id,
            "name": concept,
            "framework": framework_id,
            "module": module_id,
            "created_at": time.time()
        })
        
        # Update index
        self.concept_index[concept] = concept_id
        
        if self.is_logging:
            logger.info(f"Added new concept: {concept}")
        
        return concept_id
    
    def _add_relationship(self, concept1: str, concept2: str, relationship_type: str) -> None:
        """
        Add or update a relationship between concepts.
        
        Parameters:
        ----------
        concept1 : str
            First concept name
        concept2 : str
            Second concept name
        relationship_type : str
            Type of relationship
        """
        # Get concept IDs
        concept1_id = self.concept_index.get(concept1)
        concept2_id = self.concept_index.get(concept2)
        
        if not concept1_id or not concept2_id:
            return
        
        # Check if relationship exists
        for rel in self.graph["relationships"]:
            if (rel["source"] == concept1_id and rel["target"] == concept2_id and
                rel["type"] == relationship_type):
                # Update strength
                rel["strength"] = min(1.0, rel.get("strength", 0.5) + 0.1)
                rel["updated_at"] = time.time()
                return
        
        # Add new relationship
        self.graph["relationships"].append({
            "source": concept1_id,
            "target": concept2_id,
            "type": relationship_type,
            "strength": 0.5,
            "created_at": time.time()
        })
        
        if self.is_logging:
            logger.info(f"Added relationship: {concept1} -> {concept2} ({relationship_type})")
    
    def _update_user_knowledge(self, user_id: str, concepts: List[str]) -> None:
        """
        Update user knowledge for concepts.
        
        Parameters:
        ----------
        user_id : str
            Identifier for the user
        concepts : List[str]
            List of concepts to update
        """
        # Initialize user knowledge if not exists
        if "user_knowledge" not in self.graph:
            self.graph["user_knowledge"] = {}
        
        if user_id not in self.graph["user_knowledge"]:
            self.graph["user_knowledge"][user_id] = []
        
        # Update knowledge for each concept
        for concept in concepts:
            if concept in self.concept_index:
                concept_id = self.concept_index[concept]
                
                # Check if concept exists in user knowledge
                found = False
                for item in self.graph["user_knowledge"][user_id]:
                    if item["concept_id"] == concept_id:
                        # Update mastery (small increment)
                        item["mastery"] = min(1.0, item["mastery"] + 0.05)
                        item["updated_at"] = time.time()
                        found = True
                        break
                
                # Add if not found
                if not found:
                    self.graph["user_knowledge"][user_id].append({
                        "concept_id": concept_id,
                        "mastery": 0.1,
                        "created_at": time.time()
                    })
    
    def _get_concept_by_id(self, concept_id: str) -> Optional[Dict]:
        """
        Get a concept by its ID.
        
        Parameters:
        ----------
        concept_id : str
            Concept ID
            
        Returns:
        -------
        Optional[Dict]
            Concept data or None if not found
        """
        for concept in self.graph["concepts"]:
            if concept["id"] == concept_id:
                return concept
        return None
    
    def _topological_sort(self, concepts: List[Dict]) -> List[Dict]:
        """
        Sort concepts by prerequisites.
        
        Parameters:
        ----------
        concepts : List[Dict]
            List of concepts to sort
            
        Returns:
        -------
        List[Dict]
            Sorted list of concepts
        """
        # Build adjacency list
        adj_list = {}
        for concept in concepts:
            concept_id = concept["id"]
            adj_list[concept_id] = []
            
            # Add prerequisites
            for prereq_id in concept.get("prerequisites", []):
                adj_list[concept_id].append(prereq_id)
        
        # Perform topological sort
        visited = set()
        temp = set()
        order = []
        
        def dfs(node):
            if node in temp:
                # Cycle detected, skip
                return
            if node in visited:
                return
            
            temp.add(node)
            
            # Visit prerequisites
            for prereq in adj_list.get(node, []):
                dfs(prereq)
            
            temp.remove(node)
            visited.add(node)
            order.append(node)
        
        # Visit all nodes
        for concept in concepts:
            dfs(concept["id"])
        
        # Convert back to concepts
        sorted_concepts = []
        for concept_id in order:
            for concept in concepts:
                if concept["id"] == concept_id:
                    sorted_concepts.append(concept)
                    break
        
        return sorted_concepts