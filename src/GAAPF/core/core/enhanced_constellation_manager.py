"""Enhanced Constellation Manager for GAAPF Architecture

This module provides an enhanced constellation manager that integrates:
1. Dynamic LLM-driven constellation generation
2. Existing fixed constellation types as fallback
3. Intelligent constellation selection and optimization
4. Performance tracking and adaptive learning

This extends the existing constellation system with dynamic capabilities
while maintaining backward compatibility.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from langchain_core.language_models import BaseLanguageModel

# Import existing modules
from .constellation_types import (
    CONSTELLATION_TYPES, 
    get_constellation_type, 
    get_recommended_constellation_types,
    get_all_constellation_types
)
from .dynamic_constellation_generator import DynamicConstellationGenerator
from .dynamic_learning_calibrator import DynamicLearningCalibrator
from .llm_recommendation_engine import LLMRecommendationEngine
from .constellation import Constellation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedConstellationManager:
    """
    Enhanced constellation manager that combines dynamic LLM-driven generation
    with existing fixed constellation types for optimal learning experiences.
    
    This manager provides:
    - Dynamic constellation generation based on learning context
    - Intelligent fallback to fixed constellation types
    - Performance tracking and optimization
    - Adaptive constellation selection
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        enable_dynamic_generation: bool = True,
        dynamic_threshold: float = 0.7,
        performance_tracking: bool = True,
        is_logging: bool = False
    ):
        """
        Initialize the enhanced constellation manager.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model to use for dynamic generation
        enable_dynamic_generation : bool, optional
            Whether to enable dynamic constellation generation
        dynamic_threshold : float, optional
            Confidence threshold for using dynamic constellations
        performance_tracking : bool, optional
            Whether to track constellation performance
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        self.llm = llm
        self.enable_dynamic_generation = enable_dynamic_generation
        self.dynamic_threshold = dynamic_threshold
        self.performance_tracking = performance_tracking
        self.is_logging = is_logging
        
        # Initialize dynamic components if enabled
        self.dynamic_generator = None
        self.learning_calibrator = None
        self.recommendation_engine = None
        
        if self.enable_dynamic_generation:
            self.dynamic_generator = DynamicConstellationGenerator(
                llm=llm,
                fallback_to_fixed=True,
                cache_enabled=True,
                is_logging=is_logging
            )
            self.learning_calibrator = DynamicLearningCalibrator(
                llm=llm,
                calibration_cache_ttl=3600,
                is_logging=is_logging
            )
            self.recommendation_engine = LLMRecommendationEngine(
                llm=llm,
                recommendation_cache_ttl=3600,
                is_logging=is_logging
            )
        
        # Performance tracking
        self.constellation_performance = {}
        self.usage_stats = {
            "dynamic_used": 0,
            "fixed_used": 0,
            "fallback_used": 0,
            "total_requests": 0
        }
        
        if self.is_logging:
            logger.info(f"✅ Enhanced Constellation Manager initialized with full dynamic capabilities (dynamic: {enable_dynamic_generation})")
    
    def get_optimal_constellation(
        self,
        learning_context: Dict,
        user_query: str = "",
        force_dynamic: bool = False,
        custom_requirements: Optional[Dict] = None
    ) -> Dict:
        """
        Get the optimal constellation for the given learning context.
        
        This method intelligently chooses between dynamic generation and
        fixed constellation types based on context and performance data.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        user_query : str, optional
            Current user query for context
        force_dynamic : bool, optional
            Force dynamic generation even if confidence is low
        custom_requirements : Dict, optional
            Custom requirements for constellation generation
            
        Returns:
        -------
        Dict
            Optimal constellation configuration
        """
        start_time = time.time()
        self.usage_stats["total_requests"] += 1
        
        if self.is_logging:
            logger.info(f"🎯 Getting optimal constellation for: {learning_context.get('current_activity', 'unknown')}")
        
        constellation = None
        constellation_source = "unknown"
        
        # Get intelligent constellation recommendation using LLM engine
        if (self.enable_dynamic_generation and self.recommendation_engine and 
            (force_dynamic or self._should_use_dynamic(learning_context))):
            
            try:
                # Get LLM-powered constellation recommendation
                recommendation = self.recommendation_engine.recommend_constellation_type(
                    user_id="default_user",
                    learning_context=learning_context,
                    performance_data=None,
                    user_preferences=None
                )
                
                constellation = self.dynamic_generator.generate_dynamic_constellation(
                    learning_context=learning_context,
                    user_query=user_query,
                    custom_requirements=custom_requirements
                )
                
                if constellation and self._meets_quality_threshold(constellation):
                    # Calibrate learning parameters
                    if self.learning_calibrator:
                        calibrated_params = self.learning_calibrator.calibrate_learning_parameters(
                            user_id="default_user",
                            learning_context=learning_context,
                            performance_data=None
                        )
                        constellation["learning_parameters"] = calibrated_params
                    
                    constellation["recommendation_details"] = recommendation
                    constellation_source = "dynamic"
                    self.usage_stats["dynamic_used"] += 1
                    
                    if self.is_logging:
                        logger.info("✨ Using dynamic constellation with calibrated parameters")
                else:
                    constellation = None  # Force fallback
                    
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"Dynamic generation failed: {e}")
                constellation = None
        
        # Fallback to fixed constellation types
        if not constellation:
            constellation = self._get_fixed_constellation(learning_context)
            if constellation:
                constellation_source = "fixed" if not self.enable_dynamic_generation else "fallback"
                self.usage_stats["fixed_used" if constellation_source == "fixed" else "fallback_used"] += 1
                
                if self.is_logging:
                    logger.info(f"📋 Using {constellation_source} constellation")
        
        # Ultimate fallback
        if not constellation:
            constellation = get_constellation_type("learning")
            constellation_source = "ultimate_fallback"
            
            if self.is_logging:
                logger.warning("⚠️ Using ultimate fallback constellation")
        
        # Add metadata
        if constellation:
            constellation["_metadata"] = {
                "source": constellation_source,
                "generation_time": time.time() - start_time,
                "timestamp": time.time(),
                "context_hash": self._generate_context_hash(learning_context)
            }
            
            # Track performance if enabled
            if self.performance_tracking:
                self._track_constellation_usage(constellation, learning_context)
        
        return constellation
    
    def get_learning_activity_recommendations(
        self,
        learning_context: Dict,
        constellation_type: str,
        user_query: str = "",
        performance_data: Optional[Dict] = None,
        session_goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get intelligent learning activity recommendations.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        constellation_type : str
            Selected constellation type
        user_query : str, optional
            Current user query
        performance_data : Dict, optional
            Recent performance data
        session_goals : List[str], optional
            Specific goals for the current session
            
        Returns:
        -------
        Dict[str, Any]
            Activity recommendations with details
        """
        if self.recommendation_engine:
            try:
                return self.recommendation_engine.recommend_learning_activities(
                    user_id="default_user",
                    learning_context=learning_context,
                    constellation_type=constellation_type,
                    performance_data=performance_data,
                    session_goals=session_goals
                )
            except Exception as e:
                if self.is_logging:
                    logger.error(f"❌ Activity recommendation failed: {e}")
                return self._get_fallback_activity_recommendations(constellation_type)
        else:
            return self._get_fallback_activity_recommendations(constellation_type)
    
    def calibrate_learning_parameters(
        self,
        learning_context: Dict,
        user_query: str = "",
        performance_data: Optional[Dict] = None,
        user_feedback: Optional[Dict] = None,
        force_recalibration: bool = False
    ) -> Dict[str, float]:
        """
        Calibrate learning parameters for optimal learning experience.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        user_query : str, optional
            Current user query
        performance_data : Dict, optional
            Recent performance metrics
        user_feedback : Dict, optional
            Direct user feedback
        force_recalibration : bool, optional
            Force recalibration even if cached
            
        Returns:
        -------
        Dict[str, float]
            Calibrated learning parameters
        """
        if self.learning_calibrator:
            try:
                return self.learning_calibrator.calibrate_learning_parameters(
                    user_id="default_user",
                    learning_context=learning_context,
                    performance_data=performance_data,
                    user_feedback=user_feedback,
                    force_recalibration=force_recalibration
                )
            except Exception as e:
                if self.is_logging:
                    logger.error(f"❌ Parameter calibration failed: {e}")
                return self._get_default_learning_parameters()
        else:
            return self._get_default_learning_parameters()
    
    def create_constellation_instance(
        self,
        learning_context: Dict,
        user_query: str = "",
        user_id: str = "default_user",
        memory_path: Optional[Path] = None,
        is_logging: bool = False,
        **kwargs
    ) -> Constellation:
        """
        Create a constellation instance using the optimal constellation configuration.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        user_query : str, optional
            Current user query for context
        user_id : str, optional
            User identifier
        memory_path : Path, optional
            Path for constellation memory
        is_logging : bool, optional
            Flag to enable detailed logging
        **kwargs
            Additional arguments for constellation creation
            
        Returns:
        -------
        Constellation
            Configured constellation instance
        """
        # Get optimal constellation configuration
        constellation_config = self.get_optimal_constellation(
            learning_context=learning_context,
            user_query=user_query,
            **kwargs
        )
        
        # Determine constellation type for compatibility
        constellation_type = self._determine_constellation_type(constellation_config)
        
        # Create constellation instance
        constellation = Constellation(
            llm=self.llm,
            constellation_type=constellation_type,
            constellation_config=constellation_config,
            is_logging=is_logging or self.is_logging
        )
        
        if self.is_logging:
            logger.info(f"🏗️ Created constellation instance: {constellation_config.get('name', 'Unknown')}")
        
        return constellation
    
    def get_recommended_constellations(
        self,
        learning_context: Dict,
        include_dynamic: bool = True,
        max_recommendations: int = 3
    ) -> List[Dict]:
        """
        Get multiple recommended constellations for the learning context.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        include_dynamic : bool, optional
            Whether to include dynamic recommendations
        max_recommendations : int, optional
            Maximum number of recommendations to return
            
        Returns:
        -------
        List[Dict]
            List of recommended constellation configurations
        """
        recommendations = []
        
        # Get dynamic recommendation if enabled
        if include_dynamic and self.enable_dynamic_generation and self.dynamic_generator:
            try:
                dynamic_constellation = self.dynamic_generator.generate_dynamic_constellation(
                    learning_context=learning_context
                )
                if dynamic_constellation:
                    dynamic_constellation["_recommendation_source"] = "dynamic"
                    recommendations.append(dynamic_constellation)
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"Dynamic recommendation failed: {e}")
        
        # Get fixed recommendations
        fixed_types = get_recommended_constellation_types(learning_context)
        for constellation_type in fixed_types[:max_recommendations]:
            fixed_constellation = get_constellation_type(constellation_type)
            if fixed_constellation:
                fixed_constellation["_recommendation_source"] = "fixed"
                recommendations.append(fixed_constellation)
        
        # Remove duplicates and limit results
        unique_recommendations = []
        seen_names = set()
        
        for rec in recommendations:
            name = rec.get("name", "")
            if name not in seen_names:
                seen_names.add(name)
                unique_recommendations.append(rec)
                
                if len(unique_recommendations) >= max_recommendations:
                    break
        
        return unique_recommendations
    
    def _should_use_dynamic(self, learning_context: Dict) -> bool:
        """
        Determine if dynamic generation should be used based on context.
        """
        # Check if context suggests complex or unique requirements
        complex_indicators = [
            learning_context.get("current_activity") not in ["introduction", "concept_learning"],
            learning_context.get("user_profile", {}).get("level") in ["advanced", "expert"],
            len(learning_context.get("custom_requirements", {})) > 0,
            learning_context.get("interaction_count", 0) > 10
        ]
        
        # Use dynamic if multiple complexity indicators are present
        return sum(complex_indicators) >= 2
    
    def _meets_quality_threshold(self, constellation: Dict) -> bool:
        """
        Check if the dynamic constellation meets quality thresholds.
        """
        if not constellation:
            return False
        
        # Check confidence score if available
        confidence = constellation.get("confidence", 0.5)
        if confidence < self.dynamic_threshold:
            return False
        
        # Check basic structure
        required_fields = ["name", "description", "primary_goal", "agents"]
        for field in required_fields:
            if field not in constellation:
                return False
        
        # Check agent composition
        agents = constellation.get("agents", [])
        if len(agents) < 2 or len(agents) > 8:
            return False
        
        return True
    
    def _get_fixed_constellation(self, learning_context: Dict) -> Optional[Dict]:
        """
        Get a fixed constellation type based on learning context.
        """
        recommended_types = get_recommended_constellation_types(learning_context)
        
        if recommended_types:
            return get_constellation_type(recommended_types[0])
        
        return get_constellation_type("learning")
    
    def _determine_constellation_type(self, constellation_config: Dict) -> str:
        """
        Determine the constellation type for backward compatibility.
        """
        # Check if it's a known fixed type
        constellation_name = constellation_config.get("name", "")
        
        for type_key, type_config in CONSTELLATION_TYPES.items():
            if type_config.get("name") == constellation_name:
                return type_key
        
        # For dynamic constellations, infer type from primary goal or agents
        primary_goal = constellation_config.get("primary_goal", "").lower()
        agents = constellation_config.get("agents", [])
        
        # Simple heuristics for type inference
        if "assess" in primary_goal or "evaluat" in primary_goal:
            return "assessment"
        elif "practice" in primary_goal or "exercise" in primary_goal:
            return "practice"
        elif "project" in primary_goal or "build" in primary_goal:
            return "project"
        elif "troubleshoot" in primary_goal or "debug" in primary_goal:
            return "troubleshooting"
        elif any(agent.get("type") == "practice_facilitator" for agent in agents):
            return "practice"
        elif any(agent.get("type") == "assessment" for agent in agents):
            return "assessment"
        else:
            return "learning"  # Default fallback
    
    def _track_constellation_usage(self, constellation: Dict, learning_context: Dict):
        """
        Track constellation usage for performance analysis.
        """
        constellation_id = constellation.get("name", "unknown")
        
        if constellation_id not in self.constellation_performance:
            self.constellation_performance[constellation_id] = {
                "usage_count": 0,
                "total_generation_time": 0.0,
                "contexts": [],
                "source_types": {}
            }
        
        perf_data = self.constellation_performance[constellation_id]
        perf_data["usage_count"] += 1
        
        # Track generation time
        metadata = constellation.get("_metadata", {})
        generation_time = metadata.get("generation_time", 0.0)
        perf_data["total_generation_time"] += generation_time
        
        # Track source type
        source = metadata.get("source", "unknown")
        perf_data["source_types"][source] = perf_data["source_types"].get(source, 0) + 1
        
        # Track context (limited to avoid memory bloat)
        if len(perf_data["contexts"]) < 10:
            context_summary = {
                "activity": learning_context.get("current_activity"),
                "stage": learning_context.get("learning_stage"),
                "module": learning_context.get("current_module")
            }
            perf_data["contexts"].append(context_summary)
    
    def _generate_context_hash(self, learning_context: Dict) -> str:
        """
        Generate a hash for the learning context for tracking.
        """
        key_components = [
            learning_context.get("current_activity", ""),
            learning_context.get("learning_stage", ""),
            learning_context.get("current_module", ""),
            str(learning_context.get("user_profile", {}).get("level", ""))
        ]
        return str(hash(tuple(key_components)))
    
    def _get_fallback_activity_recommendations(self, constellation_type: str) -> Dict[str, Any]:
        """
        Provide fallback activity recommendations.
        """
        activities_by_type = {
            "learning": [
                {"name": "Concept Introduction", "type": "exploration", "duration": 15},
                {"name": "Guided Practice", "type": "practice", "duration": 20}
            ],
            "practice": [
                {"name": "Skill Drills", "type": "practice", "duration": 25},
                {"name": "Application Exercises", "type": "application", "duration": 20}
            ],
            "assessment": [
                {"name": "Knowledge Check", "type": "assessment", "duration": 15},
                {"name": "Performance Evaluation", "type": "assessment", "duration": 30}
            ],
            "project": [
                {"name": "Creative Project", "type": "application", "duration": 45}
            ],
            "troubleshooting": [
                {"name": "Diagnostic Assessment", "type": "assessment", "duration": 20},
                {"name": "Remediation Practice", "type": "practice", "duration": 25}
            ]
        }
        
        activities = activities_by_type.get(constellation_type, activities_by_type["learning"])
        
        return {
            "recommended_activities": activities,
            "total_duration": sum(act["duration"] for act in activities),
            "fallback": True
        }
    
    def _get_default_learning_parameters(self) -> Dict[str, float]:
        """
        Get default learning parameters as fallback.
        """
        return {
            "difficulty": 0.5,
            "pacing": 1.0,
            "content_depth": 0.6,
            "practice_frequency": 0.5,
            "explanation_detail": 0.6,
            "interaction_density": 0.5
        }
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics for the constellation manager.
        
        Returns:
        -------
        Dict
            Performance statistics including usage counts and timing data
        """
        stats = {
            "usage_stats": self.usage_stats.copy(),
            "constellation_performance": {},
            "summary": {}
        }
        
        # Calculate performance summaries
        for constellation_id, perf_data in self.constellation_performance.items():
            avg_generation_time = (
                perf_data["total_generation_time"] / perf_data["usage_count"]
                if perf_data["usage_count"] > 0 else 0.0
            )
            
            stats["constellation_performance"][constellation_id] = {
                "usage_count": perf_data["usage_count"],
                "avg_generation_time": avg_generation_time,
                "source_distribution": perf_data["source_types"]
            }
        
        # Overall summary
        total_requests = self.usage_stats["total_requests"]
        if total_requests > 0:
            stats["summary"] = {
                "dynamic_usage_rate": self.usage_stats["dynamic_used"] / total_requests,
                "fixed_usage_rate": self.usage_stats["fixed_used"] / total_requests,
                "fallback_rate": self.usage_stats["fallback_used"] / total_requests,
                "total_constellations_tracked": len(self.constellation_performance)
            }
        
        return stats
    
    def reset_performance_stats(self):
        """
        Reset performance tracking statistics.
        """
        self.constellation_performance.clear()
        self.usage_stats = {
            "dynamic_used": 0,
            "fixed_used": 0,
            "fallback_used": 0,
            "total_requests": 0
        }
        
        if self.is_logging:
            logger.info("📊 Performance statistics reset")


# Convenience functions for easy integration

def create_enhanced_constellation_manager(
    llm: BaseLanguageModel,
    **kwargs
) -> EnhancedConstellationManager:
    """
    Create an enhanced constellation manager instance.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    **kwargs
        Additional arguments for the manager
        
    Returns:
    -------
    EnhancedConstellationManager
        Configured manager instance
    """
    return EnhancedConstellationManager(llm=llm, **kwargs)


def get_enhanced_constellation_for_context(
    llm: BaseLanguageModel,
    learning_context: Dict,
    user_query: str = "",
    **kwargs
) -> Dict:
    """
    Get an enhanced constellation for the given context.
    
    This function provides a simple interface for getting enhanced constellations
    without managing the manager instance.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    learning_context : Dict
        Current learning context
    user_query : str, optional
        Current user query
    **kwargs
        Additional arguments for constellation generation
        
    Returns:
    -------
    Dict
        Enhanced constellation configuration
    """
    manager = EnhancedConstellationManager(llm=llm)
    
    return manager.get_optimal_constellation(
        learning_context=learning_context,
        user_query=user_query,
        **kwargs
    )


def get_learning_recommendations(
    llm: BaseLanguageModel,
    learning_context: Dict,
    constellation_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Get learning activity recommendations for a specific context.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    learning_context : Dict
        Current learning context
    constellation_type : str
        Selected constellation type
    **kwargs
        Additional arguments for recommendations
        
    Returns:
    -------
    Dict[str, Any]
        Activity recommendations
    """
    manager = EnhancedConstellationManager(llm=llm)
    
    return manager.get_learning_activity_recommendations(
        learning_context=learning_context,
        constellation_type=constellation_type,
        **kwargs
    )


def calibrate_learning_for_context(
    llm: BaseLanguageModel,
    learning_context: Dict,
    **kwargs
) -> Dict[str, float]:
    """
    Calibrate learning parameters for a specific context.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    learning_context : Dict
        Current learning context
    **kwargs
        Additional arguments for calibration
        
    Returns:
    -------
    Dict[str, float]
        Calibrated learning parameters
    """
    manager = EnhancedConstellationManager(llm=llm)
    
    return manager.calibrate_learning_parameters(
        learning_context=learning_context,
        **kwargs
    )