#!/usr/bin/env python3
"""
Dynamic Integration Module

This module provides a unified interface for integrating all the new LLM-driven
dynamic capabilities into the existing GAAPF framework. It serves as a bridge
between the existing fixed constellation system and the new dynamic components.

Key Features:
- Seamless integration with existing codebase
- Backward compatibility with fixed constellation types
- Progressive adoption of dynamic capabilities
- Unified API for all dynamic features
- Performance monitoring and fallback mechanisms

Author: AI Assistant
Date: 2024
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

try:
    from langchain.schema import BaseLanguageModel
except ImportError:
    from langchain_core.language_models import BaseLanguageModel

from .enhanced_constellation_manager import EnhancedConstellationManager
from .dynamic_constellation_generator import DynamicConstellationGenerator
from .dynamic_learning_calibrator import DynamicLearningCalibrator
from .llm_recommendation_engine import LLMRecommendationEngine
from .constellation_types import (
    get_constellation_type,
    get_recommended_constellation_types,
    get_all_constellation_types,
    CONSTELLATION_TYPES
)

# Configure logging
logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """Integration modes for dynamic capabilities."""
    FIXED_ONLY = "fixed_only"  # Use only fixed constellation types
    HYBRID = "hybrid"  # Mix of fixed and dynamic based on context
    DYNAMIC_PREFERRED = "dynamic_preferred"  # Prefer dynamic, fallback to fixed
    DYNAMIC_ONLY = "dynamic_only"  # Use only dynamic generation


@dataclass
class DynamicCapabilities:
    """Configuration for dynamic capabilities."""
    constellation_generation: bool = True
    learning_calibration: bool = True
    intelligent_recommendations: bool = True
    performance_monitoring: bool = True
    adaptive_fallback: bool = True


class DynamicIntegrationManager:
    """
    Unified manager for integrating dynamic LLM capabilities into GAAPF.
    
    This class provides a single interface for accessing all dynamic features
    while maintaining backward compatibility with existing code.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        integration_mode: IntegrationMode = IntegrationMode.HYBRID,
        capabilities: Optional[DynamicCapabilities] = None,
        cache_ttl: int = 3600,
        is_logging: bool = False
    ):
        """
        Initialize the dynamic integration manager.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model for dynamic capabilities
        integration_mode : IntegrationMode, optional
            How to integrate dynamic capabilities
        capabilities : DynamicCapabilities, optional
            Which dynamic capabilities to enable
        cache_ttl : int, optional
            Cache time-to-live in seconds
        is_logging : bool, optional
            Enable detailed logging
        """
        self.llm = llm
        self.integration_mode = integration_mode
        self.capabilities = capabilities or DynamicCapabilities()
        self.cache_ttl = cache_ttl
        self.is_logging = is_logging
        
        # Initialize enhanced constellation manager
        self.constellation_manager = EnhancedConstellationManager(
            llm=llm,
            enable_dynamic_generation=self._should_enable_dynamic(),
            performance_tracking=self.capabilities.performance_monitoring,
            is_logging=is_logging
        )
        
        # Initialize individual components if needed
        self.dynamic_generator = None
        self.learning_calibrator = None
        self.recommendation_engine = None
        
        if self.capabilities.constellation_generation and self._should_enable_dynamic():
            self.dynamic_generator = DynamicConstellationGenerator(
                llm=llm,
                fallback_to_fixed=True,
                cache_enabled=True,
                is_logging=is_logging
            )
        
        if self.capabilities.learning_calibration and self._should_enable_dynamic():
            self.learning_calibrator = DynamicLearningCalibrator(
                llm=llm,
                performance_history_limit=50,
                calibration_cache_ttl=cache_ttl,
                is_logging=is_logging
            )
        
        if self.capabilities.intelligent_recommendations and self._should_enable_dynamic():
            self.recommendation_engine = LLMRecommendationEngine(
                llm=llm,
                recommendation_cache_ttl=cache_ttl,
                max_recommendations=5,
                is_logging=is_logging
            )
        
        # Performance tracking
        self.integration_stats = {
            "total_requests": 0,
            "dynamic_used": 0,
            "fixed_used": 0,
            "hybrid_used": 0,
            "fallback_used": 0,
            "errors": 0
        }
        
        if self.is_logging:
            logger.info(f"✅ Dynamic Integration Manager initialized (mode: {integration_mode.value})")
    
    def get_constellation_for_context(
        self,
        learning_context: Dict,
        user_query: str = "",
        user_id: str = "default_user",
        performance_data: Optional[Dict] = None,
        user_preferences: Optional[Dict] = None,
        force_mode: Optional[IntegrationMode] = None
    ) -> Dict[str, Any]:
        """
        Get optimal constellation for a learning context.
        
        This is the main entry point for constellation selection that
        intelligently chooses between fixed and dynamic approaches.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        user_query : str, optional
            Current user query
        user_id : str, optional
            User identifier
        performance_data : Dict, optional
            Recent performance data
        user_preferences : Dict, optional
            User preferences
        force_mode : IntegrationMode, optional
            Force a specific integration mode
            
        Returns:
        -------
        Dict[str, Any]
            Constellation configuration with metadata
        """
        start_time = time.time()
        self.integration_stats["total_requests"] += 1
        
        # Determine which mode to use
        mode = force_mode or self.integration_mode
        
        try:
            if mode == IntegrationMode.FIXED_ONLY:
                result = self._get_fixed_constellation(learning_context, user_query)
                self.integration_stats["fixed_used"] += 1
                
            elif mode == IntegrationMode.DYNAMIC_ONLY:
                result = self._get_dynamic_constellation(
                    learning_context, user_query, user_id, performance_data, user_preferences
                )
                self.integration_stats["dynamic_used"] += 1
                
            elif mode == IntegrationMode.DYNAMIC_PREFERRED:
                result = self._get_dynamic_with_fallback(
                    learning_context, user_query, user_id, performance_data, user_preferences
                )
                
            else:  # HYBRID mode
                result = self._get_hybrid_constellation(
                    learning_context, user_query, user_id, performance_data, user_preferences
                )
                self.integration_stats["hybrid_used"] += 1
            
            # Add integration metadata
            result["integration_mode"] = mode.value
            result["processing_time"] = time.time() - start_time
            result["capabilities_used"] = self._get_capabilities_used(result)
            
            if self.is_logging:
                logger.info(f"✅ Constellation provided via {mode.value} in {result['processing_time']:.2f}s")
            
            return result
            
        except Exception as e:
            self.integration_stats["errors"] += 1
            
            if self.is_logging:
                logger.error(f"❌ Constellation selection failed: {e}")
            
            # Emergency fallback to fixed constellation
            if mode != IntegrationMode.FIXED_ONLY:
                self.integration_stats["fallback_used"] += 1
                return self._get_fixed_constellation(learning_context, user_query)
            else:
                raise e
    
    def get_learning_recommendations(
        self,
        learning_context: Dict,
        constellation_type: str,
        user_query: str = "",
        user_id: str = "default_user",
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
        user_id : str, optional
            User identifier
        performance_data : Dict, optional
            Recent performance data
        session_goals : List[str], optional
            Specific goals for the current session
            
        Returns:
        -------
        Dict[str, Any]
            Activity recommendations with details
        """
        if self.capabilities.intelligent_recommendations and self.recommendation_engine:
            try:
                return self.recommendation_engine.recommend_learning_activities(
                    user_id=user_id,
                    learning_context=learning_context,
                    constellation_type=constellation_type,
                    performance_data=performance_data,
                    session_goals=session_goals
                )
            except Exception as e:
                if self.is_logging:
                    logger.error(f"❌ Dynamic recommendations failed: {e}")
                return self._get_fallback_recommendations(constellation_type)
        else:
            return self._get_fallback_recommendations(constellation_type)
    
    def calibrate_learning_parameters(
        self,
        learning_context: Dict,
        user_query: str = "",
        user_id: str = "default_user",
        performance_data: Optional[Dict] = None,
        user_feedback: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Calibrate learning parameters for optimal experience.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        user_query : str, optional
            Current user query
        user_id : str, optional
            User identifier
        performance_data : Dict, optional
            Recent performance metrics
        user_feedback : Dict, optional
            Direct user feedback
            
        Returns:
        -------
        Dict[str, float]
            Calibrated learning parameters
        """
        if self.capabilities.learning_calibration and self.learning_calibrator:
            try:
                return self.learning_calibrator.calibrate_learning_parameters(
                    user_id=user_id,
                    learning_context=learning_context,
                    performance_data=performance_data,
                    user_feedback=user_feedback
                )
            except Exception as e:
                if self.is_logging:
                    logger.error(f"❌ Dynamic calibration failed: {e}")
                return self._get_default_parameters()
        else:
            return self._get_default_parameters()
    
    def get_personalized_learning_path(
        self,
        learning_context: Dict,
        user_id: str = "default_user",
        learning_objectives: Optional[List[str]] = None,
        performance_history: Optional[List[Dict]] = None,
        time_constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get personalized learning path recommendations.
        
        Parameters:
        ----------
        learning_context : Dict
            Current learning context
        user_id : str, optional
            User identifier
        learning_objectives : List[str], optional
            Specific learning objectives
        performance_history : List[Dict], optional
            Historical performance data
        time_constraints : Dict, optional
            Time constraints and deadlines
            
        Returns:
        -------
        Dict[str, Any]
            Learning path recommendation with timeline
        """
        if self.capabilities.intelligent_recommendations and self.recommendation_engine:
            try:
                return self.recommendation_engine.recommend_learning_path(
                    user_id=user_id,
                    learning_context=learning_context,
                    performance_history=performance_history,
                    learning_objectives=learning_objectives,
                    time_constraints=time_constraints
                )
            except Exception as e:
                if self.is_logging:
                    logger.error(f"❌ Dynamic path recommendation failed: {e}")
                return self._get_fallback_learning_path(learning_objectives)
        else:
            return self._get_fallback_learning_path(learning_objectives)
    
    def _should_enable_dynamic(self) -> bool:
        """Check if dynamic capabilities should be enabled."""
        return self.integration_mode in [
            IntegrationMode.HYBRID,
            IntegrationMode.DYNAMIC_PREFERRED,
            IntegrationMode.DYNAMIC_ONLY
        ]
    
    def _should_enable_fallback(self) -> bool:
        """Check if fallback to fixed should be enabled."""
        return self.integration_mode != IntegrationMode.DYNAMIC_ONLY
    
    def _get_fixed_constellation(self, learning_context: Dict, user_query: str) -> Dict[str, Any]:
        """Get constellation using only fixed types."""
        # Use existing constellation type selection logic
        learning_stage = learning_context.get("learning_stage", "exploration")
        current_activity = learning_context.get("current_activity", "")
        
        # Simple rule-based selection
        if "assessment" in current_activity.lower() or "test" in user_query.lower():
            constellation_type = "assessment"
        elif "practice" in current_activity.lower() or "exercise" in user_query.lower():
            constellation_type = "practice"
        elif "project" in current_activity.lower() or "build" in user_query.lower():
            constellation_type = "project"
        elif "help" in user_query.lower() or "problem" in user_query.lower():
            constellation_type = "troubleshooting"
        else:
            constellation_type = "learning"
        
        constellation = get_constellation_type(constellation_type)
        
        return {
            "constellation": constellation,
            "type": "fixed",
            "constellation_type": constellation_type,
            "source": "rule_based",
            "learning_parameters": self._get_default_parameters()
        }
    
    def _get_dynamic_constellation(
        self,
        learning_context: Dict,
        user_query: str,
        user_id: str,
        performance_data: Optional[Dict],
        user_preferences: Optional[Dict]
    ) -> Dict[str, Any]:
        """Get constellation using only dynamic generation."""
        return self.constellation_manager.get_optimal_constellation(
            learning_context=learning_context,
            user_query=user_query,
            force_dynamic=True
        )
    
    def _get_dynamic_with_fallback(
        self,
        learning_context: Dict,
        user_query: str,
        user_id: str,
        performance_data: Optional[Dict],
        user_preferences: Optional[Dict]
    ) -> Dict[str, Any]:
        """Get constellation preferring dynamic with fallback to fixed."""
        try:
            result = self._get_dynamic_constellation(
                learning_context, user_query, user_id, performance_data, user_preferences
            )
            self.integration_stats["dynamic_used"] += 1
            return result
        except Exception as e:
            if self.is_logging:
                logger.warning(f"⚠️ Dynamic generation failed, falling back to fixed: {e}")
            self.integration_stats["fallback_used"] += 1
            return self._get_fixed_constellation(learning_context, user_query)
    
    def _get_hybrid_constellation(
        self,
        learning_context: Dict,
        user_query: str,
        user_id: str,
        performance_data: Optional[Dict],
        user_preferences: Optional[Dict]
    ) -> Dict[str, Any]:
        """Get constellation using hybrid approach."""
        # Decide whether to use dynamic or fixed based on context complexity
        complexity_score = self._calculate_context_complexity(learning_context, user_query)
        
        if complexity_score > 0.6:  # High complexity - use dynamic
            try:
                result = self._get_dynamic_constellation(
                    learning_context, user_query, user_id, performance_data, user_preferences
                )
                result["selection_reason"] = "high_complexity"
                self.integration_stats["dynamic_used"] += 1
                return result
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"⚠️ Dynamic generation failed for complex context: {e}")
                self.integration_stats["fallback_used"] += 1
        
        # Low/medium complexity or dynamic failed - use enhanced fixed
        result = self._get_fixed_constellation(learning_context, user_query)
        
        # Enhance with dynamic parameters if available
        if self.capabilities.learning_calibration and self.learning_calibrator:
            try:
                calibrated_params = self.learning_calibrator.calibrate_learning_parameters(
                    user_id=user_id,
                    learning_context=learning_context,
                    performance_data=performance_data
                )
                result["learning_parameters"] = calibrated_params
                result["enhanced"] = True
            except Exception as e:
                if self.is_logging:
                    logger.warning(f"⚠️ Parameter calibration failed: {e}")
        
        result["selection_reason"] = "low_complexity" if complexity_score <= 0.6 else "dynamic_fallback"
        return result
    
    def _calculate_context_complexity(self, learning_context: Dict, user_query: str) -> float:
        """Calculate complexity score for context to decide on dynamic vs fixed."""
        complexity = 0.0
        
        # Query complexity
        if len(user_query.split()) > 10:
            complexity += 0.2
        if any(word in user_query.lower() for word in ["complex", "advanced", "detailed", "specific"]):
            complexity += 0.3
        
        # Context complexity
        if learning_context.get("difficulty_level", "medium") == "advanced":
            complexity += 0.3
        if learning_context.get("learning_style") in ["visual", "kinesthetic", "multimodal"]:
            complexity += 0.2
        if learning_context.get("special_needs") or learning_context.get("accommodations"):
            complexity += 0.4
        
        return min(complexity, 1.0)
    
    def _get_capabilities_used(self, result: Dict[str, Any]) -> List[str]:
        """Determine which capabilities were used in the result."""
        capabilities = []
        
        if result.get("type") == "dynamic":
            capabilities.append("dynamic_constellation_generation")
        
        if "learning_parameters" in result and result.get("enhanced"):
            capabilities.append("learning_calibration")
        
        if "recommendation_details" in result:
            capabilities.append("intelligent_recommendations")
        
        return capabilities
    
    def _get_fallback_recommendations(self, constellation_type: str) -> Dict[str, Any]:
        """Get fallback activity recommendations."""
        return {
            "activities": [
                {"activity": "Read documentation", "type": "reading"},
                {"activity": "Practice exercises", "type": "practice"},
                {"activity": "Review concepts", "type": "review"}
            ],
            "fallback": True
        }
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default learning parameters."""
        return {
            "difficulty": 0.5,
            "pacing": 0.5,
            "content_depth": 0.5,
            "practice_frequency": 0.5
        }
    
    def _get_fallback_learning_path(self, learning_objectives: Optional[List[str]]) -> Dict[str, Any]:
        """Get fallback learning path."""
        modules = [
            {
                "module_id": "intro",
                "module_name": "Introduction and Foundations",
                "learning_objectives": learning_objectives[:2] if learning_objectives else ["Understand basics"],
                "estimated_duration": "2-3 hours",
                "recommended_constellation": "learning"
            },
            {
                "module_id": "practice",
                "module_name": "Practice and Application",
                "learning_objectives": learning_objectives[2:4] if learning_objectives and len(learning_objectives) > 2 else ["Apply knowledge"],
                "estimated_duration": "3-4 hours",
                "recommended_constellation": "practice"
            }
        ]
        
        return {
            "learning_path": {
                "path_name": "Standard Learning Path",
                "total_duration": "5-7 hours",
                "learning_modules": modules
            },
            "fallback": True
        }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration usage statistics."""
        stats = self.integration_stats.copy()
        
        # Calculate percentages
        total = stats["total_requests"]
        if total > 0:
            stats["dynamic_percentage"] = (stats["dynamic_used"] / total) * 100
            stats["fixed_percentage"] = (stats["fixed_used"] / total) * 100
            stats["hybrid_percentage"] = (stats["hybrid_used"] / total) * 100
            stats["fallback_percentage"] = (stats["fallback_used"] / total) * 100
            stats["error_rate"] = (stats["errors"] / total) * 100
        
        # Add component stats if available
        if self.constellation_manager and hasattr(self.constellation_manager, 'usage_stats'):
            stats["constellation_manager_stats"] = getattr(self.constellation_manager, 'usage_stats', {})
        
        return stats
    
    def update_integration_mode(self, new_mode: IntegrationMode) -> None:
        """Update the integration mode dynamically."""
        old_mode = self.integration_mode
        self.integration_mode = new_mode
        
        if self.is_logging:
            logger.info(f"🔄 Integration mode changed from {old_mode.value} to {new_mode.value}")
    
    def update_capabilities(self, new_capabilities: DynamicCapabilities) -> None:
        """Update the enabled capabilities dynamically."""
        self.capabilities = new_capabilities
        
        if self.is_logging:
            logger.info(f"🔄 Capabilities updated: {new_capabilities}")


# Convenience functions for easy integration

def create_dynamic_integration_manager(
    llm: BaseLanguageModel,
    integration_mode: IntegrationMode = IntegrationMode.HYBRID,
    **kwargs
) -> DynamicIntegrationManager:
    """
    Create a dynamic integration manager instance.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    integration_mode : IntegrationMode, optional
        Integration mode to use
    **kwargs
        Additional arguments for the manager
        
    Returns:
    -------
    DynamicIntegrationManager
        Configured manager instance
    """
    return DynamicIntegrationManager(
        llm=llm,
        integration_mode=integration_mode,
        **kwargs
    )


def get_constellation_with_dynamic_capabilities(
    llm: BaseLanguageModel,
    learning_context: Dict,
    user_query: str = "",
    integration_mode: IntegrationMode = IntegrationMode.HYBRID,
    **kwargs
) -> Dict[str, Any]:
    """
    Get constellation with dynamic capabilities - simple interface.
    
    This function provides the easiest way to get a constellation with
    all dynamic capabilities enabled.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    learning_context : Dict
        Current learning context
    user_query : str, optional
        Current user query
    integration_mode : IntegrationMode, optional
        Integration mode to use
    **kwargs
        Additional arguments
        
    Returns:
    -------
    Dict[str, Any]
        Constellation configuration with all dynamic enhancements
    """
    manager = DynamicIntegrationManager(
        llm=llm,
        integration_mode=integration_mode
    )
    
    return manager.get_constellation_for_context(
        learning_context=learning_context,
        user_query=user_query,
        **kwargs
    )


def get_enhanced_learning_experience(
    llm: BaseLanguageModel,
    learning_context: Dict,
    user_query: str = "",
    user_id: str = "default_user",
    **kwargs
) -> Dict[str, Any]:
    """
    Get a complete enhanced learning experience with all dynamic features.
    
    This function provides constellation, recommendations, and calibrated
    parameters in a single call.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    learning_context : Dict
        Current learning context
    user_query : str, optional
        Current user query
    user_id : str, optional
        User identifier
    **kwargs
        Additional arguments
        
    Returns:
    -------
    Dict[str, Any]
        Complete learning experience configuration
    """
    manager = DynamicIntegrationManager(llm=llm)
    
    # Get constellation
    constellation_result = manager.get_constellation_for_context(
        learning_context=learning_context,
        user_query=user_query,
        user_id=user_id,
        **kwargs
    )
    
    constellation_type = constellation_result.get("constellation_type", "learning")
    
    # Get recommendations
    recommendations = manager.get_learning_recommendations(
        learning_context=learning_context,
        constellation_type=constellation_type,
        user_query=user_query,
        user_id=user_id
    )
    
    # Get learning path
    learning_path = manager.get_personalized_learning_path(
        learning_context=learning_context,
        user_id=user_id
    )
    
    return {
        "constellation": constellation_result,
        "activity_recommendations": recommendations,
        "learning_path": learning_path,
        "enhanced_experience": True
    }