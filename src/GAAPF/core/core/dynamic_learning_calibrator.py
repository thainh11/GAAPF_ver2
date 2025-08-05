"""Dynamic Learning Parameter Calibrator for GAAPF Architecture

This module provides LLM-driven dynamic calibration of learning parameters that:
1. Adjusts difficulty levels based on user performance and context
2. Optimizes pacing and content depth using AI intelligence
3. Personalizes learning parameters beyond fixed ranges
4. Provides intelligent recommendations for learning progression

This extends the existing progressive learning system with dynamic capabilities.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicLearningCalibrator:
    """
    LLM-powered dynamic learning parameter calibrator.
    
    This class uses an LLM to intelligently calibrate learning parameters
    based on user performance, learning context, and personalized factors,
    going beyond fixed parameter ranges to provide truly adaptive learning.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        performance_history_limit: int = 50,
        calibration_cache_ttl: int = 1800,  # 30 minutes
        is_logging: bool = False
    ):
        """
        Initialize the dynamic learning calibrator.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model to use for parameter calibration
        performance_history_limit : int, optional
            Maximum number of performance records to keep
        calibration_cache_ttl : int, optional
            Cache time-to-live in seconds
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        self.llm = llm
        self.performance_history_limit = performance_history_limit
        self.calibration_cache_ttl = calibration_cache_ttl
        self.is_logging = is_logging
        
        # Performance tracking
        self.user_performance_history = {}
        self.calibration_cache = {}
        
        # Default parameter ranges (as fallback)
        self.default_ranges = {
            "difficulty": {"min": 0.1, "max": 1.0, "default": 0.5},
            "pacing": {"min": 0.1, "max": 2.0, "default": 1.0},
            "content_depth": {"min": 0.2, "max": 1.0, "default": 0.6},
            "practice_frequency": {"min": 0.1, "max": 1.0, "default": 0.5},
            "explanation_detail": {"min": 0.2, "max": 1.0, "default": 0.6},
            "interaction_density": {"min": 0.1, "max": 1.0, "default": 0.5}
        }
        
        # Learning style mappings
        self.learning_style_adjustments = {
            "visual": {"content_depth": 0.1, "explanation_detail": 0.1},
            "auditory": {"explanation_detail": 0.15, "interaction_density": 0.1},
            "kinesthetic": {"practice_frequency": 0.2, "interaction_density": 0.15},
            "reading": {"content_depth": 0.15, "explanation_detail": 0.1},
            "social": {"interaction_density": 0.2},
            "solitary": {"content_depth": 0.1, "practice_frequency": -0.1}
        }
        
        if self.is_logging:
            logger.info("✅ Dynamic Learning Calibrator initialized")
    
    def calibrate_learning_parameters(
        self,
        user_id: str,
        learning_context: Dict,
        performance_data: Optional[Dict] = None,
        user_feedback: Optional[Dict] = None,
        force_recalibration: bool = False
    ) -> Dict[str, float]:
        """
        Calibrate learning parameters using LLM intelligence.
        
        Parameters:
        ----------
        user_id : str
            Unique identifier for the user
        learning_context : Dict
            Current learning context and session information
        performance_data : Dict, optional
            Recent performance metrics
        user_feedback : Dict, optional
            Direct user feedback on learning experience
        force_recalibration : bool, optional
            Force recalibration even if cached results exist
            
        Returns:
        -------
        Dict[str, float]
            Calibrated learning parameters
        """
        start_time = time.time()
        
        if self.is_logging:
            logger.info(f"🎯 Calibrating learning parameters for user: {user_id}")
        
        # Check cache first
        cache_key = self._generate_cache_key(user_id, learning_context, performance_data)
        if not force_recalibration and cache_key in self.calibration_cache:
            cached_result, timestamp = self.calibration_cache[cache_key]
            if time.time() - timestamp < self.calibration_cache_ttl:
                if self.is_logging:
                    logger.info("📋 Using cached calibration")
                return cached_result
        
        # Update performance history
        if performance_data:
            self._update_performance_history(user_id, performance_data)
        
        try:
            # Generate calibrated parameters using LLM
            calibrated_params = self._llm_calibrate_parameters(
                user_id=user_id,
                learning_context=learning_context,
                performance_data=performance_data,
                user_feedback=user_feedback
            )
            
            if calibrated_params:
                # Validate and normalize parameters
                validated_params = self._validate_and_normalize_parameters(calibrated_params)
                
                # Cache successful result
                self.calibration_cache[cache_key] = (validated_params, time.time())
                
                calibration_time = time.time() - start_time
                if self.is_logging:
                    logger.info(f"✅ Parameters calibrated successfully in {calibration_time:.2f}s")
                
                return validated_params
            else:
                raise ValueError("LLM returned empty calibration")
                
        except Exception as e:
            if self.is_logging:
                logger.error(f"❌ Parameter calibration failed: {e}")
            
            # Fallback to adaptive baseline
            return self._get_adaptive_baseline_parameters(user_id, learning_context)
    
    def _llm_calibrate_parameters(
        self,
        user_id: str,
        learning_context: Dict,
        performance_data: Optional[Dict],
        user_feedback: Optional[Dict]
    ) -> Optional[Dict[str, float]]:
        """
        Use LLM to calibrate learning parameters based on comprehensive analysis.
        """
        # Prepare comprehensive context for LLM
        context_analysis = self._prepare_calibration_context(
            user_id, learning_context, performance_data, user_feedback
        )
        
        prompt = f"""You are an expert learning scientist and AI tutor specializing in personalized education.
Your task is to calibrate optimal learning parameters for a specific learner based on comprehensive analysis.

**LEARNER ANALYSIS:**
{context_analysis}

**PARAMETER DEFINITIONS:**
- **difficulty** (0.1-1.0): Content complexity level (0.1=very easy, 1.0=very challenging)
- **pacing** (0.1-2.0): Learning speed multiplier (0.1=very slow, 1.0=normal, 2.0=very fast)
- **content_depth** (0.2-1.0): Detail level in explanations (0.2=surface, 1.0=comprehensive)
- **practice_frequency** (0.1-1.0): Amount of practice exercises (0.1=minimal, 1.0=extensive)
- **explanation_detail** (0.2-1.0): Verbosity of explanations (0.2=concise, 1.0=detailed)
- **interaction_density** (0.1-1.0): Frequency of interactive elements (0.1=passive, 1.0=highly interactive)

**CALIBRATION GUIDELINES:**
1. **Performance-Based Adjustments:**
   - High success rate (>80%): Increase difficulty, pacing
   - Low success rate (<60%): Decrease difficulty, increase support
   - Inconsistent performance: Adjust pacing, increase practice

2. **Learning Style Adaptations:**
   - Visual learners: Increase content_depth, explanation_detail
   - Kinesthetic learners: Increase practice_frequency, interaction_density
   - Fast learners: Increase pacing, difficulty
   - Struggling learners: Decrease pacing, increase practice_frequency

3. **Context Considerations:**
   - Complex topics: Increase content_depth, decrease pacing
   - Practice sessions: Increase practice_frequency, interaction_density
   - Assessment preparation: Increase difficulty, practice_frequency
   - Review sessions: Decrease difficulty, increase content_depth

4. **Feedback Integration:**
   - "Too easy" feedback: Increase difficulty, pacing
   - "Too hard" feedback: Decrease difficulty, increase support
   - "Too fast" feedback: Decrease pacing, increase explanation_detail
   - "Boring" feedback: Increase interaction_density, practice_frequency

**YOUR TASK:**
Analyze all provided information and determine optimal parameter values that will:
- Maximize learning effectiveness
- Maintain appropriate challenge level
- Match the learner's preferences and capabilities
- Support the current learning objectives

**RESPONSE FORMAT:**
Respond with a JSON object containing the calibrated parameters:

{{
  "difficulty": <0.1-1.0>,
  "pacing": <0.1-2.0>,
  "content_depth": <0.2-1.0>,
  "practice_frequency": <0.1-1.0>,
  "explanation_detail": <0.2-1.0>,
  "interaction_density": <0.1-1.0>,
  "confidence": <0.0-1.0>,
  "reasoning": "<Brief explanation of key calibration decisions>"
}}

**IMPORTANT:**
- All parameter values must be within specified ranges
- Consider the holistic learning experience
- Balance challenge with support
- Prioritize learner success and engagement
"""
        
        try:
            messages = [
                SystemMessage(
                    content="You are an expert learning scientist. Respond only with valid JSON containing calibrated learning parameters."
                ),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # Extract JSON from response
            calibration = self._extract_json_from_response(content)
            
            if calibration and self._validate_calibration_structure(calibration):
                return calibration
            else:
                raise ValueError("Invalid calibration structure")
                
        except Exception as e:
            if self.is_logging:
                logger.error(f"LLM parameter calibration failed: {e}")
            return None
    
    def _prepare_calibration_context(
        self,
        user_id: str,
        learning_context: Dict,
        performance_data: Optional[Dict],
        user_feedback: Optional[Dict]
    ) -> str:
        """
        Prepare comprehensive context analysis for LLM calibration.
        """
        context_parts = []
        
        # User profile information
        user_profile = learning_context.get("user_profile", {})
        if user_profile:
            level = user_profile.get("level", "unknown")
            preferences = user_profile.get("learning_preferences", {})
            context_parts.append(f"**User Profile:**")
            context_parts.append(f"- Learning Level: {level}")
            if preferences:
                style = preferences.get("style", "adaptive")
                pace_pref = preferences.get("pace", "moderate")
                context_parts.append(f"- Learning Style: {style}")
                context_parts.append(f"- Pace Preference: {pace_pref}")
        
        # Current learning context
        context_parts.append(f"\n**Current Learning Context:**")
        current_module = learning_context.get("current_module", "unknown")
        current_activity = learning_context.get("current_activity", "unknown")
        learning_stage = learning_context.get("learning_stage", "unknown")
        interaction_count = learning_context.get("interaction_count", 0)
        
        context_parts.append(f"- Module: {current_module}")
        context_parts.append(f"- Activity: {current_activity}")
        context_parts.append(f"- Learning Stage: {learning_stage}")
        context_parts.append(f"- Session Interactions: {interaction_count}")
        
        # Performance analysis
        if performance_data:
            context_parts.append(f"\n**Recent Performance:**")
            success_rate = performance_data.get("success_rate", 0.0)
            avg_response_time = performance_data.get("avg_response_time", 0.0)
            difficulty_trend = performance_data.get("difficulty_trend", "stable")
            
            context_parts.append(f"- Success Rate: {success_rate:.1%}")
            context_parts.append(f"- Average Response Time: {avg_response_time:.1f}s")
            context_parts.append(f"- Difficulty Trend: {difficulty_trend}")
        
        # Historical performance
        if user_id in self.user_performance_history:
            history = self.user_performance_history[user_id]
            if len(history) >= 3:
                recent_scores = [entry["score"] for entry in history[-10:]]
                avg_score = statistics.mean(recent_scores)
                score_trend = "improving" if recent_scores[-1] > recent_scores[0] else "declining"
                
                context_parts.append(f"\n**Performance History:**")
                context_parts.append(f"- Recent Average Score: {avg_score:.2f}")
                context_parts.append(f"- Performance Trend: {score_trend}")
                context_parts.append(f"- Total Sessions: {len(history)}")
        
        # User feedback
        if user_feedback:
            context_parts.append(f"\n**User Feedback:**")
            for key, value in user_feedback.items():
                context_parts.append(f"- {key}: {value}")
        
        # Current parameter baseline
        current_params = self._get_current_parameters(user_id)
        if current_params:
            context_parts.append(f"\n**Current Parameters:**")
            for param, value in current_params.items():
                context_parts.append(f"- {param}: {value:.2f}")
        
        return "\n".join(context_parts)
    
    def _extract_json_from_response(self, content: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response using multiple patterns.
        """
        import re
        
        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _validate_calibration_structure(self, calibration: Dict) -> bool:
        """
        Validate the structure and values of calibrated parameters.
        """
        required_params = [
            "difficulty", "pacing", "content_depth", 
            "practice_frequency", "explanation_detail", "interaction_density"
        ]
        
        # Check required parameters
        for param in required_params:
            if param not in calibration:
                return False
            
            value = calibration[param]
            if not isinstance(value, (int, float)):
                return False
            
            # Check ranges
            param_range = self.default_ranges.get(param, {"min": 0.0, "max": 1.0})
            if value < param_range["min"] or value > param_range["max"]:
                return False
        
        return True
    
    def _validate_and_normalize_parameters(self, params: Dict) -> Dict[str, float]:
        """
        Validate and normalize parameter values to ensure they're within acceptable ranges.
        """
        normalized = {}
        
        for param_name, value in params.items():
            if param_name in self.default_ranges:
                param_range = self.default_ranges[param_name]
                # Clamp value to valid range
                normalized_value = max(param_range["min"], min(param_range["max"], float(value)))
                normalized[param_name] = round(normalized_value, 3)
        
        # Ensure all required parameters are present
        for param_name, param_config in self.default_ranges.items():
            if param_name not in normalized:
                normalized[param_name] = param_config["default"]
        
        return normalized
    
    def _get_adaptive_baseline_parameters(self, user_id: str, learning_context: Dict) -> Dict[str, float]:
        """
        Get adaptive baseline parameters based on available context.
        """
        # Start with defaults
        params = {param: config["default"] for param, config in self.default_ranges.items()}
        
        # Apply learning style adjustments
        user_profile = learning_context.get("user_profile", {})
        learning_style = user_profile.get("learning_preferences", {}).get("style", "adaptive")
        
        if learning_style in self.learning_style_adjustments:
            adjustments = self.learning_style_adjustments[learning_style]
            for param, adjustment in adjustments.items():
                if param in params:
                    params[param] = max(0.1, min(1.0, params[param] + adjustment))
        
        # Apply context-based adjustments
        current_activity = learning_context.get("current_activity", "")
        learning_stage = learning_context.get("learning_stage", "")
        
        # Activity-based adjustments
        if "practice" in current_activity.lower():
            params["practice_frequency"] = min(1.0, params["practice_frequency"] + 0.2)
            params["interaction_density"] = min(1.0, params["interaction_density"] + 0.1)
        elif "assessment" in current_activity.lower():
            params["difficulty"] = min(1.0, params["difficulty"] + 0.1)
            params["practice_frequency"] = min(1.0, params["practice_frequency"] + 0.15)
        elif "introduction" in current_activity.lower():
            params["pacing"] = max(0.1, params["pacing"] - 0.2)
            params["explanation_detail"] = min(1.0, params["explanation_detail"] + 0.1)
        
        # Stage-based adjustments
        if learning_stage == "exploration":
            params["content_depth"] = min(1.0, params["content_depth"] + 0.1)
            params["interaction_density"] = min(1.0, params["interaction_density"] + 0.1)
        elif learning_stage == "mastery":
            params["difficulty"] = min(1.0, params["difficulty"] + 0.2)
            params["practice_frequency"] = min(1.0, params["practice_frequency"] + 0.1)
        
        # Apply historical performance adjustments if available
        if user_id in self.user_performance_history:
            history = self.user_performance_history[user_id]
            if len(history) >= 3:
                recent_scores = [entry["score"] for entry in history[-5:]]
                avg_score = statistics.mean(recent_scores)
                
                if avg_score > 0.8:  # High performance
                    params["difficulty"] = min(1.0, params["difficulty"] + 0.1)
                    params["pacing"] = min(2.0, params["pacing"] + 0.2)
                elif avg_score < 0.6:  # Low performance
                    params["difficulty"] = max(0.1, params["difficulty"] - 0.1)
                    params["pacing"] = max(0.1, params["pacing"] - 0.2)
                    params["practice_frequency"] = min(1.0, params["practice_frequency"] + 0.1)
        
        return params
    
    def _update_performance_history(self, user_id: str, performance_data: Dict):
        """
        Update the performance history for a user.
        """
        if user_id not in self.user_performance_history:
            self.user_performance_history[user_id] = []
        
        history = self.user_performance_history[user_id]
        
        # Add new performance entry
        entry = {
            "timestamp": time.time(),
            "score": performance_data.get("success_rate", 0.0),
            "response_time": performance_data.get("avg_response_time", 0.0),
            "difficulty": performance_data.get("difficulty_level", 0.5),
            "activity": performance_data.get("activity_type", "unknown")
        }
        
        history.append(entry)
        
        # Limit history size
        if len(history) > self.performance_history_limit:
            history.pop(0)
    
    def _get_current_parameters(self, user_id: str) -> Optional[Dict[str, float]]:
        """
        Get current parameters for a user from recent calibrations.
        """
        # Look for recent calibrations in cache
        for cache_key, (params, timestamp) in self.calibration_cache.items():
            if user_id in str(cache_key) and time.time() - timestamp < 3600:  # 1 hour
                return {k: v for k, v in params.items() if k in self.default_ranges}
        
        return None
    
    def _generate_cache_key(self, user_id: str, learning_context: Dict, performance_data: Optional[Dict]) -> str:
        """
        Generate a cache key for parameter calibration.
        """
        key_components = [
            user_id,
            learning_context.get("current_module", ""),
            learning_context.get("current_activity", ""),
            learning_context.get("learning_stage", ""),
            str(performance_data.get("success_rate", 0.0) if performance_data else 0.0)
        ]
        return hash(tuple(key_components))
    
    def get_parameter_recommendations(
        self,
        user_id: str,
        learning_context: Dict,
        target_outcome: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Get parameter recommendations for specific learning outcomes.
        
        Parameters:
        ----------
        user_id : str
            User identifier
        learning_context : Dict
            Current learning context
        target_outcome : str, optional
            Target learning outcome (balanced, challenge, support, speed)
            
        Returns:
        -------
        Dict[str, Any]
            Parameter recommendations with explanations
        """
        base_params = self.calibrate_learning_parameters(user_id, learning_context)
        
        # Apply outcome-specific adjustments
        outcome_adjustments = {
            "challenge": {
                "difficulty": 0.2,
                "pacing": 0.3,
                "practice_frequency": 0.1
            },
            "support": {
                "difficulty": -0.2,
                "pacing": -0.3,
                "explanation_detail": 0.2,
                "practice_frequency": 0.2
            },
            "speed": {
                "pacing": 0.4,
                "content_depth": -0.1,
                "explanation_detail": -0.1
            },
            "depth": {
                "content_depth": 0.3,
                "explanation_detail": 0.2,
                "pacing": -0.2
            }
        }
        
        if target_outcome in outcome_adjustments:
            adjustments = outcome_adjustments[target_outcome]
            for param, adjustment in adjustments.items():
                if param in base_params:
                    param_range = self.default_ranges[param]
                    base_params[param] = max(
                        param_range["min"],
                        min(param_range["max"], base_params[param] + adjustment)
                    )
        
        return {
            "parameters": base_params,
            "target_outcome": target_outcome,
            "explanation": f"Parameters optimized for {target_outcome} learning experience",
            "timestamp": time.time()
        }
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """
        Get statistics about calibration performance and usage.
        
        Returns:
        -------
        Dict[str, Any]
            Calibration statistics and performance metrics
        """
        stats = {
            "total_users": len(self.user_performance_history),
            "total_calibrations": len(self.calibration_cache),
            "cache_hit_rate": 0.0,
            "avg_performance_history_length": 0.0,
            "parameter_distributions": {}
        }
        
        # Calculate average performance history length
        if self.user_performance_history:
            total_entries = sum(len(history) for history in self.user_performance_history.values())
            stats["avg_performance_history_length"] = total_entries / len(self.user_performance_history)
        
        # Analyze parameter distributions from cache
        if self.calibration_cache:
            param_values = {param: [] for param in self.default_ranges.keys()}
            
            for params, timestamp in self.calibration_cache.values():
                for param, value in params.items():
                    if param in param_values:
                        param_values[param].append(value)
            
            for param, values in param_values.items():
                if values:
                    stats["parameter_distributions"][param] = {
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
        
        return stats


# Convenience functions for easy integration

def create_dynamic_learning_calibrator(
    llm: BaseLanguageModel,
    **kwargs
) -> DynamicLearningCalibrator:
    """
    Create a dynamic learning calibrator instance.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    **kwargs
        Additional arguments for the calibrator
        
    Returns:
    -------
    DynamicLearningCalibrator
        Configured calibrator instance
    """
    return DynamicLearningCalibrator(llm=llm, **kwargs)


def calibrate_parameters_for_user(
    llm: BaseLanguageModel,
    user_id: str,
    learning_context: Dict,
    performance_data: Optional[Dict] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Calibrate learning parameters for a specific user.
    
    This function provides a simple interface for parameter calibration
    without managing the calibrator instance.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    user_id : str
        User identifier
    learning_context : Dict
        Current learning context
    performance_data : Dict, optional
        Recent performance data
    **kwargs
        Additional arguments for calibration
        
    Returns:
    -------
    Dict[str, float]
        Calibrated learning parameters
    """
    calibrator = DynamicLearningCalibrator(llm=llm)
    
    return calibrator.calibrate_learning_parameters(
        user_id=user_id,
        learning_context=learning_context,
        performance_data=performance_data,
        **kwargs
    )