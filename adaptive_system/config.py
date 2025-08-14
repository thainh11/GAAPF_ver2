"""
Configuration Management for Educational RLHF System
"""
import os
from typing import Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class MentorWeights:
    """Mentor evaluation weights configuration"""
    conceptual_accuracy: float = 0.20
    pedagogical_flow: float = 0.20
    practical_relevance: float = 0.20
    clarity_precision: float = 0.15
    engagement_motivation: float = 0.15
    depth_insights: float = 0.10
    
    def __post_init__(self):
        """Ensure weights sum to 1.0"""
        total = sum([
            self.conceptual_accuracy,
            self.pedagogical_flow,
            self.practical_relevance,
            self.clarity_precision,
            self.engagement_motivation,
            self.depth_insights
        ])
        
        if abs(total - 1.0) > 0.001:
            # Normalize weights
            self.conceptual_accuracy /= total
            self.pedagogical_flow /= total
            self.practical_relevance /= total
            self.clarity_precision /= total
            self.engagement_motivation /= total
            self.depth_insights /= total
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "conceptual_accuracy": self.conceptual_accuracy,
            "pedagogical_flow": self.pedagogical_flow,
            "practical_relevance": self.practical_relevance,
            "clarity_precision": self.clarity_precision,
            "engagement_motivation": self.engagement_motivation,
            "depth_insights": self.depth_insights
        }

@dataclass
class SystemConfig:
    """Main system configuration"""
    
    # API Configuration
    gemini_api_key: str
    
    # Framework Configuration
    framework_type: str = "langchain"
    target_level: str = "intermediate"
    
    # Server Configuration
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True
    
    # RLHF Parameters
    preference_batch_size: int = 3
    enable_count_trigger: bool = True
    # policy_update_interval: int = 300  # vẫn giữ time trigger nếu muốn
    
    # Allow larger evaluator outputs to avoid MAX_TOKENS truncation (adjust per quota)
    max_output_tokens: int = 4096
    # Max tokens to use for short/compact retry prompts (keeps retry bounded)
    evaluator_retry_max_tokens: int = 4096

    # Model Configuration
    base_model: str = "gemini-1.5-flash"
    evaluator_model: str = "gemini-2.5-flash"
    
    # Evaluation Configuration
    mentor_weights: MentorWeights = None
    
    # Safety Configuration
    enable_safety_filters: bool = False
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Initialize default mentor weights if not provided"""
        if self.mentor_weights is None:
            self.mentor_weights = MentorWeights(
                conceptual_accuracy=float(os.getenv("CONCEPTUAL_ACCURACY_WEIGHT", 0.20)),
                pedagogical_flow=float(os.getenv("PEDAGOGICAL_FLOW_WEIGHT", 0.20)),
                practical_relevance=float(os.getenv("PRACTICAL_RELEVANCE_WEIGHT", 0.20)),
                clarity_precision=float(os.getenv("CLARITY_PRECISION_WEIGHT", 0.15)),
                engagement_motivation=float(os.getenv("ENGAGEMENT_MOTIVATION_WEIGHT", 0.15)),
                depth_insights=float(os.getenv("DEPTH_INSIGHTS_WEIGHT", 0.10))
            )

def load_config() -> SystemConfig:
    """Load system configuration from environment variables"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    return SystemConfig(
        gemini_api_key=api_key,
        framework_type=os.getenv("FRAMEWORK_TYPE", "langchain"),
        target_level=os.getenv("TARGET_LEVEL", "intermediate"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        debug=os.getenv("DEBUG", "true").lower() == "true",
        # policy_update_interval=int(os.getenv("POLICY_UPDATE_INTERVAL", 300)),
        preference_batch_size=int(os.getenv("PREFERENCE_BATCH_SIZE", 20)),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", 3072)),
        base_model=os.getenv("BASE_MODEL", "gemini-1.5-flash"),
        evaluator_model=os.getenv("EVALUATOR_MODEL", "gemini-2.5-flash"),
        enable_safety_filters=os.getenv("ENABLE_SAFETY_FILTERS", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        enable_count_trigger=os.getenv("ENABLE_COUNT_TRIGGER", "true").lower() == "true",
        log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
    )

# Global configuration instance
CONFIG = load_config()