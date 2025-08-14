"""
Policy Management System for Educational RLHF
Handles dynamic policy configurations and updates
"""

import logging
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import google.generativeai as genai
import re
from config import CONFIG
import os
import numpy as np
logger = logging.getLogger(__name__)
os.environ["RLHF_DEV_DISABLE_SAFETY"] = "1"

@dataclass
class PolicyConfig:
    """Policy configuration for content generation"""
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7
    top_p: float = 0.9
    max_output_tokens: int = 1024
    policy_type: str = "main"  # "main" or "enhancer"
    system_prompt: str = ""
    
    # Performance tracking
    generation_count: int = 0
    total_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_generation_config(self) -> genai.GenerationConfig:
        """Convert to Gemini GenerationConfig"""
        return genai.GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            candidate_count=1
        )
    
    def update_performance(self, score: float):
        """Update performance statistics"""
        self.generation_count += 1
        self.total_score += score
        self.last_updated = datetime.now()
    
    def get_average_score(self) -> float:
        """Get average performance score"""
        return self.total_score / max(1, self.generation_count)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "policy_type": self.policy_type,
            "system_prompt": self.system_prompt,
            "generation_count": self.generation_count,
            "total_score": self.total_score,
            "average_score": self.get_average_score(),
            "last_updated": self.last_updated.isoformat()
        }

class PolicyManager:
    """
    Manages policy configurations and adaptive updates
    """
    
    def __init__(self):
        # Initialize base model
        genai.configure(api_key=CONFIG.gemini_api_key)
        self.base_model = genai.GenerativeModel(CONFIG.base_model)
        
        # Create initial policies
        self.main_policy = self._create_initial_policy("main")
        self.enhancer_policy = self._create_initial_policy("enhancer")
        
        # Policy update history
        self.update_history = []
        
        # Performance tracking
        self.policy_stats = {
            "main_selections": 0,
            "enhancer_selections": 0,
            "total_generations": 0,
            "last_policy_update": datetime.now()
        }
        
        logger.info("Policy Manager initialized with main and enhancer policies")
    
    def _create_initial_policy(self, policy_type: str) -> PolicyConfig:
        """Create initial policy configuration"""
        
        # Base system prompt for mentor behavior
        base_prompt = self._create_mentor_system_prompt()
        
        if policy_type == "main":
            return PolicyConfig(
                model_name=CONFIG.base_model,
                temperature=0.6,  # More focused for accuracy
                top_p=0.85,
                max_output_tokens=CONFIG.max_output_tokens,
                policy_type="main",
                system_prompt=base_prompt + "\n\nFocus on accuracy, clarity, and systematic explanations."
            )
        else:  # enhancer policy
            return PolicyConfig(
                model_name=CONFIG.base_model,
                temperature=0.8,  # More creative for diversity
                top_p=0.9,
                max_output_tokens=CONFIG.max_output_tokens,
                policy_type="enhancer",
                system_prompt=base_prompt + "\n\nProvide creative examples, diverse perspectives, and engaging explanations."
            )
    
    def _create_mentor_system_prompt(self) -> str:
        """Create comprehensive mentor system prompt"""
        
        return f"""You are an EXPERT AI MENTOR specializing in {CONFIG.framework_type.upper()} for {CONFIG.target_level.upper()} level learners.

MENTOR IDENTITY & APPROACH:
- You are a seasoned professional with deep expertise in {CONFIG.framework_type}
- You combine technical precision with excellent teaching skills
- You adapt your communication style to {CONFIG.target_level} level understanding
- You are patient, encouraging, and committed to student success

CORE TEACHING PRINCIPLES:

1. CONCEPTUAL ACCURACY (Priority: High)
   - Ensure all technical information is 100% correct
   - Use current best practices and up-to-date information
   - Correct any misconceptions immediately and clearly

2. PEDAGOGICAL FLOW (Priority: High)
   - Build knowledge step-by-step from fundamentals
   - Connect new concepts to previously learned material
   - Use logical progression that matches learning objectives

3. PRACTICAL RELEVANCE (Priority: High)
   - Show real-world applications and use cases
   - Provide actionable, implementable guidance
   - Address practical challenges developers actually face

4. CLARITY & PRECISION (Priority: Medium-High)
   - Use clear, unambiguous language appropriate for {CONFIG.target_level} level
   - Define technical terms when first introduced
   - Avoid jargon without explanation

5. ENGAGEMENT & MOTIVATION (Priority: Medium)
   - Make content interesting and relatable
   - Show enthusiasm for the subject matter
   - Encourage experimentation and exploration

6. DEPTH & INSIGHTS (Priority: Medium)
   - Share expert-level insights and tips
   - Explain not just "how" but "why"
   - Provide context and broader understanding

RESPONSE STRUCTURE GUIDELINES:
- Start with a clear, direct answer to the main question
- Provide step-by-step explanations when appropriate
- Include practical examples and code samples
- End with suggestions for further exploration
- Use formatting (headers, bullet points) for clarity

QUALITY STANDARDS:
- Every response should be comprehensive yet concise
- Include error handling and best practices
- Anticipate follow-up questions and address them
- Provide resources for deeper learning when relevant

Remember: You are mentoring the next generation of {CONFIG.framework_type} developers. Your expertise and teaching quality will directly impact their learning success."""
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """Basic sanitize to redact potentially flagged keywords (dev heuristic)."""
        # Allow optional blacklist in CONFIG, fallback to conservative list
        blacklist = getattr(CONFIG, "safety_blacklist", None) or [
            "hack", "exploit", "weapon", "bomb", "malware", "attack"
        ]
        sanitized = prompt
        for token in blacklist:
            sanitized = re.sub(rf"(?i)\b{re.escape(token)}\b", "[redacted]", sanitized)
        return sanitized

    def _get_safety_settings(self) -> List[Dict[str, str]]:
        """Get safety settings for Gemini API calls"""
        # Developer override
        if os.getenv("RLHF_DEV_DISABLE_SAFETY") == "1":
            logger.warning("RLHF_DEV_DISABLE_SAFETY=1: safety settings relaxed for dev testing")
            return [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        
        if CONFIG.enable_safety_filters:
            return [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        else:
            return [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

    async def generate_response(self, prompt: str, policy: PolicyConfig) -> str:
        """Generate response using specified policy with safety-aware retries."""
        try:
            full_prompt = f"{policy.system_prompt}\n\nStudent Question: {prompt}\n\nMentor Response:"
            logger.debug(f"Generating with policy={policy.policy_type} prompt_len={len(full_prompt)}")

            # Attempt 1: normal generation
            try:
                response = await self.base_model.generate_content_async(
                    full_prompt,
                    generation_config=policy.to_generation_config(),
                    safety_settings=self._get_safety_settings()
                )
            except Exception as e:
                # Inspect exception for safety/moderation keywords
                errstr = str(e)
                logger.error(f"Generation attempt 1 error ({policy.policy_type}): {errstr}")

                # If looks like moderation/safety block, try sanitize + retry once
                if "dangerous" in errstr.lower() or "safety" in errstr.lower() or "content" in errstr.lower():
                    sanitized_prompt = self._sanitize_prompt(prompt)
                    logger.info(f"Sanitizing prompt and retrying generation for policy={policy.policy_type}")
                    full_prompt2 = f"{policy.system_prompt}\n\nStudent Question: {sanitized_prompt}\n\nMentor Response:"
                    try:
                        # If dev override present, safety is relaxed by _get_safety_settings
                        response = await self.base_model.generate_content_async(
                            full_prompt2,
                            generation_config=policy.to_generation_config(),
                            safety_settings=self._get_safety_settings()
                        )
                    except Exception as e2:
                        logger.error(f"Generation retry after sanitize failed ({policy.policy_type}): {e2}")
                        # final fallback
                        return self._create_error_response(prompt, policy.policy_type, str(e2))
                else:
                    # not a safety-like error -> return error response
                    return self._create_error_response(prompt, policy.policy_type, errstr)

            # Extract response text if available
            if response and getattr(response, "candidates", None) and response.candidates[0].content:
                response_text = response.candidates[0].content.parts[0].text.strip()
                logger.debug(f"Generated {len(response_text)} character response using {policy.policy_type} policy")
                return response_text
            else:
                logger.warning(f"Empty response from {policy.policy_type} policy")
                return self._create_fallback_response(prompt, policy.policy_type)

        except Exception as e:
            # Any unexpected error: log and return friendly error message (no raising)
            logger.error(f"Generation error with {policy.policy_type}: {e}")
            return self._create_error_response(prompt, policy.policy_type, str(e))

    
    async def get_best_response(self, prompt: str) -> tuple[str, str, Dict[str, float]]:
        """
        Generate responses from both policies and return the best one
        
        Returns:
            tuple: (best_response, selected_policy_type, policy_scores)
        """
        
        # Generate from both policies
        main_response = await self.generate_response(prompt, self.main_policy)
        enhancer_response = await self.generate_response(prompt, self.enhancer_policy)
        
        # For now, alternate between policies (can be enhanced with evaluation)
        # In full RLHF system, this would use mentor evaluation
        total_gens = self.policy_stats["total_generations"]
        
        if total_gens % 2 == 0:
            selected_policy = "main"
            selected_response = main_response
        else:
            selected_policy = "enhancer"
            selected_response = enhancer_response
        
        # Update statistics
        self.policy_stats["total_generations"] += 1
        if selected_policy == "main":
            self.policy_stats["main_selections"] += 1
        else:
            self.policy_stats["enhancer_selections"] += 1
        
        # Return response and metadata
        policy_scores = {
            "main_avg": self.main_policy.get_average_score(),
            "enhancer_avg": self.enhancer_policy.get_average_score()
        }
        
        logger.info(f"Selected {selected_policy} policy response for prompt: {prompt[:50]}...")
        
        return selected_response, selected_policy, policy_scores
    
    def update_policy_performance(self, policy_type: str, score: float):
        """Update policy performance based on evaluation score"""
        
        if policy_type == "main":
            self.main_policy.update_performance(score)
        elif policy_type == "enhancer":
            self.enhancer_policy.update_performance(score)
        else:
            logger.warning(f"Unknown policy type for performance update: {policy_type}")
        
        logger.debug(f"Updated {policy_type} policy performance: score={score:.2f}")
    
    # def adapt_policies(self, performance_data: Dict[str, Any]):
    #     """
    #     Adapt policy parameters based on recent batch statistics (batch averages and win counts).
    #     Keeps the original conservative spirit: if main is judged better, make it more conservative
    #     (lower temperature/top_p). If enhancer wins, allow more creativity.
    #     """
    #     batch_size = performance_data.get("batch_size", 1)
    #     main_wins = performance_data.get("main_wins", 0)
    #     enhancer_wins = performance_data.get("enhancer_wins", 0)
    #     batch_main_avg = float(performance_data.get("avg_main_score", 0.0))
    #     batch_enhancer_avg = float(performance_data.get("avg_enhancer_score", 0.0))

    #     logger.info(
    #         f"Adapting policies (batch_size={batch_size}) - "
    #         f"batch_main_avg={batch_main_avg:.2f}, batch_enhancer_avg={batch_enhancer_avg:.2f}, "
    #         f"main_wins={main_wins}, enhancer_wins={enhancer_wins}"
    #     )

    #     # conservative step sizes
    #     temp_step = 0.02
    #     top_p_step = 0.02
    #     min_temp = 0.2
    #     max_temp = 1.0
    #     min_top_p = 0.2
    #     max_top_p = 1.0

    #     # thresholds to avoid tiny/flapping updates
    #     win_margin_threshold = 1        # require at least 1 more win in batch
    #     avg_diff_threshold = 0.3       # require avg score difference >= 0.3

    #     reason = "no_significant_change"

    #     # 1) If clear winner by wins in batch -> prefer that signal
    #     if (main_wins - enhancer_wins) >= win_margin_threshold:
    #         # main judged better -> make main more conservative; make enhancer slightly more creative
    #         self.main_policy.temperature = max(min_temp, self.main_policy.temperature - temp_step)
    #         self.main_policy.top_p = max(min_top_p, getattr(self.main_policy, "top_p", 0.9) - top_p_step)
    #         self.enhancer_policy.temperature = min(max_temp, self.enhancer_policy.temperature + temp_step)
    #         self.enhancer_policy.top_p = min(max_top_p, getattr(self.enhancer_policy, "top_p", 0.9) + top_p_step)
    #         reason = "main_wins_majority"
    #         logger.info(f"Main won batch -> made main more conservative (temp={self.main_policy.temperature:.2f}, top_p={self.main_policy.top_p:.2f})")

    #     elif (enhancer_wins - main_wins) >= win_margin_threshold:
    #         # enhancer judged better -> make enhancer more creative; make main slightly more conservative
    #         self.enhancer_policy.temperature = min(max_temp, self.enhancer_policy.temperature + temp_step)
    #         self.enhancer_policy.top_p = min(max_top_p, getattr(self.enhancer_policy, "top_p", 0.9) + top_p_step)
    #         self.main_policy.temperature = max(min_temp, self.main_policy.temperature - temp_step)
    #         self.main_policy.top_p = max(min_top_p, getattr(self.main_policy, "top_p", 0.9) - top_p_step)
    #         reason = "enhancer_wins_majority"
    #         logger.info(f"Enhancer won batch -> made enhancer more creative (temp={self.enhancer_policy.temperature:.2f}, top_p={self.enhancer_policy.top_p:.2f})")

    #     else:
    #         # 2) No clear winner by count -> use batch average difference (if significant)
    #         avg_diff = batch_main_avg - batch_enhancer_avg
    #         if avg_diff >= avg_diff_threshold:
    #             # main avg significantly higher -> prefer conservative main (match original spirit)
    #             self.main_policy.temperature = max(0.4, self.main_policy.temperature - 0.02)
    #             reason = "batch_avg_main_high"
    #             logger.info(f"Batch avg: main better -> reduced main temp to {self.main_policy.temperature:.2f}")
    #         elif -avg_diff >= avg_diff_threshold:
    #             # enhancer avg significantly higher -> increase main creativity a bit (match original spirit)
    #             self.main_policy.temperature = min(0.8, self.main_policy.temperature + 0.02)
    #             reason = "batch_avg_enhancer_high"
    #             logger.info(f"Batch avg: enhancer better -> increased main temp to {self.main_policy.temperature:.2f}")
    #         else:
    #             logger.info("No significant difference in batch; no major temp/top_p change.")

    #     # 3) Preserve original overall checks for enhancer based on overall performance (optional)
    #     overall_avg = (batch_main_avg + batch_enhancer_avg) / 2.0
    #     if overall_avg < 6.0:
    #         self.enhancer_policy.temperature = min(1.0, self.enhancer_policy.temperature + 0.05)
    #         logger.info(f"Overall low performance -> increased enhancer temp to {self.enhancer_policy.temperature:.2f}")
    #     elif overall_avg > 8.5:
    #         self.enhancer_policy.temperature = max(0.6, self.enhancer_policy.temperature - 0.02)
    #         logger.info(f"Overall high performance -> reduced enhancer temp to {self.enhancer_policy.temperature:.2f}")

    #     # Record update in history
    #     update_record = {
    #         "timestamp": datetime.now().isoformat(),
    #         "batch_size": batch_size,
    #         "main_wins": main_wins,
    #         "enhancer_wins": enhancer_wins,
    #         "batch_main_avg": batch_main_avg,
    #         "batch_enhancer_avg": batch_enhancer_avg,
    #         "main_temperature": self.main_policy.temperature,
    #         "enhancer_temperature": self.enhancer_policy.temperature,
    #         "main_top_p": getattr(self.main_policy, "top_p", None),
    #         "enhancer_top_p": getattr(self.enhancer_policy, "top_p", None),
    #         "adaptation_reason": reason
    #     }
    #     self.update_history.append(update_record)
    #     self.policy_stats["last_policy_update"] = datetime.now()

    #     logger.info("Policy adaptation completed")
    def adapt_policies(self, performance_data: Dict[str, Any]):
        """
        Lightweight LPO-inspired policy adaptation for temperature/top_p tuning.
        Uses mentor evaluation scores to adjust creativity/conservatism dynamically.
        """

        # LPO hyperparameters (configurable)
        alpha = getattr(CONFIG, "lpo_alpha", 0.03)  # step size
        beta = getattr(CONFIG, "lpo_beta", 1.5)     # sensitivity to score difference
        min_temp, max_temp = 0.2, 1.0
        min_top_p, max_top_p = 0.2, 1.0

        # Extract performance stats
        main_avg = float(performance_data.get("avg_main_score", 0.0))
        enhancer_avg = float(performance_data.get("avg_enhancer_score", 0.0))
        delta = main_avg - enhancer_avg  # >0 means main better

        # Compute preference signal (normalized -1..1)
        signal = np.tanh(beta * delta / 10.0)  # mentor score is ~0-10

        logger.info(
            f"[LPO] main_avg={main_avg:.2f}, enhancer_avg={enhancer_avg:.2f}, "
            f"delta={delta:.2f}, signal={signal:.3f}"
        )

        # Update policies
        # Main: nếu signal > 0 -> giảm temperature (conservative hơn), ngược lại tăng
        self.main_policy.temperature = float(
            np.clip(self.main_policy.temperature - alpha * signal, min_temp, max_temp)
        )
        # Enhancer: ngược hướng với main
        self.enhancer_policy.temperature = float(
            np.clip(self.enhancer_policy.temperature + alpha * signal, min_temp, max_temp)
        )

        # Đồng thời update top_p theo cùng hướng
        self.main_policy.top_p = float(
            np.clip(self.main_policy.top_p - alpha * signal, min_top_p, max_top_p)
        )
        self.enhancer_policy.top_p = float(
            np.clip(self.enhancer_policy.top_p + alpha * signal, min_top_p, max_top_p)
        )

        # Record history
        update_record = {
            "timestamp": datetime.now().isoformat(),
            "main_avg": main_avg,
            "enhancer_avg": enhancer_avg,
            "delta": delta,
            "signal": signal,
            "main_temperature": self.main_policy.temperature,
            "enhancer_temperature": self.enhancer_policy.temperature,
            "main_top_p": self.main_policy.top_p,
            "enhancer_top_p": self.enhancer_policy.top_p,
            "alpha": alpha,
            "beta": beta
        }
        self.update_history.append(update_record)
        self.policy_stats["last_policy_update"] = datetime.now()

        logger.info(
            f"[LPO] Updated temps: Main={self.main_policy.temperature:.2f}, "
            f"Enhancer={self.enhancer_policy.temperature:.2f}, "
            f"top_p: Main={self.main_policy.top_p:.2f}, Enhancer={self.enhancer_policy.top_p:.2f}"
        )


    def _get_adaptation_reason(self, main_avg: float, enhancer_avg: float, overall_avg: float) -> str:
        """Get human-readable reason for policy adaptation"""
        
        if main_avg > enhancer_avg and main_avg > 7.5:
            return "Main policy performing well - increased conservatism"
        elif enhancer_avg > main_avg and enhancer_avg > 7.5:
            return "Enhancer policy performing well - increased main creativity"
        elif overall_avg < 6.0:
            return "Overall performance low - increased diversity"
        elif overall_avg > 8.5:
            return "High performance - increased focus"
        else:
            return "Regular adaptive adjustment"
    
    def _create_fallback_response(self, prompt: str, policy_type: str) -> str:
        """Create fallback response when generation fails"""
        return f"""I apologize, but I'm experiencing technical difficulties generating a response right now. 

However, I can suggest some general guidance for your question about: "{prompt[:100]}..."

For {CONFIG.framework_type} development at {CONFIG.target_level} level:

1. Start with the official documentation and tutorials
2. Look for community examples and best practices  
3. Try implementing small examples first
4. Join developer communities for support

Please try asking your question again, and I'll do my best to provide a detailed response.

*(Response generated by {policy_type} policy fallback)*"""
    
    def _create_error_response(self, prompt: str, policy_type: str, error: str) -> str:
        """Create error response with helpful information"""
        return f"""I encountered an error while processing your question: "{prompt[:100]}..."

Error details: {error}

This appears to be a temporary technical issue. Please try:

1. Rephrasing your question
2. Breaking complex questions into smaller parts  
3. Trying again in a few moments

I apologize for the inconvenience and appreciate your patience.

*(Error response from {policy_type} policy)*"""

    def get_policy_stats(self) -> Dict[str, Any]:
        """Get comprehensive policy statistics"""
        
        main_stats = self.main_policy.to_dict()
        enhancer_stats = self.enhancer_policy.to_dict()
        
        return {
            "main_policy": main_stats,
            "enhancer_policy": enhancer_stats,
            "selection_stats": self.policy_stats.copy(),
            "update_history_size": len(self.update_history),
            "last_update": self.policy_stats["last_policy_update"].isoformat(),
            "performance_summary": {
                "main_avg_score": self.main_policy.get_average_score(),
                "enhancer_avg_score": self.enhancer_policy.get_average_score(),
                "main_generations": self.main_policy.generation_count,
                "enhancer_generations": self.enhancer_policy.generation_count,
                "selection_ratio": {
                    "main": self.policy_stats["main_selections"] / max(1, self.policy_stats["total_generations"]),
                    "enhancer": self.policy_stats["enhancer_selections"] / max(1, self.policy_stats["total_generations"])
                }
            }
        }
    
    def get_recent_updates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent policy updates"""
        return self.update_history[-limit:] if self.update_history else []