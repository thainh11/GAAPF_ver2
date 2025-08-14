"""
Mentor Evaluation System for Educational RLHF
Strict 6-criteria evaluation using Gemini 2.5-Flash
"""

import json
import asyncio
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import google.generativeai as genai
from typing import List, Tuple
import re
from config import CONFIG
import os
import math

logger = logging.getLogger(__name__)
os.environ["RLHF_DEV_DISABLE_SAFETY"] = "1"

@dataclass
class MentorRewardComponents:
    """6 strict mentor evaluation criteria with weights"""
    conceptual_accuracy: float = 0.0      # Technical correctness (20%)
    pedagogical_flow: float = 0.0         # Teaching progression (20%)
    practical_relevance: float = 0.0      # Real-world application (20%)
    clarity_precision: float = 0.0        # Clear communication (15%)
    engagement_motivation: float = 0.0     # Student engagement (15%)
    depth_insights: float = 0.0           # Deep insights (10%)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "conceptual_accuracy": self.conceptual_accuracy,
            "pedagogical_flow": self.pedagogical_flow,
            "practical_relevance": self.practical_relevance,
            "clarity_precision": self.clarity_precision,
            "engagement_motivation": self.engagement_motivation,
            "depth_insights": self.depth_insights
        }
    
    def get_weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        scores = self.to_dict()
        return sum(weights.get(k, 0.0) * v for k, v in scores.items())

@dataclass
class MentorEvaluationResult:
    """Complete mentor evaluation result"""
    prompt: str
    response: str
    reward_components: MentorRewardComponents
    composite_score: float
    reasoning: str
    mentor_feedback: str
    improvement_suggestions: list = field(default_factory=list)
    strengths: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response[:100] + "..." if len(self.response) > 100 else self.response,
            "reward_components": self.reward_components.to_dict(),
            "composite_score": self.composite_score,
            "reasoning": self.reasoning,
            "mentor_feedback": self.mentor_feedback,
            "improvement_suggestions": self.improvement_suggestions,
            "strengths": self.strengths,
            "metadata": self.metadata
        }

class StrictMentorEvaluator:
    """
    Strict AI mentor evaluator using Gemini 2.5-Flash
    
    Evaluates educational responses on 6 rigorous criteria:
    1. Conceptual Accuracy - Technical correctness
    2. Pedagogical Flow - Teaching progression 
    3. Practical Relevance - Real-world application
    4. Clarity & Precision - Communication quality
    5. Engagement & Motivation - Student connection
    6. Depth & Insights - Expert-level insights
    """
    
    def __init__(self):
        # Initialize Gemini model
        genai.configure(api_key=CONFIG.gemini_api_key)
        self.evaluator_model = genai.GenerativeModel(CONFIG.evaluator_model)
        
        # Use configured weights
        self.weights = CONFIG.mentor_weights.to_dict()
        
        # Evaluation cache to avoid duplicate evaluations
        self.evaluation_cache = {}
        
        # Performance tracking
        self.evaluation_stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "fallback_evaluations": 0,
            "avg_evaluation_time": 0.0
        }
        
        logger.info(f"Mentor Evaluator initialized with weights: {self.weights}")
    
    def _create_mentor_evaluation_prompt(self, prompt: str, response: str) -> str:
        return f"""
You are an AI MENTOR, professional in evaluation LLM responses. Rate the RESPONSE to the QUESTION using these criteria (0-10, strict scoring):

1. CONCEPTUAL ACCURACY – Technical correctness, no factual errors.
2. PEDAGOGICAL FLOW – Logical, well-structured teaching progression.
3. PRACTICAL RELEVANCE – Useful, real-world applicable content.
4. CLARITY & PRECISION – Clear, concise, unambiguous language.
5. ENGAGEMENT & MOTIVATION – Keeps learner interested and motivated.
6. DEPTH & INSIGHTS – Goes beyond basics, provides deeper understanding.

QUESTION:
{prompt}

RESPONSE:
{response}

Return ONLY valid JSON in this exact format:
{{
    "conceptual_accuracy": 0-10,
    "pedagogical_flow": 0-10,
    "practical_relevance": 0-10,
    "clarity_precision": 0-10,
    "engagement_motivation": 0-10,
    "depth_insights": 0-10
}}
"""
    def _clean_response_text(self, raw: str) -> str:
        """Remove markdown fences/backticks and trivial outputs, return cleaned text or '' if empty"""
        if raw is None:
            return ""
        text = raw.strip()

        # Remove triple backtick blocks like ```json ... ```
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)

        # Remove any leading/trailing single backticks or quotes
        text = re.sub(r"^`+", "", text)
        text = re.sub(r"`+$", "", text)
        text = text.strip()

        # If text contains nothing or only punctuation/backticks, treat as empty
        if not text or re.fullmatch(r"[`\s'\".]+", text):
            return ""

        return text
    def _sanitize_prompt(self, prompt: str) -> str:
        """Basic sanitize for evaluator prompts (reuse same blacklist idea)."""
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

    async def evaluate_response(self, prompt: str, response: str, 
                              context: Optional[Dict[str, Any]] = None) -> MentorEvaluationResult:
        """
        Perform strict mentor evaluation with comprehensive retry logic and safety-aware retries
            """
        start_time = datetime.now()
        
        # Check cache first
        cache_key = hash(prompt + response + str(context))
        if cache_key in self.evaluation_cache:
            cached_result = self.evaluation_cache[cache_key]
            logger.debug(f"Using cached evaluation for prompt: {prompt[:50]}...")
            return cached_result
        
        # Create evaluation prompt
        evaluation_prompt = self._create_mentor_evaluation_prompt(prompt, response)
        
        # Attempt evaluation with robust retry mechanism
        for attempt in range(3):
            try:
                logger.debug(f"Mentor evaluation attempt {attempt + 1} for: {prompt[:50]}...")
                
                # Call Gemini 2.5-Flash for evaluation
                result = await self.evaluator_model.generate_content_async(
                    evaluation_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,  # Low temperature for consistent scoring
                        max_output_tokens=CONFIG.max_output_tokens,
                        candidate_count=1
                    ),
                    safety_settings=self._get_safety_settings()
                )
                # logger.debug(f"=== RAW GEMINI API RESULT === {result}")
                # print("=== DEBUG GEMINI RESULT ===", result)

                # Basic candidate existence check
                if not getattr(result, "candidates", None) or len(result.candidates) == 0:
                    logger.warning(f"No candidates returned on attempt {attempt + 1}")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        return self._create_fallback_evaluation(prompt, response, "empty_candidates")

                # --- normalize finish_reason and robustly handle truncation ---
                cand = result.candidates[0]
                finish_reason = getattr(cand, "finish_reason", None)

                # Normalize finish reason (support string enum or numeric enum)
                fr_norm = ""
                if finish_reason is None:
                    fr_norm = ""
                elif isinstance(finish_reason, int):
                    # In practice proto enum '2' corresponds to MAX_TOKENS for many clients — treat 2 as MAX_TOKENS
                    fr_norm = "MAX_TOKENS" if finish_reason == 2 else str(finish_reason)
                else:
                    fr_norm = str(finish_reason).upper()

                logger.warning(f"Candidate finish_reason: {finish_reason} (normalized: {fr_norm})")

                # If truncated / max tokens, immediately retry with compact JSON-only prompt (but not infinite loop)
                if ("MAX_TOKENS" in fr_norm or "MAX" in fr_norm) and attempt < 2:
                    logger.warning("Model hit MAX_TOKENS (truncated). Retrying with compact JSON-only prompt.")
                    short_eval_prompt = (
                        f"Return ONLY a JSON object with numeric scores 0-10 for keys: "
                        f"conceptual_accuracy,pedagogical_flow,practical_relevance,clarity_precision,"
                        f"engagement_motivation,depth_insights.\nQUESTION: {prompt}\nRESPONSE: {response}\n"
                    )
                    try:
                        retry_max_tokens = min(CONFIG.max_output_tokens * 2, 4096)
                        retry_result = await self.evaluator_model.generate_content_async(
                            short_eval_prompt,
                            generation_config=genai.GenerationConfig(
                                temperature=0.0,
                                max_output_tokens=retry_max_tokens,
                                candidate_count=1
                            ),
                            safety_settings=self._get_safety_settings()
                        )
                        logger.debug("Retry (short prompt) raw result: %s", retry_result)
                        if getattr(retry_result, "candidates", None):
                            cand = retry_result.candidates[0]
                        else:
                            # keep original cand (will be handled below)
                            logger.warning("Retry returned no candidates")
                    except Exception as e:
                        logger.error("Retry after MAX_TOKENS failed: %s", e)
                        # fallthrough to existing retry logic / fallback

                # Now check content parts
                if not cand or not hasattr(cand, "content") or not hasattr(cand.content, "parts") or len(cand.content.parts) == 0:
                    logger.warning(f"Empty or invalid response structure on attempt {attempt + 1} (no parts)")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        return self._create_fallback_evaluation(prompt, response, "empty_response")

                # Safely concatenate all part texts (some responses may split across parts)
                try:
                    parts = cand.content.parts
                    raw_text = "".join(getattr(p, "text", "") for p in parts)
                except Exception as e:
                    logger.error(f"Error reading parts on attempt {attempt + 1}: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        return self._create_fallback_evaluation(prompt, response, "text_extraction_error")

                cleaned = self._clean_response_text(raw_text)
                if not cleaned:
                    logger.warning(f"Cleaned response is empty on attempt {attempt + 1} (raw_len={len(raw_text)})")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        return self._create_fallback_evaluation(prompt, response, "empty_after_clean")

                logger.debug(f"Raw evaluation response (cleaned): {cleaned[:400]}...")

                # Parse and validate JSON
                evaluation_data = self._parse_evaluation_json(cleaned)
                if not evaluation_data:
                    logger.warning("Parsed evaluation_data is None (json parsing/validation failed)")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        return self._create_fallback_evaluation(prompt, response, "json_parse_error")
                
                # Create evaluation result
                mentor_result = self._create_evaluation_result(
                    prompt, response, evaluation_data, context
                )
                
                # Cache successful evaluation
                self.evaluation_cache[cache_key] = mentor_result
                
                # Update statistics
                evaluation_time = (datetime.now() - start_time).total_seconds()
                self._update_evaluation_stats(evaluation_time, success=True)
                
                logger.info(f"Mentor evaluation completed: Score {mentor_result.composite_score:.2f} "
                           f"in {evaluation_time:.2f}s (attempt {attempt + 1})")
                
                return mentor_result
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                
            except Exception as e:
                logger.error(f"Evaluation error on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
        
        # All attempts failed - return fallback
        logger.error("All mentor evaluation attempts failed, using fallback")
        evaluation_time = (datetime.now() - start_time).total_seconds()
        self._update_evaluation_stats(evaluation_time, success=False)
        
        return self._create_fallback_evaluation(prompt, response, "all_attempts_failed")
    
    def _parse_evaluation_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse and validate JSON response from evaluator (robust to short JSON)"""
        try:
            cleaned_text = response_text.strip()
            # If the model returned markdown fences that _clean_response_text missed, remove again
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.I)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

            # Extract JSON object from text (existing logic)
            json_blob = self._extract_json_from_text(cleaned_text)

            evaluation_data = json.loads(json_blob)

            # REQUIRED numeric score fields (must be present)
            required_fields = [
                "conceptual_accuracy", "pedagogical_flow", "practical_relevance",
                "clarity_precision", "engagement_motivation", "depth_insights"
            ]

            for field in required_fields:
                if field not in evaluation_data:
                    logger.warning(f"Missing required numeric field: {field}")
                    return None

            # Convert values to floats and clamp 0..10
            for field in required_fields:
                try:
                    val = evaluation_data[field]
                    # accept strings like "7.5" or ints
                    if isinstance(val, str):
                        val = val.strip()
                        # remove trailing '%' or other punctuation if any
                        val = re.sub(r"[^\d\.\-]+", "", val)
                        valf = float(val) if val != "" else 0.0
                    else:
                        valf = float(val)
                except Exception as e:
                    logger.warning(f"Cannot parse score for {field}: {evaluation_data[field]} ({e}), defaulting to 0.0")
                    valf = 0.0
                evaluation_data[field] = max(0.0, min(10.0, valf))

            # reasoning / mentor_feedback are optional now
            return evaluation_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.debug("Attempted to parse cleaned text: %s", cleaned_text[:400])
            return None
        except Exception as e:
            logger.error(f"Evaluation data validation failed: {e}")
            return None
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from potentially messy response text"""
        # Find first opening brace and last closing brace
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx]
        
        # If no braces found, return original text
        return text
    
    def _create_evaluation_result(self, prompt: str, response: str, 
                                evaluation_data: Dict[str, Any],
                                context: Optional[Dict[str, Any]] = None) -> MentorEvaluationResult:
        """Create MentorEvaluationResult from evaluation data"""
        
        # Create reward components
        reward_components = MentorRewardComponents(
            conceptual_accuracy=float(evaluation_data["conceptual_accuracy"]),
            pedagogical_flow=float(evaluation_data["pedagogical_flow"]),
            practical_relevance=float(evaluation_data["practical_relevance"]),
            clarity_precision=float(evaluation_data["clarity_precision"]),
            engagement_motivation=float(evaluation_data["engagement_motivation"]),
            depth_insights=float(evaluation_data["depth_insights"])
        )
        
        # Calculate composite score
        composite_score = reward_components.get_weighted_score(self.weights)
        
        # Extract feedback components
        strengths = evaluation_data.get("strengths", [])
        improvement_suggestions = evaluation_data.get("improvement_suggestions", [])
        
        # Create metadata
        metadata = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluator_model": CONFIG.evaluator_model,
            "framework": CONFIG.framework_type,
            "target_level": CONFIG.target_level,
            "overall_assessment": evaluation_data.get("overall_assessment", ""),
            "grade_level": evaluation_data.get("grade_level", "N/A"),
            "weights_used": self.weights.copy(),
            "context": context or {}
        }
        
        return MentorEvaluationResult(
            prompt=prompt,
            response=response,
            reward_components=reward_components,
            composite_score=composite_score,
            reasoning=evaluation_data.get("reasoning", ""),
            mentor_feedback=evaluation_data.get("mentor_feedback", ""),
            improvement_suggestions=improvement_suggestions,
            strengths=strengths,
            metadata=metadata
        )
    
    def _create_fallback_evaluation(self, prompt: str, response: str, 
                                  reason: str) -> MentorEvaluationResult:
        """Create fallback evaluation when API evaluation fails"""
        
        logger.warning(f"Creating fallback evaluation due to: {reason}")
        
        # Heuristic-based evaluation
        response_length = len(response.split())
        
        # Basic quality indicators
        has_examples = any(indicator in response.lower() 
                          for indicator in ["example", "for instance", "such as", "like"])
        has_code = any(indicator in response 
                      for indicator in ["```", "def ", "import ", "class "])
        has_explanations = any(indicator in response.lower()
                             for indicator in ["because", "this means", "in other words", "specifically"])
        is_structured = any(indicator in response 
                           for indicator in ["1.", "2.", "- ", "* ", "## "])
        has_errors = any(error in response.lower()
                        for error in ["error", "empty", "unable", "sorry", "apologize"])
        
        # Calculate heuristic scores
        base_score = 5.0  # Neutral starting point for mentor evaluation
        
        # Adjust based on content quality indicators
        if response_length > 100:
            base_score += 0.5
        if response_length > 200:
            base_score += 0.3
        if has_examples:
            base_score += 0.8
        if has_code:
            base_score += 0.7
        if has_explanations:
            base_score += 0.6
        if is_structured:
            base_score += 0.4
        if has_errors:
            base_score -= 2.5
        
        # Clamp to mentor scoring range
        base_score = max(2.0, min(8.5, base_score))
        
        # Create component scores with slight variations
        components = MentorRewardComponents(
            conceptual_accuracy=max(1.0, base_score - 0.5),
            pedagogical_flow=max(1.0, base_score - 0.3),
            practical_relevance=max(1.0, base_score - 0.2),
            clarity_precision=base_score,
            engagement_motivation=max(1.0, base_score - 0.4),
            depth_insights=max(1.0, base_score - 0.8)
        )
        
        composite_score = components.get_weighted_score(self.weights)
        
        return MentorEvaluationResult(
            prompt=prompt,
            response=response,
            reward_components=components,
            composite_score=composite_score,
            reasoning=f"Fallback heuristic evaluation (reason: {reason})",
            mentor_feedback="Unable to provide detailed mentor feedback due to evaluation system error. "
                          "Response shows basic quality indicators but requires manual review.",
            improvement_suggestions=[
                "Ensure response system is functioning properly",
                "Add more specific examples and explanations",
                "Include practical code samples when relevant"
            ],
            strengths=["Response was generated successfully"] if not has_errors else [],
            metadata={
                "evaluation_method": "heuristic_fallback",
                "fallback_reason": reason,
                "evaluation_timestamp": datetime.now().isoformat(),
                "quality_indicators": {
                    "has_examples": has_examples,
                    "has_code": has_code,
                    "has_explanations": has_explanations,
                    "is_structured": is_structured,
                    "has_errors": has_errors,
                    "response_length": response_length
                }
            }
        )
    
    def _update_evaluation_stats(self, evaluation_time: float, success: bool):
        """Update evaluation performance statistics"""
        self.evaluation_stats["total_evaluations"] += 1
        
        if success:
            self.evaluation_stats["successful_evaluations"] += 1
        else:
            self.evaluation_stats["fallback_evaluations"] += 1
        
        # Update moving average for evaluation time
        total_evals = self.evaluation_stats["total_evaluations"]
        current_avg = self.evaluation_stats["avg_evaluation_time"]
        self.evaluation_stats["avg_evaluation_time"] = (
            (current_avg * (total_evals - 1) + evaluation_time) / total_evals
        )
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get current evaluation performance statistics"""
        total = self.evaluation_stats["total_evaluations"]
        successful = self.evaluation_stats["successful_evaluations"]
        fallback = self.evaluation_stats["fallback_evaluations"]
        
        return {
            **self.evaluation_stats,
            "success_rate": successful / total if total > 0 else 0.0,
            "fallback_rate": fallback / total if total > 0 else 0.0,
            "cache_size": len(self.evaluation_cache),
            "weights_config": self.weights.copy()
        }
    
    def clear_cache(self):
        """Clear evaluation cache"""
        cache_size = len(self.evaluation_cache)
        self.evaluation_cache.clear()
        logger.info(f"Cleared evaluation cache ({cache_size} entries)")
    
    async def batch_evaluate(self, evaluation_requests: List[Tuple[str, str]], 
                           max_concurrent: int = 3) -> List[MentorEvaluationResult]:
        """
        Evaluate multiple responses concurrently with rate limiting
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(prompt: str, response: str) -> MentorEvaluationResult:
            async with semaphore:
                result = await self.evaluate_response(prompt, response)

                await asyncio.sleep(0.5)
                return result
        
        tasks = [
            evaluate_with_semaphore(prompt, response)
            for prompt, response in evaluation_requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch evaluation error for item {i}: {result}")
                prompt, response = evaluation_requests[i]
                fallback_result = self._create_fallback_evaluation(
                    prompt, response, f"batch_error: {str(result)}"
                )
                processed_results.append(fallback_result)
            else:
                processed_results.append(result)
        
        logger.info(f"Batch evaluation completed: {len(processed_results)} results")
        return processed_results