"""
Real-time RLHF System - Complete Implementation
Educational RLHF with single API key architecture and strict mentor evaluation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import re
from config import CONFIG
from mentor_evaluator import StrictMentorEvaluator, MentorEvaluationResult
from policy_manager import PolicyManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, CONFIG.log_level.upper()),
    format=CONFIG.log_format
)
logger = logging.getLogger(__name__)

@dataclass
class PreferenceData:
    """Preference data for RLHF training"""
    prompt: str
    main_response: str
    enhancer_response: str
    main_evaluation: MentorEvaluationResult
    enhancer_evaluation: MentorEvaluationResult
    chosen_policy: str
    preference_strength: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "main_response": self.main_response[:200] + "..." if len(self.main_response) > 200 else self.main_response,
            "enhancer_response": self.enhancer_response[:200] + "..." if len(self.enhancer_response) > 200 else self.enhancer_response,
            "main_score": self.main_evaluation.composite_score,
            "enhancer_score": self.enhancer_evaluation.composite_score,
            "chosen_policy": self.chosen_policy,
            "preference_strength": self.preference_strength,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class SystemStats:
    """Real-time system statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    avg_mentor_score: float = 0.0
    main_policy_selections: int = 0
    enhancer_policy_selections: int = 0
    last_policy_update: datetime = field(default_factory=datetime.now)
    startup_time: datetime = field(default_factory=datetime.now)
    
    def update_request(self, success: bool, response_time: float, mentor_score: float, policy_type: str):
        """Update request statistics"""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            
            # Update moving averages
            alpha = 0.1  # Smoothing factor
            self.avg_response_time = alpha * response_time + (1 - alpha) * self.avg_response_time
            self.avg_mentor_score = alpha * mentor_score + (1 - alpha) * self.avg_mentor_score
            
            # Track policy selections
            if policy_type == "main":
                self.main_policy_selections += 1
            else:
                self.enhancer_policy_selections += 1
        else:
            self.failed_requests += 1
    
    def get_success_rate(self) -> float:
        return self.successful_requests / max(1, self.total_requests)
    
    def get_uptime(self) -> timedelta:
        return datetime.now() - self.startup_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "avg_response_time": self.avg_response_time,
            "avg_mentor_score": self.avg_mentor_score,
            "main_policy_selections": self.main_policy_selections,
            "enhancer_policy_selections": self.enhancer_policy_selections,
            "policy_selection_ratio": {
                "main": self.main_policy_selections / max(1, self.total_requests),
                "enhancer": self.enhancer_policy_selections / max(1, self.total_requests)
            },
            "last_policy_update": self.last_policy_update.isoformat(),
            "uptime_seconds": self.get_uptime().total_seconds(),
            "startup_time": self.startup_time.isoformat()
        }

class RealTimeRLHFSystem:
    """
    Complete real-time RLHF system for educational content generation
    
    Features:
    - Single API key architecture (Gemini 1.5-Flash + 2.5-Flash)
    - Strict mentor evaluation with 6 criteria
    - Adaptive policy management
    - Real-time preference learning
    - Comprehensive monitoring and statistics
    """
    
    def __init__(self):
        logger.info("Initializing Real-time RLHF System...")
        
        # Initialize core components
        self.mentor_evaluator = StrictMentorEvaluator()
        self.policy_manager = PolicyManager()
        
        # System statistics
        self.stats = SystemStats()
        
        # Preference data storage for RLHF learning
        self.preference_data: List[PreferenceData] = []
        self.max_preference_data = 1000  # Maximum stored preference data
        
        # Performance tracking
        self.performance_history = []
        
        # Last count used for triggering count-based updates
        self._last_update_pref_count: int = 0
        # Lock to avoid concurrent policy updates (serialize count-trigger updates)
        self._update_lock = asyncio.Lock()
        
        logger.info("Real-time RLHF System initialized successfully")
    
    async def process_learning_request(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, MentorEvaluationResult, Dict[str, Any]]:
        """
        Process a learning request and return the best response
        
        Returns:
            tuple: (best_response, evaluation_result, metadata)
        """
        start_time = datetime.now()
        request_metadata = {
            "request_timestamp": start_time.isoformat(),
            "context": context or {}
        }
        
        try:
            logger.info(f"Processing learning request: {prompt[:100]}...")
            
            # Generate responses from both policies
            main_response = await self.policy_manager.generate_response(
                prompt, self.policy_manager.main_policy
            )
            
            enhancer_response = await self.policy_manager.generate_response(
                prompt, self.policy_manager.enhancer_policy
            )
            
            # Evaluate both responses with mentor system
            logger.debug("Evaluating responses with mentor system...")
            
            main_evaluation = await self.mentor_evaluator.evaluate_response(
                prompt, main_response, context
            )
            
            enhancer_evaluation = await self.mentor_evaluator.evaluate_response(
                prompt, enhancer_response, context
            )
            
            # Select best response based on mentor evaluation
            if main_evaluation.composite_score >= enhancer_evaluation.composite_score:
                best_response = main_response
                best_evaluation = main_evaluation
                chosen_policy = "main"
                preference_strength = main_evaluation.composite_score - enhancer_evaluation.composite_score
            else:
                best_response = enhancer_response
                best_evaluation = enhancer_evaluation
                chosen_policy = "enhancer"
                preference_strength = enhancer_evaluation.composite_score - main_evaluation.composite_score
            
            # Store preference data for RLHF learning
            preference_entry = PreferenceData(
                prompt=prompt,
                main_response=main_response,
                enhancer_response=enhancer_response,
                main_evaluation=main_evaluation,
                enhancer_evaluation=enhancer_evaluation,
                chosen_policy=chosen_policy,
                preference_strength=preference_strength
            )
            
            await self._store_preference_data(preference_entry)
            
            # Update policy performance
            self.policy_manager.update_policy_performance("main", main_evaluation.composite_score)
            self.policy_manager.update_policy_performance("enhancer", enhancer_evaluation.composite_score)
            
            # Calculate response time and update statistics
            response_time = (datetime.now() - start_time).total_seconds()
            self.stats.update_request(True, response_time, best_evaluation.composite_score, chosen_policy)
            
            # Prepare response metadata
            request_metadata.update({
                "response_time": response_time,
                "chosen_policy": chosen_policy,
                "preference_strength": preference_strength,
                "main_score": main_evaluation.composite_score,
                "enhancer_score": enhancer_evaluation.composite_score,
                "policy_scores": {
                    "main_avg": self.policy_manager.main_policy.get_average_score(),
                    "enhancer_avg": self.policy_manager.enhancer_policy.get_average_score()
                }
            })
            
            logger.info(f"Request processed successfully in {response_time:.2f}s - "
                       f"Score: {best_evaluation.composite_score:.2f}, Policy: {chosen_policy}")
            
            return best_response, best_evaluation, request_metadata
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self.stats.update_request(False, response_time, 0.0, "error")
            
            logger.error(f"Error processing learning request: {e}")
            raise e
        
    async def _store_preference_data(self, preference_data: PreferenceData):
        """Store preference data with size management + count-trigger"""
        self.preference_data.append(preference_data)

        # Maintain maximum size
        if len(self.preference_data) > self.max_preference_data:
            self.preference_data = self.preference_data[-self.max_preference_data:]

        logger.debug(f"Stored preference data - Total: {len(self.preference_data)}")

        # Count-trigger: chỉ update khi đủ batch size và chạy ngay sau request này
        if getattr(CONFIG, "enable_count_trigger", False):
            pending = len(self.preference_data) - getattr(self, "_last_update_pref_count", 0)
            if pending >= getattr(CONFIG, "preference_batch_size", 3):
                logger.info(
                    f"Collected {pending} new preferences (batch_size={CONFIG.preference_batch_size}) -> running count-trigger policy update now."
                )
                await self._run_count_trigger_update()

    async def _run_count_trigger_update(self):
        pending = len(self.preference_data) - getattr(self, "_last_update_pref_count", 0)
        if pending < getattr(CONFIG, "preference_batch_size", 3):
            logger.info("Count-trigger found insufficient new preferences; skipping.")
            return
        try:
            logger.info("🚀 Starting count-trigger policy update...")
            await self.update_policies_from_preferences(trigger_type="count")
            self._last_update_pref_count = len(self.preference_data)
            self.stats.last_policy_update = datetime.now()
            logger.info(f"✔️ Count-trigger update completed at {self._last_update_pref_count} preferences.")
        except Exception as e:
            logger.error(f"Count-trigger update error: {e}")


    async def update_policies_from_preferences(self, batch_size: Optional[int] = None, trigger_type: str = "count"):
        """
        Update policies based on recent preference data
        """
        # Serialize updates using lock to avoid concurrent adaptations
        async with self._update_lock:
            if not self.preference_data:
                logger.info("No preference data available for policy update")
                return

            batch_size = batch_size or CONFIG.preference_batch_size
            recent_data = self.preference_data[-batch_size:] if len(self.preference_data) >= batch_size else self.preference_data

            logger.info(f"Updating policies based on {len(recent_data)} recent preferences (trigger={trigger_type})")

            # Analyze performance patterns
            main_wins = sum(1 for d in recent_data if d.chosen_policy == "main")
            enhancer_wins = sum(1 for d in recent_data if d.chosen_policy == "enhancer")

            avg_main_score = np.mean([d.main_evaluation.composite_score for d in recent_data])
            avg_enhancer_score = np.mean([d.enhancer_evaluation.composite_score for d in recent_data])
            avg_preference_strength = np.mean([d.preference_strength for d in recent_data])

            # Prepare performance data for policy adaptation
            performance_data = {
                "batch_size": len(recent_data),
                "main_wins": main_wins,
                "enhancer_wins": enhancer_wins,
                "avg_main_score": avg_main_score,
                "avg_enhancer_score": avg_enhancer_score,
                "avg_preference_strength": avg_preference_strength,
                "trigger": trigger_type,
                "win_ratio": main_wins / len(recent_data)
            }

            # Update policies
            try:
                self.policy_manager.adapt_policies(performance_data)
                logger.info(
                f"New temperatures -> Main: {self.policy_manager.main_policy.temperature:.2f}, "
                f"Enhancer: {self.policy_manager.enhancer_policy.temperature:.2f}"
)
            except Exception as e:
                logger.error(f"Policy adaptation error: {e}")
                raise

            # Record performance history
            performance_record = {
                "timestamp": datetime.now().isoformat(),
                **performance_data,
                "policy_temperatures": {
                    "main": self.policy_manager.main_policy.temperature,
                    "enhancer": self.policy_manager.enhancer_policy.temperature
                }
            }

            self.performance_history.append(performance_record)
            self.stats.last_policy_update = datetime.now()

            logger.info(f"Policy update completed - Main wins: {main_wins}, "
                       f"Enhancer wins: {enhancer_wins}, Avg scores: M={avg_main_score:.2f}, E={avg_enhancer_score:.2f}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "system_stats": self.stats.to_dict(),
            "policy_stats": self.policy_manager.get_policy_stats(),
            "evaluator_stats": self.mentor_evaluator.get_evaluation_stats(),
            "preference_data_stats": {
                "total_entries": len(self.preference_data),
                "recent_entries_24h": self._count_recent_preferences(hours=24),
                "recent_entries_1h": self._count_recent_preferences(hours=1)
            },
            "performance_history_size": len(self.performance_history),
            "system_health": self._assess_system_health()
        }
    
    def _count_recent_preferences(self, hours: int) -> int:
        """Count preference entries within the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return sum(1 for p in self.preference_data if p.timestamp >= cutoff)
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        
        health_status = "healthy"
        issues = []
        
        # Check success rate
        success_rate = self.stats.get_success_rate()
        if success_rate < 0.9:
            health_status = "degraded"
            issues.append(f"Low success rate: {success_rate:.2%}")
        
        # Check average response time
        if self.stats.avg_response_time > 10.0:
            health_status = "degraded"
            issues.append(f"High response time: {self.stats.avg_response_time:.2f}s")
        
        # Check mentor score
        if self.stats.avg_mentor_score < 6.0:
            health_status = "degraded"
            issues.append(f"Low mentor scores: {self.stats.avg_mentor_score:.2f}")
        
        # Check if system is receiving requests
        if self.stats.total_requests == 0:
            health_status = "idle"
            issues.append("No requests processed yet")
        
        return {
            "status": health_status,
            "issues": issues,
            "uptime": self.stats.get_uptime().total_seconds(),
            "last_activity": self.stats.last_policy_update.isoformat() if self.stats.total_requests > 0 else None
        }
    
    def get_recent_performance_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance history"""
        return self.performance_history[-limit:] if self.performance_history else []
    
    def get_recent_preferences(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent preference data"""
        return [p.to_dict() for p in self.preference_data[-limit:]] if self.preference_data else []
    
    async def cleanup(self):
        """Cleanup system resources"""
        logger.info("Cleaning up RLHF system resources...")
        
        # Clear caches (call only if exists)
        clear_fn = getattr(self.mentor_evaluator, "clear_cache", None)
        if callable(clear_fn):
            try:
                clear_fn()
            except Exception as e:
                logger.warning(f"mentor_evaluator.clear_cache() error: {e}")
        else:
            logger.debug("mentor_evaluator.clear_cache() not available.")
        
        # Log final statistics
        logger.info(f"Final system stats: {self.stats.to_dict()}")
        logger.info("RLHF system cleanup completed")

# Global system instance
rlhf_system: Optional[RealTimeRLHFSystem] = None

# FastAPI application with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global rlhf_system
    
    # Startup
    logger.info("Starting Educational RLHF Real-time System")
    
    try:
        rlhf_system = RealTimeRLHFSystem()
        logger.info("RLHF system startup completed")
        
        # No time-based scheduler: policy updates triggered only by count-trigger (n preferences)
        # System will update policies when enough preference data accumulates.

        yield

        # Shutdown
        logger.info("Shutting down RLHF system...")
        if rlhf_system:
            await rlhf_system.cleanup()
            
    except Exception as e:
        logger.error(f"System initialization error: {e}")
        raise

# FastAPI application
app = FastAPI(
    title="Educational RLHF Real-time System",
    description="Real-time RLHF system with strict mentor evaluation for educational content generation",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class LearnRequest(BaseModel):
    prompt: str
    framework: Optional[str] = None
    level: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class LearnResponse(BaseModel):
    answer: str
    mentor_evaluation: Dict[str, Any]
    response_metadata: Dict[str, Any]

# API Endpoints
@app.post("/learn", response_model=LearnResponse)
async def learn(request: LearnRequest):
    """
    Main learning endpoint - process educational queries with RLHF
    """
    if not rlhf_system:
        raise HTTPException(status_code=503, detail="RLHF system not initialized")
    
    try:
        # Process the learning request
        response, evaluation, metadata = await rlhf_system.process_learning_request(
            prompt=request.prompt,
            context=request.context
        )
        
        return LearnResponse(
            answer=response,
            mentor_evaluation=evaluation.to_dict(),
            response_metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Learning endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing learning request: {str(e)}")

@app.get("/status")
async def get_system_status():
    """Get comprehensive system status and statistics"""
    if not rlhf_system:
        raise HTTPException(status_code=503, detail="RLHF system not initialized")
    
    return rlhf_system.get_system_status()

@app.get("/performance")
async def get_performance_metrics():
    """Get performance metrics and history"""
    if not rlhf_system:
        raise HTTPException(status_code=503, detail="RLHF system not initialized")
    
    return {
        "performance_history": rlhf_system.get_recent_performance_history(20),
        "recent_preferences": rlhf_system.get_recent_preferences(10),
        "policy_stats": rlhf_system.policy_manager.get_policy_stats(),
        "evaluator_stats": rlhf_system.mentor_evaluator.get_evaluation_stats()
    }

@app.post("/admin/update-policies")
async def trigger_policy_update(background_tasks: BackgroundTasks, batch_size: Optional[int] = None):
    """Manually trigger policy update"""
    if not rlhf_system:
        raise HTTPException(status_code=503, detail="RLHF system not initialized")
    
    # Pass trigger type 'manual' so the adapt logic knows invocation source
    background_tasks.add_task(rlhf_system.update_policies_from_preferences, batch_size, "manual")
    
    return {
        "message": "Policy update triggered",
        "batch_size": batch_size or CONFIG.preference_batch_size,
        "available_data": len(rlhf_system.preference_data)
    }

@app.post("/admin/clear-cache")
async def clear_evaluation_cache():
    """Clear evaluation cache"""
    if not rlhf_system:
        raise HTTPException(status_code=503, detail="RLHF system not initialized")
    
    rlhf_system.mentor_evaluator.clear_cache()
    
    return {"message": "Evaluation cache cleared"}

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    if not rlhf_system:
        return {
            "status": "unhealthy",
            "message": "RLHF system not initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    health_info = rlhf_system._assess_system_health()
    
    return {
        "status": health_info["status"],
        "uptime": health_info["uptime"],
        "issues": health_info["issues"],
        "config": {
            "framework": CONFIG.framework_type,
            "level": CONFIG.target_level,
            "models": {
                "base": CONFIG.base_model,
                "evaluator": CONFIG.evaluator_model
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "service": "Educational RLHF Real-time System",
        "version": "1.0.0",
        "description": "Real-time RLHF system with strict mentor evaluation",
        "endpoints": {
            "learn": "POST /learn - Process educational queries",
            "status": "GET /status - System status and statistics", 
            "performance": "GET /performance - Performance metrics",
            "health": "GET /health - Health check"
        },
        "config": {
            "framework": CONFIG.framework_type,
            "target_level": CONFIG.target_level
        },
        "timestamp": datetime.now().isoformat()
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=CONFIG.host,
        port=CONFIG.port,
        reload=CONFIG.debug,
        log_level=CONFIG.log_level.lower()
    )