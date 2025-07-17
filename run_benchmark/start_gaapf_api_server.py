#!/usr/bin/env python3
"""
GAAPF API Server for AgentBench Evaluation

This script starts a FastAPI server that interfaces with the GAAPF system
for benchmark evaluation using AgentBench.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the GAAPF source directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install fastapi uvicorn")
    sys.exit(1)

try:
    from GAAPF.core.core.constellation import Constellation
    from GAAPF.core.agents.registry import agent_registry
    from langchain_google_vertexai import ChatVertexAI
    import google.auth
    from google.oauth2 import service_account
except ImportError as e:
    print(f"GAAPF import error: {e}")
    print("Please ensure GAAPF and langchain-google-vertexai are properly installed")
    print("Install with: pip install langchain-google-vertexai google-cloud-aiplatform")
    sys.exit(1)

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gaapf_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Create separate loggers for input/output tracking
input_logger = logging.getLogger('gaapf.input')
output_logger = logging.getLogger('gaapf.output')
performance_logger = logging.getLogger('gaapf.performance')

# Configure file handlers for detailed logging
input_handler = logging.FileHandler('gaapf_inputs.log')
output_handler = logging.FileHandler('gaapf_outputs.log')
performance_handler = logging.FileHandler('gaapf_performance.log')

# Set formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
input_handler.setFormatter(detailed_formatter)
output_handler.setFormatter(detailed_formatter)
performance_handler.setFormatter(detailed_formatter)

# Add handlers to loggers
input_logger.addHandler(input_handler)
output_logger.addHandler(output_handler)
performance_logger.addHandler(performance_handler)

# Set log levels
input_logger.setLevel(logging.INFO)
output_logger.setLevel(logging.INFO)
performance_logger.setLevel(logging.INFO)

# FastAPI app
app = FastAPI(
    title="GAAPF API Server",
    description="API server for GAAPF agent evaluation with AgentBench",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = "benchmark_user"

class AgentResponse(BaseModel):
    response: str
    agent_type: Optional[str] = None
    status: str = "success"
    error: Optional[str] = None

# Global constellation instance
constellation: Optional[Constellation] = None

def initialize_gaapf_system():
    """
    Initialize the GAAPF system with Vertex AI configuration.
    """
    global constellation
    
    try:
        # Add current directory to Python path to ensure imports work
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Initialize Vertex AI credentials and configuration
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        logger.info(f"Initializing Vertex AI with project: {project_id}, location: {location}")
        
        # Handle different authentication methods
        credentials = None
        if credentials_path and os.path.exists(credentials_path):
            logger.info(f"Using service account credentials from: {credentials_path}")
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
        else:
            logger.info("Using Application Default Credentials (ADC)")
            try:
                credentials, _ = google.auth.default()
            except Exception as auth_error:
                logger.warning(f"ADC not available: {auth_error}")
                logger.info("Attempting to use Vertex AI without explicit credentials")
        
        # Initialize Vertex AI LLM
        llm_kwargs = {
            "model": "gemini-2.0-flash-001",  # Using Gemini 2.0 Flash for better performance
            "temperature": 0.7,
            "max_tokens": 8192,
            "max_retries": 3,
        }
        
        if project_id:
            llm_kwargs["project"] = project_id
        if location:
            llm_kwargs["location"] = location
        if credentials:
            llm_kwargs["credentials"] = credentials
        
        llm = ChatVertexAI(**llm_kwargs)
        
        logger.info(f"Vertex AI LLM initialized with model: {llm_kwargs['model']}")
        performance_logger.info(f"LLM_CONFIG: {llm_kwargs}")
        
        # Initialize agent registry
        agent_registry.initialize_default_agents()
        logger.info("Agent registry initialized")
        
        # Create constellation
        memory_path = project_root / "memory" / "benchmark"
        memory_path.mkdir(parents=True, exist_ok=True)
        
        constellation = Constellation(
            llm=llm,
            constellation_type="learning",
            user_id="benchmark_user",
            memory_path=memory_path,
            is_logging=True
        )
        
        logger.info("GAAPF system initialized successfully with Vertex AI")
        performance_logger.info("SYSTEM_INIT: GAAPF constellation created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize GAAPF system: {e}")
        performance_logger.error(f"SYSTEM_INIT_ERROR: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """
    Initialize the GAAPF system when the server starts.
    """
    import time
    start_time = time.time()
    
    logger.info("Starting GAAPF API Server...")
    
    # Change to parent directory to access templates and src
    import os
    original_cwd = os.getcwd()
    parent_dir = os.path.dirname(os.getcwd())
    os.chdir(parent_dir)
    
    try:
        success = initialize_gaapf_system()
        logger.info("GAAPF system initialized successfully")
        performance_logger.info(f"SYSTEM_INIT_SUCCESS: Initialization completed in {time.time() - start_time:.2f} seconds")
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        if not success:
            logger.error("Failed to initialize GAAPF system")
            raise RuntimeError("GAAPF initialization failed")
    except Exception as e:
        # Change back to original directory in case of error
        os.chdir(original_cwd)
        logger.error(f"Failed to initialize GAAPF system: {e}")
        performance_logger.error(f"SYSTEM_INIT_ERROR: {e}")
        raise RuntimeError("GAAPF initialization failed")

@app.get("/")
async def root():
    """
    Health check endpoint.
    """
    return {
        "message": "GAAPF API Server is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint.
    """
    global constellation
    
    health_status = {
        "status": "healthy" if constellation is not None else "unhealthy",
        "constellation_initialized": constellation is not None,
        "available_agents": list(constellation.agents.keys()) if constellation else []
    }
    
    return health_status

@app.post("/", response_model=AgentResponse)
async def process_question(request: QuestionRequest):
    """
    Main endpoint for processing questions through the GAAPF system.
    This endpoint matches the format expected by AgentBench.
    """
    global constellation
    
    import time
    import json
    
    # Start timing
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}_{hash(request.question) % 10000}"
    
    # Log detailed input
    input_data = {
        "request_id": request_id,
        "question": request.question,
        "question_length": len(request.question),
        "user_id": request.user_id,
        "context": request.context,
        "timestamp": start_time
    }
    input_logger.info(f"INPUT: {json.dumps(input_data, indent=2)}")
    
    if constellation is None:
        error_msg = "GAAPF system not initialized"
        performance_logger.error(f"REQUEST_ERROR: {request_id} - {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )
    
    try:
        logger.info(f"[{request_id}] Processing question: {request.question[:100]}...")
        performance_logger.info(f"REQUEST_START: {request_id} - Question length: {len(request.question)}")
        
        # Prepare interaction data
        interaction_data = {
            "query": request.question,
            "user_id": request.user_id,
            "interaction_type": "question",
            "timestamp": start_time,
            "request_id": request_id
        }
        
        # Prepare learning context
        learning_context = {
            "user_profile": {
                "name": request.user_id,
                "learning_style": "adaptive"
            },
            "framework_config": {
                "name": "benchmark_evaluation",
                "type": "assessment"
            },
            "current_module": "evaluation",
            "messages": [],
            "context": request.context or {},
            "request_id": request_id
        }
        
        # Log processing start
        processing_start = time.time()
        performance_logger.info(f"PROCESSING_START: {request_id} - Constellation processing initiated")
        
        # Process through constellation
        response = constellation.process_interaction(
            interaction_data=interaction_data,
            learning_context=learning_context
        )
        
        processing_end = time.time()
        processing_time = processing_end - processing_start
        
        # Extract response content
        response_content = response.get("content", "")
        if not response_content:
            response_content = "I apologize, but I couldn't generate a proper response to your question."
        
        agent_type = response.get("agent_type", "unknown")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Log detailed output
        output_data = {
            "request_id": request_id,
            "response": response_content,
            "response_length": len(response_content),
            "agent_type": agent_type,
            "processing_time_seconds": processing_time,
            "total_time_seconds": total_time,
            "status": "success",
            "timestamp": time.time(),
            "full_response_metadata": response
        }
        output_logger.info(f"OUTPUT: {json.dumps(output_data, indent=2)}")
        
        # Log performance metrics
        performance_logger.info(
            f"REQUEST_COMPLETE: {request_id} - "
            f"Total: {total_time:.3f}s, Processing: {processing_time:.3f}s, "
            f"Agent: {agent_type}, Input: {len(request.question)} chars, "
            f"Output: {len(response_content)} chars"
        )
        
        logger.info(
            f"[{request_id}] Generated response from {agent_type} "
            f"({len(response_content)} chars in {total_time:.3f}s): {response_content[:100]}..."
        )
        
        return AgentResponse(
            response=response_content,
            agent_type=agent_type,
            status="success"
        )
        
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = str(e)
        
        # Log detailed error
        error_data = {
            "request_id": request_id,
            "error": error_msg,
            "error_type": type(e).__name__,
            "processing_time_seconds": error_time,
            "status": "error",
            "timestamp": time.time()
        }
        output_logger.error(f"ERROR_OUTPUT: {json.dumps(error_data, indent=2)}")
        
        performance_logger.error(
            f"REQUEST_ERROR: {request_id} - {error_msg} "
            f"(after {error_time:.3f}s)"
        )
        
        logger.error(f"[{request_id}] Error processing question: {e}")
        
        return AgentResponse(
            response=f"I encountered an error while processing your question: {error_msg}",
            status="error",
            error=error_msg
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start GAAPF API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "start_gaapf_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )