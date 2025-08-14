"""
Server startup script for Educational RLHF System
"""
import os
import sys
import uvicorn
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import CONFIG

def start_development_server():
    """Start development server with hot reload"""
    print("🚀 Starting Educational RLHF System - Development Mode")
    print(f"   Framework: {CONFIG.framework_type}")
    print(f"   Target Level: {CONFIG.target_level}")
    print(f"   Server: http://{CONFIG.host}:{CONFIG.port}")
    print("   Hot reload: Enabled")
    
    uvicorn.run(
        "main:app",
        host=CONFIG.host,
        port=CONFIG.port,
        reload=True,
        log_level=CONFIG.log_level.lower(),
        access_log=True
    )

def start_production_server():
    """Start production server"""
    print("🏭 Starting Educational RLHF System - Production Mode")
    print(f"   Framework: {CONFIG.framework_type}")
    print(f"   Target Level: {CONFIG.target_level}")
    print(f"   Server: http://{CONFIG.host}:{CONFIG.port}")
    print("   Hot reload: Disabled")
    
    uvicorn.run(
        "main:app",
        host=CONFIG.host,
        port=CONFIG.port,
        reload=False,
        log_level=CONFIG.log_level.lower(),
        access_log=False,
        workers=1  # Single worker for shared state
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Educational RLHF Server")
    parser.add_argument("--prod", action="store_true", help="Run in production mode")
    args = parser.parse_args()
    
    if args.prod:
        start_production_server()
    else:
        start_development_server()