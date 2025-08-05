#!/usr/bin/env python3
"""
Simple entry point for GAAPF - Simplified Learning System
Replaces the complex run_cli.py with streamlined startup.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_environment():
    """Setup environment variables and API keys."""
    # Load environment variables
    load_dotenv()
    
    # Check for required API keys
    required_keys = {
        "GOOGLE_API_KEY": "Google AI/VertexAI",
        "TOGETHER_API_KEY": "Together AI"
    }
    
    missing_keys = []
    for key, service in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{key} (for {service})")
    
    if missing_keys:
        print("⚠️  Missing API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease set these in your .env file or environment variables.")
        return False
    
    return True

def create_llm():
    """Create and return an LLM instance."""
    try:
        # Try Google AI first
        if os.getenv("GOOGLE_API_KEY"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        
        # Fallback to Together AI
        elif os.getenv("TOGETHER_API_KEY"):
            from langchain_together import ChatTogether
            return ChatTogether(
                model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                temperature=0.7,
                together_api_key=os.getenv("TOGETHER_API_KEY")
            )
        
        else:
            raise ValueError("No API keys available")
            
    except Exception as e:
        print(f"❌ Error creating LLM: {e}")
        return None

async def main():
    """Main entry point for simplified GAAPF."""
    print("🤖 Starting GAAPF - Simplified Learning System")
    
    # Setup environment
    if not setup_environment():
        return
    
    # Create LLM
    llm = create_llm()
    if not llm:
        print("❌ Failed to initialize LLM. Please check your API keys.")
        return
    
    print("✅ LLM initialized successfully")
    
    # Import and start simplified CLI
    try:
        from src.GAAPF.interfaces.cli.simple_cli import start_simple_cli
        
        print("🚀 Starting simplified CLI...")
        await start_simple_cli(llm, is_logging=False)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting CLI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())