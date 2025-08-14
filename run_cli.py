#!/usr/bin/env python3
"""
Main CLI runner for the GAAPF Learning System.
This script provides an easy way to run the GAAPF CLI from the project root.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the src directory to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """Main entry point for the GAAPF CLI runner."""
    try:
        # Import the simplified CLI and required dependencies
        from src.GAAPF.interfaces.cli.simple_cli import SimpleCLI
        
        # Set asyncio policy for Windows to prevent event loop errors
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Initialize LLM (prefer Google Vertex AI)
        try:
            from langchain_google_vertexai import ChatVertexAI
            # Ensure Google credentials are set
            credentials_path = project_root / "google-credentials.json"
            if credentials_path.exists():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0305686287")
            llm_instance = ChatVertexAI(
                model_name="gemini-2.5-flash",
                temperature=0.3,
                project=project_id,
                location="us-central1"
            )
            print("🤖 Using Google Vertex AI LLM...")
        except Exception as e:
            print(f"⚠️  Vertex AI initialization failed: {e}")
            try:
                from langchain_core.language_models.fake import FakeListLLM
                llm_instance = FakeListLLM(responses=[
                    "Hello! I'm your AI learning assistant. How can I help you today?",
                    "I understand you're learning. Let me help you with that.",
                    "That's a great question! Let me explain...",
                    "I'm here to help you learn step by step."
                ])
                print("🤖 Using mock LLM for testing...")
            except ImportError:
                class MockLLM:
                    def invoke(self, prompt):
                        return "Hello! This is a mock response. Please configure a proper LLM."
                    def __call__(self, prompt):
                        return self.invoke(prompt)
                llm_instance = MockLLM()
                print("🤖 Using basic mock LLM...")

        # Create and run the simplified CLI
        cli = SimpleCLI(llm=llm_instance, is_logging=True)
        asyncio.run(cli.start())
        
    except ImportError as e:
        print("🤖 GAAPF - Guidance AI Agent for Python Framework")
        print("=" * 55)
        print(f"❌ Error importing CLI: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Or use poetry: poetry install")
        print("4. Check that all required packages are installed")
        print("5. See QUICK_START.md for detailed setup instructions")
        print("\n💡 Common issues:")
        print("• Missing Rich library: pip install rich")
        print("• Missing LangChain: pip install langchain-core")
        print("• Path issues: run from the GAAPF-main directory")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n👋 GAAPF CLI interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print("🤖 GAAPF - Guidance AI Agent for Python Framework")
        print("=" * 55)
        print(f"❌ Unexpected error: {e}")
        print("\n🔧 Debug options:")
        print("• Run with debug: python run_cli.py --debug")
        print("• Check logs: look for cli_debug.log")
        print("• Verify setup: python -c 'import rich; print(\"Rich OK\")'")
        sys.exit(1)

if __name__ == "__main__":
    main()