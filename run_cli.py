#!/usr/bin/env python3
"""
Main CLI runner for the GAAPF Learning System.
This script provides an easy way to run the GAAPF CLI from the project root.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add the src directory to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def get_available_frameworks():
    """Get list of available frameworks from allowed_domains and vectordb."""
    frameworks = ["langchain", "langgraph", "crewai", "autogen"]  # fallback
    
    try:
        from GAAPF.core.tools.framework_collector import FrameworkCollector
        fc = FrameworkCollector(is_logging=False)
        allowed = list((fc.allowed_domains or {}).keys())
    except Exception:
        allowed = []
    
    try:
        vectordb_root = Path("data/frameworks/vectordb")
        existing = [p.name for p in vectordb_root.iterdir() if p.is_dir()]
    except Exception:
        existing = []
    
    # Merge and sort unique list
    if allowed or existing:
        frameworks = sorted(set(allowed) | set(existing))
    
    return frameworks

def create_parser():
    """Create argument parser with enhanced help."""
    
    # Combined formatter for better readability
    class CombinedFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass
    
    # Get available frameworks for help text
    available_fw = get_available_frameworks()
    fw_list = ", ".join(available_fw)
    
    description = """
🤖 GAAPF - Guidance AI Agent for Python Framework Learning

A powerful CLI tool for learning Python frameworks through AI-guided instruction.
Supports both Study Mode (Socratic questioning) and Direct Mode (immediate answers).
    """.strip()
    
    epilog = f"""
Examples:
  python run_cli.py                           # Interactive mode with prompts
  python run_cli.py --framework langchain     # Start with LangChain framework
  python run_cli.py --study --user-id alice   # Enable study mode for user 'alice'
  python run_cli.py --no-study --debug        # Direct mode with debug logging
  python run_cli.py --list-frameworks         # Show available frameworks

Available Frameworks: {fw_list}

Study vs Direct Mode:
  • Study Mode (--study):    Learn through guided questions (Socratic method)
  • Direct Mode (--no-study): Get immediate answers and explanations

Note: Uses Google Vertex AI as LLM provider. Configure via environment variables.
For setup instructions, see project documentation.
    """.strip()
    
    parser = argparse.ArgumentParser(
        prog="run_cli.py",
        description=description,
        epilog=epilog,
        formatter_class=CombinedFormatter,
        add_help=True
    )
    
    # Framework selection
    parser.add_argument(
        "--framework", "-f",
        choices=available_fw,
        help="Framework to learn (default: interactive selection)"
    )
    
    # Study mode options
    study_group = parser.add_mutually_exclusive_group()
    study_group.add_argument(
        "--study", "-s",
        action="store_true",
        help="Enable Study Mode (Socratic questioning)"
    )
    study_group.add_argument(
        "--no-study", "-d",
        action="store_true", 
        help="Enable Direct Mode (immediate answers)"
    )
    
    # User identification
    parser.add_argument(
        "--user-id", "-u",
        default="default",
        help="User ID for session tracking"
    )
    
    # Debug and logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    # List frameworks
    parser.add_argument(
        "--list-frameworks",
        action="store_true",
        help="List available frameworks and exit"
    )
    
    return parser

def main():
    """Main entry point for the GAAPF CLI runner."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle --list-frameworks
    if args.list_frameworks:
        frameworks = get_available_frameworks()
        print("🤖 GAAPF - Available Frameworks:")
        print("=" * 40)
        for i, fw in enumerate(frameworks, 1):
            print(f"  {i}. {fw.title()}")
        print(f"\nTotal: {len(frameworks)} frameworks available")
        return
    
    try:
        # Import the simplified CLI and required dependencies
        from GAAPF.interfaces.cli.simple_cli import SimpleCLI
        
        # Set asyncio policy for Windows to prevent event loop errors
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Initialize LLM (prefer Google Vertex AI)
        try:
            from langchain_google_vertexai import ChatVertexAI
            from GAAPF.core.utils.credentials_helper import get_vertex_ai_config
            
            # Get standardized Vertex AI configuration
            vertex_config = get_vertex_ai_config()
            project_id = vertex_config["project"]
            model_name = vertex_config["model_name"]
            temperature = vertex_config["temperature"]
            location = vertex_config["location"]
            top_p = vertex_config["top_p"]
            
            llm_instance = ChatVertexAI(
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                project=project_id,
                location=location
            )            
            print("🤖 Using Google Vertex AI LLM...")
        except Exception as e:
            if args.debug:
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

        # Create CLI with enhanced configuration from args
        cli = SimpleCLI(llm=llm_instance, is_logging=args.debug)
        
        # Pre-configure CLI based on command line arguments
        if args.framework:
            cli.current_framework = args.framework
            cli._preconfigured_framework = True
            print(f"🎯 Framework pre-selected: {args.framework.title()}")
        
        if args.user_id and args.user_id != "default":
            cli.user_id = args.user_id
            cli._preconfigured_user_id = True
            print(f"👤 User ID set: {args.user_id}")
        
        # Force Study Mode only
        cli.study_mode = True
        cli._preconfigured_study = True
        print("🤔 Study Mode enabled")
        
        # Start the CLI (it will skip interactive prompts for pre-configured options)
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
        if args.debug:
            import traceback
            traceback.print_exc()
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
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()