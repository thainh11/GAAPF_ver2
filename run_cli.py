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
        # Import the CLI main function from the correct package path
        from src.GAAPF.core.interfaces.cli.cli import main as cli_main
        
        # Set asyncio policy for Windows to prevent event loop errors
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Run the modern CLI
        asyncio.run(cli_main())
        
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