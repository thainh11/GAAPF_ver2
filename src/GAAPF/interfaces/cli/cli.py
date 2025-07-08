#!/usr/bin/env python3
"""
Entry point script for the GAAPF CLI interface.
This script provides an easy way to run the learning system.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ...core.interfaces.cli import GAAPFCLI

def main():
    """Main entry point for the CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GAAPF Learning System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cli.py                    # Start with default settings
  python run_cli.py --verbose         # Enable verbose logging
  python run_cli.py --debug           # Enable debug mode
  python run_cli.py --memory custom   # Use custom memory directory
        """
    )
    
    parser.add_argument(
        "--user-profiles", 
        type=str, 
        default="user_profiles",
        help="Path to user profiles directory (default: user_profiles)"
    )
    
    parser.add_argument(
        "--frameworks", 
        type=str, 
        default="frameworks",
        help="Path to frameworks directory (default: frameworks)"
    )
    
    parser.add_argument(
        "--memory", 
        type=str, 
        default="memory",
        help="Path to memory directory (default: memory)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with detailed output"
    )
    
    args = parser.parse_args()
    
    # Set debug mode environment variable
    if args.debug:
        os.environ["DEBUG_MODE"] = "true"
    
    # Create CLI instance
    cli = GAAPFCLI(
        user_profiles_path=Path(args.user_profiles),
        frameworks_path=Path(args.frameworks),
        memory_path=Path(args.memory),
        is_logging=args.verbose or args.debug
    )
    
    try:
        # Run the CLI
        asyncio.run(cli.start())
    except KeyboardInterrupt:
        print("\n🔥 Exiting GAAPF Learning System")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting CLI: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 