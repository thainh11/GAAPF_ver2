#!/usr/bin/env python3
"""
Simple launcher for GAAPF Learning System.

This script provides a simple command-line interface (CLI) to launch 
the GAAPF agent. It's a convenient way to start the system
with default settings.
"""

import asyncio
from pathlib import Path
import sys

def main():
    """Launch the GAAPF CLI with default settings."""
    # Add the project root to the Python path
    # This is important for resolving modules correctly
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    from GAAPF.interfaces.cli.cli import main as cli_main
    print("🚀 Starting GAAPF Learning System...")
    
    # Run the CLI main function
    asyncio.run(cli_main())

if __name__ == "__main__":
    main() 