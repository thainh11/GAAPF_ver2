#!/usr/bin/env python3
"""
DEPRECATED: Legacy CLI entry point.
Please use run_cli.py in the project root instead.
"""

import warnings
import sys
from pathlib import Path

# Redirect to the new CLI runner
if __name__ == "__main__":
    print("🔄 Redirecting to simplified CLI...")
    print("Please use: python run_cli.py")
    
    # Try to run the new CLI
    try:
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from run_cli import main
        main()
    except ImportError:
        print("❌ Cannot find run_cli.py. Please run from project root:")
        print("python run_cli.py")
        sys.exit(1)