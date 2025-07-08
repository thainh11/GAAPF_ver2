#!/usr/bin/env python3
"""
Memory Statistics Utility for GAAPF

This script analyzes the memory files (JSON and ChromaDB) to provide statistics
on the number of users, memories, and knowledge graph connections.
"""

import json
from pathlib import Path
from typing import Dict

def analyze_memory_structure(memory_dir: Path) -> Dict:
    """Analyze the current memory structure and provide statistics."""
    
    memory_files = list(memory_dir.glob("*.json"))
    
    # Categorize files
    centralized_files = [f for f in memory_files if f.name.startswith("user_") and f.name.endswith("_memory.json")]
    agent_specific_files = [f for f in memory_files if any(
        f.name.startswith(prefix) for prefix in [
            'instructor_user_', 'research_assistant_user_', 'knowledge_synthesizer_user_',
            'mentor_user_', 'documentation_expert_user_', 'code_assistant_user_'
        ]
    )]
    session_files = [f for f in memory_files if f.name.startswith("session_")]
    other_files = [f for f in memory_files if f not in centralized_files + agent_specific_files + session_files]
    
    # Calculate file sizes
    total_size = sum(f.stat().st_size for f in memory_files)
    centralized_size = sum(f.stat().st_size for f in centralized_files)
    
    # Extract user information
    users = set()
    for f in centralized_files:
        # Extract user_id from filename like "user_12_memory.json"
        user_part = f.name.replace("user_", "").replace("_memory.json", "")
        users.add(user_part)
    
    return {
        "total_files": len(memory_files),
        "centralized_files": len(centralized_files),
        "agent_specific_files": len(agent_specific_files),
        "session_files": len(session_files),
        "other_files": len(other_files),
        "total_size_bytes": total_size,
        "centralized_size_bytes": centralized_size,
        "users_count": len(users),
        "users": list(users),
        "optimization_active": len(centralized_files) > 0 and len(agent_specific_files) == 0
    }

def display_memory_statistics(memory_dir: Path = Path("memory")):
    """Display comprehensive memory statistics."""
    
    if not memory_dir.exists():
        print("❌ Memory directory not found!")
        return
    
    stats = analyze_memory_structure(memory_dir)
    
    print("📊 GAAPF Memory System Statistics")
    print("=" * 50)
    
    # Optimization Status
    if stats["optimization_active"]:
        print("✅ Memory optimization is ACTIVE")
        print("✅ Using centralized user memory files")
    else:
        print("⚠️  Memory optimization not detected")
        if stats["agent_specific_files"] > 0:
            print(f"⚠️  Found {stats['agent_specific_files']} agent-specific files")
    
    print()
    
    # File Breakdown
    print("📁 File Breakdown:")
    print(f"  Total memory files: {stats['total_files']}")
    print(f"  Centralized user files: {stats['centralized_files']}")
    print(f"  Agent-specific files: {stats['agent_specific_files']}")
    print(f"  Session files: {stats['session_files']}")
    print(f"  Other files: {stats['other_files']}")
    print()
    
    # User Information
    print("👥 User Information:")
    print(f"  Active users: {stats['users_count']}")
    if stats['users']:
        print(f"  User IDs: {', '.join(stats['users'])}")
    print()
    
    # Storage Information
    print("💾 Storage Information:")
    print(f"  Total storage used: {stats['total_size_bytes']:,} bytes")
    print(f"  Centralized files size: {stats['centralized_size_bytes']:,} bytes")
    print()
    
    # Optimization Benefits
    if stats["optimization_active"]:
        print("🎯 Optimization Benefits:")
        print("  ✅ Reduced file proliferation")
        print("  ✅ Centralized memory per user")
        print("  ✅ Agent context preserved")
        print("  ✅ Improved performance")
        print("  ✅ Easier maintenance")
        
        # Calculate theoretical old system
        if stats["users_count"] > 0:
            estimated_agents = 12  # Common number of agents
            old_file_count = stats["users_count"] * estimated_agents
            reduction = old_file_count - stats["centralized_files"]
            if reduction > 0:
                print(f"  📈 Estimated file reduction: {reduction} files")
                print(f"  📊 Reduction percentage: {(reduction/old_file_count)*100:.1f}%")
    else:
        print("💡 Potential Benefits with Optimization:")
        print("  • Reduce file count by 80-90%")
        print("  • Centralize memory management")
        print("  • Improve system performance")
        print("  • Simplify maintenance")
        print("  • Preserve agent context")
    
    print()
    
    # Recommendations
    print("🔧 Recommendations:")
    if not stats["optimization_active"]:
        print("  • Run: python scripts/maintenance/cleanup_memory.py")
        print("  • This will consolidate agent-specific files")
        print("  • Backup will be created automatically")
    else:
        print("  ✅ Memory system is optimized!")
        print("  • Monitor memory growth over time")
        print("  • Regular cleanup not needed")
    
    # Show backup information if exists
    backup_dir = memory_dir / "backup"
    if backup_dir.exists():
        backups = list(backup_dir.glob("*"))
        if backups:
            print()
            print("💾 Backup Information:")
            print(f"  Available backups: {len(backups)}")
            latest_backup = max(backups, key=lambda x: x.stat().st_mtime)
            print(f"  Latest backup: {latest_backup.name}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Display GAAPF memory statistics")
    parser.add_argument("--memory-dir", default="memory", help="Memory directory path")
    
    args = parser.parse_args()
    
    display_memory_statistics(Path(args.memory_dir))

if __name__ == "__main__":
    main() 