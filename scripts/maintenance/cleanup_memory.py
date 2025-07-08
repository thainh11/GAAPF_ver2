#!/usr/bin/env python3
"""
Memory Cleanup Utility for GAAPF

This script consolidates agent-specific memory files into centralized user memory files,
reducing memory file bloat and improving storage efficiency.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryCleanup:
    """Utility class for consolidating agent-specific memory files."""
    
    def __init__(self, memory_dir: Path):
        """
        Initialize the memory cleanup utility.
        
        Parameters:
        ----------
        memory_dir : Path
            Directory containing memory files
        """
        self.memory_dir = Path(memory_dir)
        self.backup_dir = self.memory_dir / "backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.agent_prefixes = [
            'instructor_', 'research_assistant_', 'knowledge_synthesizer_',
            'mentor_', 'documentation_expert_', 'code_assistant_',
            'troubleshooter_', 'practice_facilitator_', 'assessment_',
            'project_guide_', 'motivational_coach_', 'progress_tracker_'
        ]
    
    def create_backup(self) -> None:
        """Create backup of all memory files before cleanup."""
        logger.info("Creating backup of memory files...")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        memory_files = list(self.memory_dir.glob("*.json"))
        for memory_file in memory_files:
            if memory_file.is_file():
                shutil.copy2(memory_file, self.backup_dir / memory_file.name)
        
        logger.info(f"Backup created at: {self.backup_dir}")
        logger.info(f"Backed up {len(memory_files)} files")
    
    def identify_agent_files(self) -> Dict[str, List[Path]]:
        """
        Identify agent-specific memory files grouped by user.
        
        Returns:
        -------
        Dict[str, List[Path]]
            Dictionary mapping user_id to list of their agent memory files
        """
        user_agent_files = {}
        
        for memory_file in self.memory_dir.glob("*.json"):
            if memory_file.is_file():
                filename = memory_file.name
                
                # Check if it's an agent-specific file
                for prefix in self.agent_prefixes:
                    if filename.startswith(prefix):
                        # Extract user_id
                        user_part = filename.replace(prefix, '').replace('.json', '')
                        if user_part.startswith('user_'):
                            user_id = user_part
                            if user_id not in user_agent_files:
                                user_agent_files[user_id] = []
                            user_agent_files[user_id].append(memory_file)
                            break
        
        return user_agent_files
    
    def consolidate_user_memories(self, user_id: str, agent_files: List[Path]) -> Dict:
        """
        Consolidate memories from multiple agent files for a user.
        
        Parameters:
        ----------
        user_id : str
            User identifier
        agent_files : List[Path]
            List of agent-specific memory files for this user
            
        Returns:
        -------
        Dict
            Consolidated memory data
        """
        consolidated_memory = {}
        
        for agent_file in agent_files:
            try:
                # Extract agent type from filename
                agent_type = "unknown"
                for prefix in self.agent_prefixes:
                    if agent_file.name.startswith(prefix):
                        agent_type = prefix.rstrip('_')
                        break
                
                # Load memory content
                if agent_file.stat().st_size > 2:  # Not just {}
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    
                    if content and user_id in content:
                        user_memories = content[user_id]
                        if user_memories:
                            # Add agent context to each memory entry
                            for memory_entry in user_memories:
                                if isinstance(memory_entry, dict):
                                    memory_entry['original_agent_type'] = agent_type
                                    if 'consolidation_timestamp' not in memory_entry:
                                        memory_entry['consolidation_timestamp'] = datetime.now().isoformat()
                            
                            # Consolidate memories
                            if user_id not in consolidated_memory:
                                consolidated_memory[user_id] = []
                            consolidated_memory[user_id].extend(user_memories)
                
                logger.info(f"Processed {agent_file.name} for user {user_id}")
                
            except Exception as e:
                logger.error(f"Error processing {agent_file.name}: {e}")
        
        return consolidated_memory
    
    def save_consolidated_memory(self, user_id: str, consolidated_memory: Dict) -> Path:
        """
        Save consolidated memory to a new user-specific file.
        
        Parameters:
        ----------
        user_id : str
            User identifier
        consolidated_memory : Dict
            Consolidated memory data
            
        Returns:
        -------
        Path
            Path to the new consolidated memory file
        """
        consolidated_file = self.memory_dir / f"user_{user_id}_memory.json"
        
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_memory, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created consolidated memory file: {consolidated_file}")
        return consolidated_file
    
    def cleanup_old_files(self, agent_files: List[Path], dry_run: bool = False) -> None:
        """
        Remove old agent-specific memory files.
        
        Parameters:
        ----------
        agent_files : List[Path]
            List of agent files to remove
        dry_run : bool
            If True, only log what would be deleted without actually deleting
        """
        for agent_file in agent_files:
            if dry_run:
                logger.info(f"Would delete: {agent_file}")
            else:
                try:
                    agent_file.unlink()
                    logger.info(f"Deleted: {agent_file}")
                except Exception as e:
                    logger.error(f"Error deleting {agent_file}: {e}")
    
    def get_cleanup_statistics(self) -> Dict:
        """
        Get statistics about the cleanup operation.
        
        Returns:
        -------
        Dict
            Statistics about files before and after cleanup
        """
        user_agent_files = self.identify_agent_files()
        
        total_agent_files = sum(len(files) for files in user_agent_files.values())
        total_users = len(user_agent_files)
        
        # Check existing consolidated files
        existing_consolidated = len(list(self.memory_dir.glob("user_*_memory.json")))
        
        return {
            "total_agent_files": total_agent_files,
            "total_users": total_users,
            "existing_consolidated_files": existing_consolidated,
            "potential_file_reduction": total_agent_files - total_users,
            "reduction_percentage": (total_agent_files - total_users) / total_agent_files * 100 if total_agent_files > 0 else 0
        }
    
    def run_cleanup(self, dry_run: bool = False, create_backup: bool = True) -> Dict:
        """
        Run the complete memory cleanup process.
        
        Parameters:
        ----------
        dry_run : bool
            If True, only simulate the cleanup without making changes
        create_backup : bool
            If True, create backup before cleanup
            
        Returns:
        -------
        Dict
            Results of the cleanup operation
        """
        logger.info("Starting memory cleanup process...")
        
        # Get initial statistics
        initial_stats = self.get_cleanup_statistics()
        logger.info(f"Found {initial_stats['total_agent_files']} agent-specific files for {initial_stats['total_users']} users")
        logger.info(f"Potential reduction: {initial_stats['potential_file_reduction']} files ({initial_stats['reduction_percentage']:.1f}%)")
        
        if dry_run:
            logger.info("DRY RUN MODE - No files will be modified")
        
        # Create backup if requested
        if create_backup and not dry_run:
            self.create_backup()
        
        # Identify agent files by user
        user_agent_files = self.identify_agent_files()
        
        consolidation_results = {
            "users_processed": 0,
            "files_consolidated": 0,
            "files_deleted": 0,
            "errors": 0
        }
        
        # Process each user
        for user_id, agent_files in user_agent_files.items():
            try:
                logger.info(f"Processing user: {user_id} ({len(agent_files)} files)")
                
                # Consolidate memories
                consolidated_memory = self.consolidate_user_memories(user_id, agent_files)
                
                if consolidated_memory and not dry_run:
                    # Save consolidated memory
                    self.save_consolidated_memory(user_id, consolidated_memory)
                    consolidation_results["files_consolidated"] += 1
                
                # Cleanup old files
                if not dry_run:
                    self.cleanup_old_files(agent_files, dry_run=False)
                else:
                    self.cleanup_old_files(agent_files, dry_run=True)
                
                consolidation_results["files_deleted"] += len(agent_files)
                consolidation_results["users_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing user {user_id}: {e}")
                consolidation_results["errors"] += 1
        
        # Final statistics
        final_stats = self.get_cleanup_statistics()
        
        logger.info("Memory cleanup completed!")
        logger.info(f"Users processed: {consolidation_results['users_processed']}")
        logger.info(f"Files consolidated: {consolidation_results['files_consolidated']}")
        logger.info(f"Files deleted: {consolidation_results['files_deleted']}")
        
        if consolidation_results["errors"] > 0:
            logger.warning(f"Errors encountered: {consolidation_results['errors']}")
        
        return {
            "initial_stats": initial_stats,
            "final_stats": final_stats,
            "consolidation_results": consolidation_results,
            "backup_location": str(self.backup_dir) if create_backup and not dry_run else None
        }


def main():
    """Main entry point for the cleanup script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cleanup and consolidate GAAPF memory files")
    parser.add_argument("--memory-dir", default="memory", help="Memory directory path (default: memory)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate cleanup without making changes")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--stats-only", action="store_true", help="Only show statistics, don't perform cleanup")
    
    args = parser.parse_args()
    
    memory_dir = Path(args.memory_dir)
    if not memory_dir.exists():
        logger.error(f"Memory directory not found: {memory_dir}")
        return 1
    
    cleanup = MemoryCleanup(memory_dir)
    
    if args.stats_only:
        stats = cleanup.get_cleanup_statistics()
        print("\n📊 Memory File Statistics:")
        print(f"  Agent-specific files: {stats['total_agent_files']}")
        print(f"  Users: {stats['total_users']}")
        print(f"  Existing consolidated files: {stats['existing_consolidated_files']}")
        print(f"  Potential file reduction: {stats['potential_file_reduction']} files")
        print(f"  Reduction percentage: {stats['reduction_percentage']:.1f}%")
        return 0
    
    # Run cleanup
    results = cleanup.run_cleanup(
        dry_run=args.dry_run,
        create_backup=not args.no_backup
    )
    
    print("\n🎯 Cleanup Results:")
    print(f"  Users processed: {results['consolidation_results']['users_processed']}")
    print(f"  Files consolidated: {results['consolidation_results']['files_consolidated']}")
    print(f"  Files deleted: {results['consolidation_results']['files_deleted']}")
    
    if results["backup_location"]:
        print(f"  Backup location: {results['backup_location']}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 