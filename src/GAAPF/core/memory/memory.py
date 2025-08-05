from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal, Union, Dict, List
import json
import logging
import time
import threading
from collections import defaultdict, deque
from aucodb.graph import LLMGraphTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryMeta(ABC):
    @abstractmethod
    def update_memory(self, graph: list):
        pass
    
    @abstractmethod
    def save_short_term_memory(self, llm, message):
        pass

    @abstractmethod
    def save_memory(self, message: str, *args, **kwargs):
        pass

class Memory(MemoryMeta):
    '''Enhanced memory management with optimization and leak prevention.
    '''
    def __init__(self, 
            memory_path: Optional[Union[Path, str]] = Path('templates/memory.jsonl'), 
            is_reset_memory: bool=False,
            is_logging: bool=False,
            max_memory_entries: int = 10000,
            max_session_entries: int = 1000,
            cleanup_threshold: float = 0.8,
            cache_size: int = 100,
        *args, **kwargs):
        if isinstance(memory_path, str) and memory_path:
            self.memory_path = Path(memory_path)
        else:
            self.memory_path = memory_path
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_reset_memory = is_reset_memory
        self.is_logging = is_logging
        
        # Memory management settings
        self.max_memory_entries = max_memory_entries
        self.max_session_entries = max_session_entries
        self.cleanup_threshold = cleanup_threshold
        
        # In-memory cache for frequently accessed data
        self._memory_cache = {}
        self._cache_timestamps = {}
        self._cache_size = cache_size
        self._cache_lock = threading.RLock()
        
        # Memory statistics tracking
        self._memory_stats = defaultdict(lambda: {
            'total_entries': 0,
            'last_cleanup': time.time(),
            'access_count': 0
        })
        
        if not self.memory_path.exists():
            self.memory_path.write_text(json.dumps({}, indent=4), encoding="utf-8")
        if self.is_reset_memory:
            self.memory_path.write_text(json.dumps({}, indent=4), encoding="utf-8")
            self._clear_cache()

    def load_memory(self, load_type: Literal['list', 'string'] = 'list', user_id: str = None):
        # Check cache first
        cache_key = f"{user_id}_{load_type}" if user_id else f"all_{load_type}"
        
        with self._cache_lock:
            if cache_key in self._memory_cache:
                # Update access timestamp
                self._cache_timestamps[cache_key] = time.time()
                if self.is_logging:
                    logger.debug(f"Cache hit for {cache_key}")
                return self._memory_cache[cache_key]
        
        # Load from file if not in cache
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                if not user_id:  # Load all memory
                    data_user = data
                else: 
                    if user_id in data:  # Load memory by user_id
                        data_user = data[user_id]
                        # Update access statistics
                        self._memory_stats[user_id]['access_count'] += 1
                    else:
                        data_user = []
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading memory: {e}")
            data_user = [] if user_id else {}
        
        # Process data based on load_type
        if load_type == 'list':
            result = data_user
        elif load_type == 'string':
            result = self.revert_object_mess(data_user) if isinstance(data_user, list) else ""
        else:
            result = data_user
        
        # Cache the result
        self._update_cache(cache_key, result)
        
        return result

    def save_memory(self, obj: list, memory_path: Path, user_id: str):
        # Check if cleanup is needed before saving
        if len(obj) > self.max_memory_entries * self.cleanup_threshold:
            obj = self._cleanup_old_entries(obj, user_id)
        
        memory = self.load_memory(load_type='list')
        memory[user_id] = obj
        
        try:
            with open(memory_path, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=4, ensure_ascii=False)
            
            # Update statistics
            self._memory_stats[user_id]['total_entries'] = len(obj)
            
            # Invalidate cache for this user
            self._invalidate_user_cache(user_id)
            
            if self.is_logging:
                logger.info(f"Saved {len(obj)} memory entries for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            raise

    def save_short_term_memory(self, llm, message, user_id, agent_type=None):
        """
        Save short term memory with optional agent context.
        
        Parameters:
        ----------
        llm : Language model for graph generation
        message : str
            Message to process and save
        user_id : str
            User identifier
        agent_type : str, optional
            Type of agent creating this memory entry
        """
        graph_transformer = LLMGraphTransformer(
            llm = llm
        )
        graph = graph_transformer.generate_graph(message)
        
        # Add agent context and timestamp to each graph entry
        if graph:
            for entry in graph:
                if agent_type:
                    entry['agent_type'] = agent_type
                entry['timestamp'] = time.time()
        
        self.update_memory(graph, user_id)
        return graph

    def revert_object_mess(self, object: list[dict]):
        mess = []
        for line in object:
            head = line.get('head', '')
            relation = line.get('relation', '')
            relation_properties = line.get('relation_properties', '')
            tail = line.get('tail', '')
            
            relation_additional= f"[{relation_properties}]" if relation_properties else ""
            mess.append(f"{head} -> {relation}{relation_additional} -> {tail}")
        mess = "\n".join(mess)
        return mess

    def update_memory(self, graph: list, user_id: str):
        memory_about_user = self.load_memory(load_type='list', user_id=user_id)
        if memory_about_user:
            # Add safety checks for missing keys
            index_memory = [(item.get('head', ''), item.get('relation', ''), item.get('tail', '')) for item in memory_about_user if all(key in item for key in ['head', 'relation', 'tail'])]
            index_memory_head_relation_tail_type = [(item.get('head', ''), item.get('relation', ''), item.get('tail_type', '')) for item in memory_about_user if all(key in item for key in ['head', 'relation', 'tail_type'])]
        else:
            index_memory = []
            index_memory_head_relation_tail_type = []
            
        if graph:
            for line in graph:
                # Skip items that don't have required keys
                if not all(key in line for key in ['head', 'relation']):
                    if self.is_logging:
                        logger.warning(f"Skipping incomplete graph item: {line}")
                    continue
                    
                head = line.get('head')
                head_type = line.get('head_type')
                relation = line.get('relation')
                relation_properties = line.get('relation_properties')
                tail = line.get('tail', '')  # Default to empty string if missing
                tail_type = line.get('tail_type', '')
                
                lookup_hrt = (head, relation, tail)
                lookup_hrttp = (head, relation, tail_type)
                if lookup_hrt in index_memory:
                    if self.is_logging:
                        logger.info(f"Bypass {line}")
                    pass
                elif lookup_hrttp in index_memory_head_relation_tail_type:
                    index_match = index_memory_head_relation_tail_type.index(lookup_hrttp)
                    if self.is_logging:
                        logger.info(f"Update new line: {line}\nfrom old line {memory_about_user[index_match]}")
                    memory_about_user[index_match] = line
                else:
                    if self.is_logging:
                        logger.info(f"Insert new line: {line}")
                    memory_about_user.append(line)
        else:
            if self.is_logging:
                logger.info(f"No thing updated")
        
        self.save_memory(obj=memory_about_user, memory_path=self.memory_path, user_id=user_id)
        return memory_about_user
    
    def save_session_conversation(self, session_id: str, messages: list, user_id: str):
        """
        Save conversation history for a specific session.
        
        Parameters:
        ----------
        session_id : str
            Identifier for the session
        messages : list
            List of conversation messages
        user_id : str
            Identifier for the user
        """
        try:
            # Create session-specific conversation file
            conversation_file = self.memory_path.parent / f"session_conversations_{user_id}_{session_id}.json"
            
            conversation_data = {
                "session_id": session_id,
                "user_id": user_id,
                "messages": messages,
                "last_updated": json.dumps(None) 
            }
            
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=4, ensure_ascii=False)
            
            if self.is_logging:
                logger.info(f"Saved session conversation for {session_id}")
            
        except Exception as e:
            logger.error(f"Error saving session conversation: {e}")
    
    def load_session_conversation(self, session_id: str, user_id: str) -> list:
        """
        Load conversation history for a specific session.
        
        Parameters:
        ----------
        session_id : str
            Identifier for the session
        user_id : str
            Identifier for the user
            
        Returns:
        -------
        list
            List of conversation messages
        """
        try:
            conversation_file = self.memory_path.parent / f"session_conversations_{user_id}_{session_id}.json"
            
            if not conversation_file.exists():
                return []
            
            with open(conversation_file, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            messages = conversation_data.get("messages", [])
            
            if self.is_logging:
                logger.info(f"Loaded {len(messages)} messages for session {session_id}")
            
            return messages
            
        except Exception as e:
            logger.error(f"Error loading session conversation: {e}")
            return []

    def get_agent_memories(self, user_id: str, agent_type: str = None) -> list:
        """
        Get memories for specific agent or all agents.
        
        Parameters:
        ----------
        user_id : str
            User identifier
        agent_type : str, optional
            Specific agent type to filter by
            
        Returns:
        -------
        list
            List of memory entries (filtered by agent_type if specified)
        """
        memories = self.load_memory(load_type='list', user_id=user_id)
        
        if agent_type and memories:
            return [m for m in memories if m.get('agent_type') == agent_type]
        return memories
    
    def get_messages(self, user_id: str, agent_type: str = None) -> list:
        """
        Get conversation history as LangChain message objects.
        
        Parameters:
        ----------
        user_id : str
            User identifier
        agent_type : str, optional
            Specific agent type to filter by
            
        Returns:
        -------
        list
            List of HumanMessage and AIMessage objects
        """
        from langchain_core.messages import HumanMessage, AIMessage

        memories = self.load_memory(load_type='list', user_id=user_id)
        if not memories:
            return []

        # Filter by agent type if provided
        if agent_type:
            memories = [m for m in memories if m.get('agent_type') == agent_type]

        messages = []
        for mem in memories:
            role = mem.get("type", "human") # Default to human for older formats
            content = mem.get("content", "")
            
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
        
        return messages

    def get_memory_statistics(self, user_id: str) -> dict:
        """
        Get statistics about the user's memory.
        
        Parameters:
        ----------
        user_id : str
            User identifier
            
        Returns:
        -------
        dict
            Statistics including total memories and breakdown by agent
        """
        memories = self.load_memory(load_type='list', user_id=user_id)
        
        if not memories:
            return {"total_memories": 0, "agent_breakdown": {}}
        
        agent_counts = {}
        total_memories = len(memories)
        
        for memory in memories:
            agent_type = memory.get('agent_type', 'unknown')
            agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
        
        # Calculate statistics
        stats = {
            "total_entries": total_memories,
            "entries_by_agent": agent_counts,
            "entries_by_head_type": {},
            "entries_by_relation": {},
            "latest_memory_timestamp": max(memory.get('timestamp', 0) for memory in memories) if memories else None
        }
        
        return stats

    def clear_memory(self, user_id: str = None):
        """
        Clears all memory or memory for a specific user.
        
        Parameters:
        ----------
        user_id : str, optional
            If provided, only clears memory for this user. Otherwise, clears all memory.
        """
        if user_id:
            memory = self.load_memory()
            if user_id in memory:
                del memory[user_id]
                with open(self.memory_path, "w", encoding="utf-8") as f:
                    json.dump(memory, f, indent=4, ensure_ascii=False)
                
                # Clear user-specific cache and stats
                self._invalidate_user_cache(user_id)
                if user_id in self._memory_stats:
                    del self._memory_stats[user_id]
                
                if self.is_logging:
                    logger.info(f"Cleared memory for user: {user_id}")
            else:
                if self.is_logging:
                    logger.warning(f"No memory found for user: {user_id}")
        else:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=4, ensure_ascii=False)
            
            # Clear all cache and stats
            self._clear_cache()
            self._memory_stats.clear()
            
            if self.is_logging:
                logger.info("Cleared all memory.")
    
    def _clear_cache(self):
        """Clear all cached data."""
        with self._cache_lock:
            self._memory_cache.clear()
            self._cache_timestamps.clear()
    
    def _invalidate_user_cache(self, user_id: str):
        """Invalidate cache entries for a specific user."""
        with self._cache_lock:
            keys_to_remove = [key for key in self._memory_cache.keys() if key.startswith(f"{user_id}_")]
            for key in keys_to_remove:
                del self._memory_cache[key]
                del self._cache_timestamps[key]
    
    def _update_cache(self, cache_key: str, data):
        """Update cache with new data, managing cache size."""
        with self._cache_lock:
            # Remove oldest entries if cache is full
            if len(self._memory_cache) >= self._cache_size:
                oldest_key = min(self._cache_timestamps.keys(), key=lambda k: self._cache_timestamps[k])
                del self._memory_cache[oldest_key]
                del self._cache_timestamps[oldest_key]
            
            self._memory_cache[cache_key] = data
            self._cache_timestamps[cache_key] = time.time()
    
    def _cleanup_old_entries(self, entries: List[Dict], user_id: str) -> List[Dict]:
        """Clean up old memory entries to prevent memory bloat."""
        if not entries:
            return entries
        
        # Sort by timestamp (newest first)
        sorted_entries = sorted(entries, key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Keep only the most recent entries within limit
        cleaned_entries = sorted_entries[:self.max_memory_entries]
        
        # Update cleanup timestamp
        self._memory_stats[user_id]['last_cleanup'] = time.time()
        
        if self.is_logging and len(cleaned_entries) < len(entries):
            logger.info(f"Cleaned up {len(entries) - len(cleaned_entries)} old memory entries for user {user_id}")
        
        return cleaned_entries
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        with self._cache_lock:
            return {
                'cache_size': len(self._memory_cache),
                'max_cache_size': self._cache_size,
                'cache_keys': list(self._memory_cache.keys()),
                'memory_stats': dict(self._memory_stats)
            }
    
    def optimize_memory(self, user_id: str = None):
        """Perform memory optimization and cleanup."""
        if user_id:
            # Optimize specific user's memory
            memories = self.load_memory(load_type='list', user_id=user_id)
            if memories and len(memories) > self.max_memory_entries * self.cleanup_threshold:
                cleaned_memories = self._cleanup_old_entries(memories, user_id)
                self.save_memory(cleaned_memories, self.memory_path, user_id)
        else:
            # Optimize all users' memory
            all_memory = self.load_memory(load_type='list')
            if isinstance(all_memory, dict):
                for uid, memories in all_memory.items():
                    if isinstance(memories, list) and len(memories) > self.max_memory_entries * self.cleanup_threshold:
                        cleaned_memories = self._cleanup_old_entries(memories, uid)
                        self.save_memory(cleaned_memories, self.memory_path, uid)
        
        # Clean up cache
        current_time = time.time()
        with self._cache_lock:
            # Remove cache entries older than 1 hour
            old_keys = [key for key, timestamp in self._cache_timestamps.items() 
                       if current_time - timestamp > 3600]
            for key in old_keys:
                del self._memory_cache[key]
                del self._cache_timestamps[key]
        
        if self.is_logging:
            logger.info(f"Memory optimization completed for {'user ' + user_id if user_id else 'all users'}")
