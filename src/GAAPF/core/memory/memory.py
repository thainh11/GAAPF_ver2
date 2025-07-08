from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal, Union
import json
import logging
import time
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
    '''This stores the memory of the conversation.
    '''
    def __init__(self, 
            memory_path: Optional[Union[Path, str]] = Path('templates/memory.jsonl'), 
            is_reset_memory: bool=False,
            is_logging: bool=False,
        *args, **kwargs):
        if isinstance(memory_path, str) and memory_path:
            self.memory_path = Path(memory_path)
        else:
            self.memory_path = memory_path
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_reset_memory = is_reset_memory
        self.is_logging = is_logging
        if not self.memory_path.exists():
            self.memory_path.write_text(json.dumps({}, indent=4), encoding="utf-8")
        if self.is_reset_memory:
            self.memory_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

    def load_memory(self, load_type: Literal['list', 'string'] = 'list', user_id: str = None):
        data = []
        with open(self.memory_path, "r", encoding="utf-8") as f:
                # for line in f:
                #     try:
                #         route = json.loads(line)
                #         if route:
                #             if user_id:
                #                 if route['user_id'] == user_id:
                #                     data.append(route)
                #             else:
                #                 data.append(route)
                #     except json.JSONDecodeError as e:
                #         logger.warning(
                #             f"Skipping invalid JSON line: {line.strip()} - Error: {e}"
                #         )

                data = json.load(f)
                if not user_id: # Load all memory
                    data_user = data
                else: 
                    if user_id in data: # Load memory by user_id
                        data_user = data[user_id]
                    else:
                        data_user = []

        if load_type == 'list':
            return data_user
        elif load_type == 'string':
            message = self.revert_object_mess(data_user)
            return message

    def save_memory(self, obj: list, memory_path: Path, user_id: str):
        memory = self.load_memory(load_type='list')
        memory[user_id] = obj
        with open(memory_path, "w", encoding="utf-8") as f:
            # for item in obj:
            #     f.write(json.dumps(item) + '\n')
            json.dump(memory, f, indent=4, ensure_ascii=False)

        if self.is_logging:
            logger.info(f"Saved memory!")

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
            head, _, relation, relation_properties, tail, _ = list(line.values())
            relation_additional= f"[{relation_properties}]" if relation_properties else ""
            mess.append(f"{head} -> {relation}{relation_additional} -> {tail}")
        mess = "\n".join(mess)
        return mess

    def update_memory(self, graph: list, user_id: str):
        memory_about_user = self.load_memory(load_type='list', user_id=user_id)
        if memory_about_user:
            index_memory = [(item['head'], item['relation'], item['tail']) for item in memory_about_user]
            index_memory_head_relation_tail_type = [(item['head'], item['relation'],  item['tail_type']) for item in memory_about_user]
        else:
            index_memory = []
            index_memory_head_relation_tail_type = []
            
        if graph:
            for line in graph:
                head, head_type, relation, relation_properties, tail, tail_type= list(line.values())
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
                if self.is_logging:
                    logger.info(f"Cleared memory for user: {user_id}")
            else:
                if self.is_logging:
                    logger.warning(f"No memory found for user: {user_id}")
        else:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=4, ensure_ascii=False)
            if self.is_logging:
                logger.info("Cleared all memory.")
