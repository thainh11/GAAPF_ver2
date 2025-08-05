"""
Centralized Orchestration Logic for GAAPF.

This module provides functions for using an LLM to orchestrate and coordinate
the behavior of specialized AI agents within the GAAPF architecture.
"""
import json
import logging
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Agent selection cache and performance tracking
_agent_selection_cache = {}
_agent_performance_stats = defaultdict(lambda: {'success_count': 0, 'total_count': 0, 'avg_response_time': 0.0})
_cache_max_size = 100
_cache_ttl = 300  # 5 minutes


def llm_orchestrate_agent(
    llm: BaseLanguageModel,
    interaction_data: Dict,
    learning_context: Dict,
    available_agents: List[str],
    is_logging: bool = False,
    use_cache: bool = True,
    fallback_agent: str = "instructor",
) -> Optional[Dict]:
    """
    Use the LLM to determine the primary agent and task for a user query.

    Args:
        llm: The language model to use for orchestration.
        interaction_data: The current user interaction data.
        learning_context: The current learning session context.
        available_agents: A list of available agent types.
        is_logging: Flag to enable detailed logging.
        use_cache: Whether to use caching for agent selection.
        fallback_agent: Default agent to use if orchestration fails.

    Returns:
        A dictionary with "primary_agent" and "task", or None if orchestration fails.
    """
    start_time = time.time()
    
    # Generate cache key
    query = interaction_data.get('query', '')
    cache_key = _generate_cache_key(query, learning_context, available_agents)
    
    # Check cache first
    if use_cache and cache_key in _agent_selection_cache:
        cached_result, timestamp = _agent_selection_cache[cache_key]
        if time.time() - timestamp < _cache_ttl:
            if is_logging:
                logger.info(f"Cache hit for agent selection: {cached_result}")
            return cached_result
        else:
            # Remove expired cache entry
            del _agent_selection_cache[cache_key]
    try:
        # Create a concise, serializable version of the learning context
        context_summary = {
            k: v
            for k, v in learning_context.items()
            if k not in ["messages", "framework_config"]
        }
        
        # Intelligent agent pre-selection based on query patterns
        suggested_agent = _analyze_query_patterns(query, learning_context)
        if suggested_agent and suggested_agent in available_agents:
            agent_hint = f"\n\n**Suggested Agent Based on Query Analysis:** {suggested_agent}"

        # Build enhanced prompt with agent performance data
        agent_performance_info = _get_agent_performance_summary(available_agents)
        
        prompt = f"""You are the orchestrator for a multi-agent AI learning system.
Your role is to analyze the user's query and the current learning context to select the best agent to respond and define its task.

**Available Agents:**
- **instructor**: Explains concepts, provides theoretical background, and guides structured learning. Best for "what is", "explain", "how does", "teach me" questions.
- **practice_facilitator**: Creates hands-on exercises, coding challenges, and practical activities. Best for "practice", "exercise", "show me how to code", "give me an example" requests.
- **code_assistant**: Helps with implementing code, debugging, and providing code snippets.
- **documentation_expert**: Provides information from official documentation and API references.
- **mentor**: Offers high-level guidance, motivation, and learning strategies.
- **assessment**: Evaluates user knowledge with quizzes and tests.
- **project_guide**: Guides users through building larger projects.
- **troubleshooter**: Helps diagnose and fix errors.

**Agent Performance Summary:**
{agent_performance_info}

**User Query:**
"{query}"

**Learning Context:**
{json.dumps(context_summary, indent=2, default=str)}{agent_hint if 'agent_hint' in locals() else ''}

**Your Task:**
Based on all the information above, decide which single agent is best suited to handle this request. Consider both the query content and agent performance history. Respond with a JSON object containing the `primary_agent` and a concise `task` for that agent.

**JSON Response Format:**
{{
  "primary_agent": "<agent_type>",
  "task": "<A short, clear description of what the agent should do>",
  "confidence": <0.0-1.0 confidence score>
}}
"""
        messages = [
            SystemMessage(
                content="You are an expert multi-agent orchestrator. Your response must be a valid JSON object and nothing else."
            ),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        content = response.content

        # Robust JSON extraction with multiple patterns
        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
        ]
        
        orchestration = None
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    orchestration = json.loads(json_str)
                    break
                except json.JSONDecodeError:
                    continue
        
        if orchestration and "primary_agent" in orchestration and "task" in orchestration:
            # Validate agent selection
            selected_agent = orchestration["primary_agent"]
            if selected_agent not in available_agents:
                if is_logging:
                    logger.warning(f"Selected agent '{selected_agent}' not in available agents. Using fallback.")
                orchestration["primary_agent"] = fallback_agent
                orchestration["task"] = f"Handle user query: {query[:100]}..."
                orchestration["confidence"] = 0.5
            
            # Cache successful result
            if use_cache:
                _update_cache(cache_key, orchestration)
            
            # Update performance stats
            response_time = time.time() - start_time
            _update_agent_performance(selected_agent, True, response_time)
            
            if is_logging:
                logger.info(f"LLM Orchestration successful: {orchestration} (took {response_time:.2f}s)")
            return orchestration
        else:
            raise ValueError("Missing 'primary_agent' or 'task' in JSON response or invalid JSON.")

    except (json.JSONDecodeError, ValueError) as e:
        if is_logging:
            logger.error(f"LLM orchestration parsing failed: {e}\nRaw response: {content if 'content' in locals() else 'No response'}")
        # Update performance stats for failure
        response_time = time.time() - start_time
        _update_agent_performance(fallback_agent, False, response_time)
        
        # Return fallback orchestration
        fallback_result = {
            "primary_agent": fallback_agent,
            "task": f"Handle user query: {query[:100]}...",
            "confidence": 0.3,
            "fallback_reason": "JSON parsing failed"
        }
        if is_logging:
            logger.info(f"Using fallback orchestration: {fallback_result}")
        return fallback_result
        
    except Exception as e:
        if is_logging:
            logger.error(f"An unexpected error occurred during LLM orchestration: {e}")
        # Update performance stats for failure
        response_time = time.time() - start_time
        _update_agent_performance(fallback_agent, False, response_time)
        
        # Return fallback orchestration
        fallback_result = {
            "primary_agent": fallback_agent,
            "task": f"Handle user query: {query[:100]}...",
            "confidence": 0.2,
            "fallback_reason": "Unexpected error"
        }
        if is_logging:
            logger.info(f"Using fallback orchestration due to error: {fallback_result}")
        return fallback_result


def _generate_cache_key(query: str, learning_context: Dict, available_agents: List[str]) -> str:
    """Generate a cache key for agent selection."""
    # Create a simplified hash of the key components
    key_components = [
        query.lower().strip(),
        str(sorted(available_agents)),
        learning_context.get('learning_stage', ''),
        learning_context.get('current_topic', ''),
        learning_context.get('user_level', '')
    ]
    return hash(tuple(key_components))


def _update_cache(cache_key: str, result: Dict) -> None:
    """Update the agent selection cache."""
    global _agent_selection_cache
    
    # Clean up cache if it's getting too large
    if len(_agent_selection_cache) >= _cache_max_size:
        # Remove oldest entries (simple FIFO)
        oldest_key = min(_agent_selection_cache.keys(), 
                        key=lambda k: _agent_selection_cache[k][1])
        del _agent_selection_cache[oldest_key]
    
    _agent_selection_cache[cache_key] = (result, time.time())


def _update_agent_performance(agent: str, success: bool, response_time: float) -> None:
    """Update agent performance statistics."""
    global _agent_performance_stats
    
    stats = _agent_performance_stats[agent]
    stats['total_count'] += 1
    if success:
        stats['success_count'] += 1
    
    # Update average response time
    current_avg = stats['avg_response_time']
    total_count = stats['total_count']
    stats['avg_response_time'] = ((current_avg * (total_count - 1)) + response_time) / total_count


def _get_agent_performance_summary(available_agents: List[str]) -> str:
    """Get a summary of agent performance for orchestration."""
    global _agent_performance_stats
    
    summary_lines = []
    for agent in available_agents:
        stats = _agent_performance_stats[agent]
        if stats['total_count'] > 0:
            success_rate = (stats['success_count'] / stats['total_count']) * 100
            avg_time = stats['avg_response_time']
            summary_lines.append(
                f"- {agent}: {success_rate:.1f}% success rate, {avg_time:.2f}s avg response time"
            )
        else:
            summary_lines.append(f"- {agent}: No performance data available")
    
    return "\n".join(summary_lines) if summary_lines else "No performance data available."


def _analyze_query_patterns(query: str, learning_context: Dict) -> Optional[str]:
    """Analyze query patterns to suggest the best agent."""
    query_lower = query.lower()
    
    # Define pattern-based agent suggestions
    patterns = {
        'code_assistant': [
            r'\b(code|coding|program|script|function|class|method|debug|fix|implement)\b',
            r'\b(syntax|error|bug|exception|traceback)\b',
            r'\b(write.*code|show.*code|help.*cod)\b'
        ],
        'practice_facilitator': [
            r'\b(practice|exercise|example|hands.?on|try|do|build)\b',
            r'\b(challenge|task|assignment|project)\b',
            r'\b(show me how|give me.*example)\b'
        ],
        'instructor': [
            r'\b(what is|explain|how does|teach|learn|understand)\b',
            r'\b(concept|theory|principle|definition|meaning)\b',
            r'\b(why|how|when|where)\b.*\?'
        ],
        'troubleshooter': [
            r'\b(error|problem|issue|trouble|broken|not working)\b',
            r'\b(fix|solve|resolve|debug|help)\b',
            r'\b(why.*not|doesn\'t work|failed)\b'
        ],
        'documentation_expert': [
            r'\b(documentation|docs|api|reference|manual)\b',
            r'\b(official|standard|specification)\b'
        ],
        'assessment': [
            r'\b(test|quiz|assess|evaluate|check|exam)\b',
            r'\b(knowledge|understanding|skill)\b.*\b(test|check)\b'
        ]
    }
    
    # Score each agent based on pattern matches
    agent_scores = {}
    for agent, agent_patterns in patterns.items():
        score = 0
        for pattern in agent_patterns:
            matches = len(re.findall(pattern, query_lower))
            score += matches
        
        if score > 0:
            agent_scores[agent] = score
    
    # Consider learning context
    learning_stage = learning_context.get('learning_stage', '').lower()
    if learning_stage == 'beginner' and 'instructor' in agent_scores:
        agent_scores['instructor'] += 1
    elif learning_stage == 'intermediate' and 'practice_facilitator' in agent_scores:
        agent_scores['practice_facilitator'] += 1
    elif learning_stage == 'advanced' and 'code_assistant' in agent_scores:
        agent_scores['code_assistant'] += 1
    
    # Return the highest scoring agent
    if agent_scores:
        return max(agent_scores, key=agent_scores.get)
    
    return None


def get_orchestration_stats() -> Dict:
    """Get orchestration performance statistics."""
    global _agent_performance_stats, _agent_selection_cache
    
    return {
        'agent_performance': dict(_agent_performance_stats),
        'cache_size': len(_agent_selection_cache),
        'cache_hit_rate': _calculate_cache_hit_rate()
    }


def _calculate_cache_hit_rate() -> float:
    """Calculate cache hit rate (simplified implementation)."""
    # This is a simplified implementation
    # In a real system, you'd track cache hits vs misses
    return 0.0  # Placeholder


def clear_orchestration_cache() -> None:
    """Clear the orchestration cache."""
    global _agent_selection_cache
    _agent_selection_cache.clear()


def reset_performance_stats() -> None:
    """Reset all performance statistics."""
    global _agent_performance_stats
    _agent_performance_stats.clear()