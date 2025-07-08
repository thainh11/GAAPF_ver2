"""
Centralized Orchestration Logic for GAAPF.

This module provides functions for using an LLM to orchestrate and coordinate
the behavior of specialized AI agents within the GAAPF architecture.
"""
import json
import logging
import re
from typing import Dict, List, Optional

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def llm_orchestrate_agent(
    llm: BaseLanguageModel,
    interaction_data: Dict,
    learning_context: Dict,
    available_agents: List[str],
    is_logging: bool = False,
) -> Optional[Dict]:
    """
    Use the LLM to determine the primary agent and task for a user query.

    Args:
        llm: The language model to use for orchestration.
        interaction_data: The current user interaction data.
        learning_context: The current learning session context.
        available_agents: A list of available agent types.
        is_logging: Flag to enable detailed logging.

    Returns:
        A dictionary with "primary_agent" and "task", or None if orchestration fails.
    """
    try:
        # Create a concise, serializable version of the learning context
        context_summary = {
            k: v
            for k, v in learning_context.items()
            if k not in ["messages", "framework_config"]
        }

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

**User Query:**
"{interaction_data.get('query', '')}"

**Learning Context:**
{json.dumps(context_summary, indent=2, default=str)}

**Your Task:**
Based on all the information above, decide which single agent is best suited to handle this request. Respond with a JSON object containing the `primary_agent` and a concise `task` for that agent.

**JSON Response Format:**
{{
  "primary_agent": "<agent_type>",
  "task": "<A short, clear description of what the agent should do>"
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

        # Robust JSON extraction
        match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", content, re.DOTALL)
        if match:
            json_str = match.group(1) or match.group(2)
            orchestration = json.loads(json_str)
            if "primary_agent" in orchestration and "task" in orchestration:
                if is_logging:
                    logger.info(f"LLM Orchestration successful: {orchestration}")
                return orchestration
            else:
                raise ValueError("Missing 'primary_agent' or 'task' in JSON response.")
        else:
            raise ValueError("No valid JSON object found in the LLM response.")

    except (json.JSONDecodeError, ValueError) as e:
        if is_logging:
            logger.error(f"LLM orchestration parsing failed: {e}\nRaw response: {content}")
    except Exception as e:
        if is_logging:
            logger.error(f"An unexpected error occurred during LLM orchestration: {e}")

    return None 