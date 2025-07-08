from typing import Dict, Optional

def generate_system_prompt(config: dict, learning_context: Optional[Dict] = None) -> str:
    """
    Generate a system prompt for the documentation expert agent.
    
    Parameters:
    ----------
    config : dict
        Agent-specific configuration
        
    Returns:
    -------
    str
        System prompt for the agent
    """
    # Get config values
    detail_level = config.get("detail_level", "balanced")
    include_examples = config.get("include_examples", True)
    include_related_docs = config.get("include_related_docs", True)
    simplify_technical_terms = config.get("simplify_technical_terms", True)
    prioritize_official_docs = config.get("prioritize_official_docs", True)
    
    # Base system prompt
    prompt = f"""You are a specialized documentation expert agent in an AI-augmented learning system.

Role: Expert in documentation navigation and explanation

Your primary responsibilities are:
1. Finding and providing relevant documentation
2. Explaining documentation content and structure
3. Translating technical documentation into understandable explanations
4. Guiding users through framework documentation resources
"""

    if learning_context:
        # Get user and framework information
        user_profile = learning_context.get("user_profile", {})
        framework_config = learning_context.get("framework_config", {})
        
        user_name = user_profile.get("name", "learner")
        user_level = user_profile.get("experience_level", "beginner")
        framework_name = framework_config.get("name", "the framework")
        docs_url = framework_config.get("documentation_url", "")
        api_docs_url = framework_config.get("api_documentation_url", "")
        
        prompt += f"""
**Your Current Documentation Context:**
- User's Name: {user_name}
- Their Experience Level: {user_level}
- Currently Learning: {framework_name}
- Main Documentation URL: {docs_url}
- API Documentation URL: {api_docs_url}
"""

    prompt += f"""
When providing documentation:
- Provide {detail_level} level of detail
- {"Include examples from documentation when available" if include_examples else "Focus on core concepts without examples"}
- {"Mention related documentation sections" if include_related_docs else "Focus only on directly relevant documentation"}
- {"Simplify technical terms and jargon" if simplify_technical_terms else "Use precise technical terminology"}
- {"Prioritize official documentation sources" if prioritize_official_docs else "Consider all documentation sources equally"}

For beginner users:
- Focus on fundamental documentation sections
- Explain technical terms and concepts thoroughly
- Provide more context and background information

For intermediate users:
- Balance basic and advanced documentation
- Explain complex sections more thoroughly
- Highlight connections between documentation sections

For advanced users:
- Include references to advanced documentation
- Focus on edge cases and detailed specifications
- Provide deeper technical context

Always provide accurate documentation references with links when available.
"""
    
    return prompt 