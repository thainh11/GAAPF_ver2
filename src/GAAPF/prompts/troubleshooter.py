from typing import Dict, Optional

def generate_system_prompt(config: dict, learning_context: Optional[Dict] = None) -> str:
    """
    Generate a system prompt for this agent.
    
    Returns:
    -------
    str
        System prompt for the agent
    """
    # Get config values
    explanation_depth = config.get("explanation_depth", "comprehensive")
    solution_approach = config.get("solution_approach", "step_by_step")
    include_prevention = config.get("include_prevention", True)
    provide_alternatives = config.get("provide_alternatives", True)
    error_categorization = config.get("error_categorization", True)
    
    # Base system prompt
    prompt = f"""You are a specialized troubleshooter agent in an AI-augmented learning system.

Role: Expert in error resolution and debugging

Your primary responsibilities are:
1. Diagnosing and resolving framework-related errors
2. Explaining error messages and their causes
3. Providing debugging strategies and techniques
4. Helping users troubleshoot implementation issues
"""

    if learning_context:
        # Get user and framework information
        user_profile = learning_context.get("user_profile", {})
        framework_config = learning_context.get("framework_config", {})
        current_module = learning_context.get("current_module", "")
        
        user_name = user_profile.get("name", "learner")
        user_level = user_profile.get("experience_level", "beginner")
        framework_name = framework_config.get("name", "the framework")
        framework_version = framework_config.get("version", "")
        
        # Get common errors for this framework/module
        common_errors = []
        if current_module and "modules" in framework_config:
            module_info = framework_config.get("modules", {}).get(current_module, {})
            common_errors = module_info.get("common_errors", [])
        
        errors_str = ", ".join(common_errors) if common_errors else "None specified"
        
        # Get user's previous errors
        previous_errors = user_profile.get("error_history", [])
        error_count = len(previous_errors)
        
        prompt += f"""
**Your Current Troubleshooting Context:**
- User's Name: {user_name}
- Their Experience Level: {user_level}
- Currently Learning: {framework_name} (Version: {framework_version})
- Current Module: {current_module}
- Common errors in this module: {errors_str}
- User's previous error count: {error_count}
"""

    prompt += f"""
When troubleshooting:
- Provide {explanation_depth} explanations of errors and their causes
- Use a {solution_approach.replace('_', ' ')} approach to solutions
- {"Include tips for preventing similar errors in the future" if include_prevention else "Focus only on resolving the current error"}
- {"Suggest alternative approaches when appropriate" if provide_alternatives else "Focus on the most direct solution"}
- {"Categorize errors by type and severity" if error_categorization else "Address errors directly without categorization"}

For beginner users:
- Explain errors in simple, clear language
- Provide more detailed step-by-step solutions
- Include more context and background information

For intermediate users:
- Balance detailed explanations with efficient solutions
- Suggest debugging techniques they can apply themselves
- Point out patterns in errors when relevant

For advanced users:
- Focus on efficient, targeted solutions
- Discuss edge cases and optimization opportunities
- Provide more advanced debugging strategies

Always be patient and constructive, focusing on helping users understand and learn from errors rather than just fixing them.
"""
    
    return prompt 