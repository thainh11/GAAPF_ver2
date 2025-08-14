"""
Response Formatter - Scaffolded responses for Study Mode
Implements structured responses with 3 blocks: Recap, Key Connections, Next Steps
Based on OpenAI Study Mode principles for managing cognitive load.
"""

from typing import List, Dict, Any, Optional
import re


def format_scaffold(
    answer: str,
    highlights: List[str] = None,
    next_steps: List[str] = None,
    framework: str = "programming"
) -> str:
    """
    Format response into scaffolded structure with 3 blocks.
    
    Args:
        answer: Main answer content
        highlights: Key connections/concepts to highlight
        next_steps: Suggested next learning steps
        framework: Current learning framework context
        
    Returns:
        Formatted markdown string with 3 structured blocks
    """
    if highlights is None:
        highlights = []
    if next_steps is None:
        next_steps = []
    
    # Clean and format main answer
    recap = f"### 📝 Concept Recap\n{answer.strip()}"
    
    # Format key connections
    if highlights:
        links = "\n".join(f"- {highlight}" for highlight in highlights)
        connect = f"### 🔗 Key Connections\n{links}"
    else:
        connect = f"### 🔗 Key Connections\n- This concept builds on fundamental {framework} principles\n- Consider how this relates to your previous learning"
    
    # Format next steps
    if next_steps:
        steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(next_steps))
        nxt = f"### 🚀 Try Next\n{steps}"
    else:
        nxt = f"### 🚀 Try Next\n1. Practice with a simple example\n2. Explore related concepts\n3. Ask follow-up questions"
    
    return "\n\n".join([recap, connect, nxt])


def extract_key_concepts(text: str, framework: str = "programming") -> List[str]:
    """
    Extract key technical concepts from text for highlighting.
    
    Args:
        text: Text to analyze
        framework: Framework context for concept extraction
        
    Returns:
        List of key concepts found
    """
    # Framework-specific concept keywords
    framework_concepts = {
        "langchain": [
            "chain", "agent", "tool", "memory", "prompt", "llm", 
            "retriever", "vectorstore", "document", "embedding"
        ],
        "langgraph": [
            "graph", "node", "edge", "state", "workflow", "checkpoint", 
            "stream", "conditional", "parallel", "router"
        ],
        "crewai": [
            "crew", "agent", "task", "role", "goal", "backstory", 
            "tool", "collaboration", "delegation"
        ],
        "autogen": [
            "conversation", "agent", "chat", "group", "assistant", 
            "user", "proxy", "termination"
        ]
    }
    
    # General programming concepts
    general_concepts = [
        "function", "class", "method", "variable", "api", "database", 
        "file", "error", "exception", "async", "sync", "callback"
    ]
    
    concepts = []
    text_lower = text.lower()
    
    # Get framework-specific concepts
    if framework in framework_concepts:
        for concept in framework_concepts[framework]:
            if concept in text_lower:
                concepts.append(concept.title())
    
    # Add general programming concepts
    for concept in general_concepts:
        if concept in text_lower and concept.title() not in concepts:
            concepts.append(concept.title())
    
    return concepts[:3]  # Limit to top 3 concepts


def generate_next_steps(query: str, answer: str, framework: str = "programming") -> List[str]:
    """
    Generate contextual next learning steps based on query and answer.
    
    Args:
        query: Original user query
        answer: Generated answer
        framework: Learning framework context
        
    Returns:
        List of suggested next steps
    """
    query_lower = query.lower()
    answer_lower = answer.lower()
    
    next_steps = []
    
    # Query type-based suggestions
    if any(word in query_lower for word in ["what is", "define", "explain"]):
        next_steps.extend([
            f"Try implementing a simple {framework} example",
            "Ask about real-world use cases",
            "Explore related concepts and patterns"
        ])
    elif any(word in query_lower for word in ["how to", "implement", "create"]):
        next_steps.extend([
            "Practice with different variations",
            "Add error handling and edge cases",
            "Optimize for production use"
        ])
    elif any(word in query_lower for word in ["why", "difference", "compare"]):
        next_steps.extend([
            "Compare with alternative approaches",
            "Understand the trade-offs involved",
            "Test both approaches in practice"
        ])
    else:
        # Default suggestions
        next_steps.extend([
            "Practice with hands-on examples",
            "Explore advanced features",
            "Connect to real projects"
        ])
    
    return next_steps[:3]  # Limit to 3 steps


def format_study_mode_response(
    answer: str, 
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format response specifically for Study Mode with scaffolding.
    
    Args:
        answer: Main response content
        context: Learning context including framework, user level, etc.
        
    Returns:
        Dictionary with formatted content and metadata
    """
    framework = context.get("framework", "programming")
    query = context.get("original_query", "")
    
    # Extract concepts and generate next steps
    highlights = extract_key_concepts(answer, framework)
    next_steps = generate_next_steps(query, answer, framework)
    
    # Format with scaffolding
    scaffolded_content = format_scaffold(
        answer=answer,
        highlights=highlights,
        next_steps=next_steps,
        framework=framework
    )
    
    return {
        "content": scaffolded_content,
        "scaffold": True,
        "highlights": highlights,
        "next_steps": next_steps,
        "framework": framework,
        "type": "scaffolded_response"
    }


def format_direct_mode_response(
    answer: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format response for Direct Mode - still structured but more concise.
    
    Args:
        answer: Main response content
        context: Learning context
        
    Returns:
        Dictionary with formatted content and metadata
    """
    framework = context.get("framework", "programming")
    
    # Light scaffolding for cognitive load management
    highlights = extract_key_concepts(answer, framework)
    
    formatted_content = answer
    if highlights:
        key_points = "\n".join(f"• {highlight}" for highlight in highlights)
        formatted_content += f"\n\n**Key Points:** {key_points}"
    
    return {
        "content": formatted_content,
        "scaffold": False,
        "highlights": highlights,
        "framework": framework,
        "type": "direct_response"
    }