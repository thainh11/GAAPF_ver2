#!/usr/bin/env python3
"""
Script to visualize the agent's main flow graph and export it as a PNG image.
"""

import os
import sys
import json
from pathlib import Path
import graphviz
import argparse

# Add the parent directory to the path so we can import the GAAPF package
sys.path.append(str(Path(__file__).parent / "GAAPF-main"))

def create_agent_flow_diagram(output_file="agent_flow_graph.png"):
    """
    Create a visualization of the agent's main flow graph
    
    Args:
        output_file: Path to save the PNG file
    
    Returns:
        Path to the exported PNG file
    """
    # Create a new Digraph object
    dot = graphviz.Digraph(comment='Agent Flow Graph', format='png')
    
    # Set graph attributes for better visualization
    dot.attr(rankdir='LR')  # Left to right layout
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontname='Arial')
    dot.attr('edge', fontname='Arial')
    
    # Add nodes for main components
    dot.node('User', 'User', shape='ellipse', fillcolor='lightgreen')
    dot.node('Agent', 'Agent')
    dot.node('LTM', 'Long-Term Memory')
    dot.node('Tools', 'Tools Manager')
    dot.node('LLM', 'Language Model')
    
    # Add edges to show the flow
    dot.edge('User', 'Agent', label='Query')
    
    # Memory flow
    dot.edge('Agent', 'LTM', label='1. Check Memory')
    dot.edge('LTM', 'Agent', label='Return Context')
    
    # LLM flow
    dot.edge('Agent', 'LLM', label='2. Process Query + Context')
    dot.edge('LLM', 'Agent', label='Response or Tool Call')
    
    # Tool flow
    dot.edge('Agent', 'Tools', label='3. Execute Tool (if needed)')
    dot.edge('Tools', 'Agent', label='Tool Result')
    
    # Final response
    dot.edge('Agent', 'User', label='Final Response')
    
    # Save memory flow
    dot.edge('Agent', 'LTM', label='4. Save Interaction', color='darkgreen', style='dashed')
    
    # Render the graph
    output_path = dot.render(filename=output_file.replace('.png', ''), cleanup=True)
    
    print(f"Successfully created agent flow visualization: {output_path}")
    return output_path

def create_detailed_agent_flow_diagram(output_file="agent_flow_detailed.png"):
    """
    Create a more detailed visualization of the agent's flow graph
    
    Args:
        output_file: Path to save the PNG file
    
    Returns:
        Path to the exported PNG file
    """
    # Create a new Digraph object
    dot = graphviz.Digraph(comment='Detailed Agent Flow Graph', format='png')
    
    # Set graph attributes for better visualization
    dot.attr(rankdir='TB')  # Top to bottom layout
    dot.attr('node', shape='box', style='filled', fontname='Arial')
    dot.attr('edge', fontname='Arial')
    
    # Add nodes for main components with different colors
    dot.node('User', 'User', shape='ellipse', fillcolor='lightgreen')
    dot.node('Agent', 'Agent', fillcolor='lightblue')
    
    # Memory components
    dot.node('STM', 'Short-Term Memory', fillcolor='lightyellow')
    dot.node('LTM', 'Long-Term Memory', fillcolor='lightyellow')
    dot.node('VectorDB', 'Vector Database\n(ChromaDB)', fillcolor='lightyellow')
    
    # Processing components
    dot.node('LLM', 'Language Model', fillcolor='lightpink')
    dot.node('Tools', 'Tools Manager', fillcolor='lightcyan')
    dot.node('Graph', 'Memory Graph\nTransformer', fillcolor='lightgrey')
    
    # Subgraph for tools
    with dot.subgraph(name='cluster_tools') as c:
        c.attr(label='Available Tools')
        c.attr('node', shape='box', style='filled', fillcolor='lightcyan')
        c.node('Tool1', 'Computer Tools')
        c.node('Tool2', 'Web Search')
        c.node('Tool3', 'Terminal Tools')
        c.node('Tool4', 'DeepSearch')
        c.node('Tool5', 'Trending News')
    
    # Add edges to show the flow
    # User interaction
    dot.edge('User', 'Agent', label='1. Query')
    
    # Memory retrieval flow
    dot.edge('Agent', 'LTM', label='2. Check Memory')
    dot.edge('LTM', 'VectorDB', label='Query Similar')
    dot.edge('VectorDB', 'LTM', label='Return Embeddings')
    dot.edge('LTM', 'Agent', label='3. Return Context')
    
    # LLM processing flow
    dot.edge('Agent', 'LLM', label='4. Process Query + Context')
    dot.edge('LLM', 'Agent', label='5. Response or Tool Call')
    
    # Tool execution flow
    dot.edge('Agent', 'Tools', label='6. Execute Tool (if needed)')
    dot.edge('Tools', 'Tool1', style='dashed')
    dot.edge('Tools', 'Tool2', style='dashed')
    dot.edge('Tools', 'Tool3', style='dashed')
    dot.edge('Tools', 'Tool4', style='dashed')
    dot.edge('Tools', 'Tool5', style='dashed')
    dot.edge('Tools', 'Agent', label='7. Tool Result')
    
    # Final response
    dot.edge('Agent', 'User', label='8. Final Response')
    
    # Memory saving flow
    dot.edge('Agent', 'Graph', label='9. Transform to Graph', color='darkgreen')
    dot.edge('Graph', 'STM', label='Save JSON', color='darkgreen')
    dot.edge('Graph', 'LTM', label='Save Embeddings', color='darkgreen')
    dot.edge('LTM', 'VectorDB', label='Store Vectors', color='darkgreen')
    
    # Render the graph
    output_path = dot.render(filename=output_file.replace('.png', ''), cleanup=True)
    
    print(f"Successfully created detailed agent flow visualization: {output_path}")
    return output_path

def create_constellation_graph(output_file="constellation_graph.png"):
    """
    Create a visualization of the constellation graph
    
    Args:
        output_file: Path to save the PNG file
    
    Returns:
        Path to the exported PNG file
    """
    # Create a new Digraph object
    dot = graphviz.Digraph(comment='Constellation Graph', format='png')
    
    # Set graph attributes for better visualization
    dot.attr(rankdir='TB')  # Top to bottom layout
    dot.attr('node', shape='box', style='filled', fontname='Arial')
    dot.attr('edge', fontname='Arial')
    
    # Add nodes for main components
    dot.node('User', 'User', shape='ellipse', fillcolor='lightgreen')
    dot.node('Constellation', 'Constellation Graph', fillcolor='lightblue')
    
    # Add specialized agent nodes
    dot.node('Instructor', 'Instructor Agent', fillcolor='lightyellow')
    dot.node('CodeAssistant', 'Code Assistant Agent', fillcolor='lightyellow')
    dot.node('Mentor', 'Mentor Agent', fillcolor='lightyellow')
    dot.node('ResearchAssistant', 'Research Assistant Agent', fillcolor='lightyellow')
    dot.node('KnowledgeSynthesizer', 'Knowledge Synthesizer Agent', fillcolor='lightyellow')
    dot.node('DocExpert', 'Documentation Expert Agent', fillcolor='lightyellow')
    
    # Add edges to show the flow
    dot.edge('User', 'Constellation', label='Query')
    
    # Constellation to agents
    dot.edge('Constellation', 'Instructor', label='Primary Agent')
    dot.edge('Instructor', 'CodeAssistant', label='Handoff (code tasks)')
    dot.edge('Instructor', 'ResearchAssistant', label='Handoff (research tasks)')
    dot.edge('ResearchAssistant', 'KnowledgeSynthesizer', label='Handoff (synthesis)')
    dot.edge('CodeAssistant', 'DocExpert', label='Handoff (documentation)')
    dot.edge('Instructor', 'Mentor', label='Handoff (guidance)')
    
    # Return flow
    dot.edge('CodeAssistant', 'Constellation', label='Response', style='dashed')
    dot.edge('ResearchAssistant', 'Constellation', label='Response', style='dashed')
    dot.edge('KnowledgeSynthesizer', 'Constellation', label='Response', style='dashed')
    dot.edge('DocExpert', 'Constellation', label='Response', style='dashed')
    dot.edge('Mentor', 'Constellation', label='Response', style='dashed')
    
    # Final response
    dot.edge('Constellation', 'User', label='Final Response')
    
    # Render the graph
    output_path = dot.render(filename=output_file.replace('.png', ''), cleanup=True)
    
    print(f"Successfully created constellation graph visualization: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Create visualizations of agent flow graphs')
    parser.add_argument('--type', type=str, default='all', choices=['simple', 'detailed', 'constellation', 'all'], 
                        help='Type of graph to generate')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save the PNG files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.type == 'simple' or args.type == 'all':
        output_file = os.path.join(args.output_dir, 'agent_flow_graph.png')
        create_agent_flow_diagram(output_file)
    
    if args.type == 'detailed' or args.type == 'all':
        output_file = os.path.join(args.output_dir, 'agent_flow_detailed.png')
        create_detailed_agent_flow_diagram(output_file)
    
    if args.type == 'constellation' or args.type == 'all':
        output_file = os.path.join(args.output_dir, 'constellation_graph.png')
        create_constellation_graph(output_file)

if __name__ == "__main__":
    main() 