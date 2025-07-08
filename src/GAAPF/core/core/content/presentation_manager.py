"""
Content presentation and formatting module for the GAAPF framework.

This module handles formatting and presentation of generated content
including theory, code examples, and quiz materials.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.table import Table
from rich.columns import Columns
from rich.text import Text

logger = logging.getLogger(__name__)

class PresentationManager:
    """
    Content presentation and formatting manager.
    
    Handles formatting and presentation of:
    - Theory content with rich formatting
    - Code examples with syntax highlighting
    - Quiz questions with interactive elements
    - Progress tracking displays
    """
    
    def __init__(self, console: Console = None, is_logging: bool = False):
        """
        Initialize the presentation manager.
        
        Parameters:
        ----------
        console : Console, optional
            Rich console instance for output
        is_logging : bool
            Enable detailed logging
        """
        self.console = console or Console()
        self.is_logging = is_logging
        
        # Formatting styles
        self.styles = {
            "title": "bold blue",
            "subtitle": "bold cyan",
            "concept": "bold green",
            "code": "bright_black on white",
            "success": "bold green",
            "warning": "bold yellow",
            "error": "bold red",
            "info": "blue"
        }
        
        # Content templates
        self.content_templates = {
            "theory": self._get_theory_template(),
            "code": self._get_code_template(),
            "quiz": self._get_quiz_template(),
            "progress": self._get_progress_template()
        }
        
        if self.is_logging:
            logger.info("PresentationManager initialized")
    
    def present_theory_content(
        self,
        theory_content: Dict[str, Any],
        display_mode: str = "interactive"
    ) -> None:
        """
        Present theory content with rich formatting.
        
        Parameters:
        ----------
        theory_content : Dict[str, Any]
            Theory content from TheoryGenerator
        display_mode : str
            Display mode (interactive/summary/full)
        """
        try:
            if self.is_logging:
                logger.info(f"Presenting theory content: {theory_content.get('title', 'Unknown')}")
            
            content = theory_content.get("content", {})
            metadata = theory_content.get("metadata", {})
            
            # Display title and introduction
            self._display_content_header(
                title=content.get("title", "Theory Content"),
                topic=metadata.get("topic", ""),
                framework=metadata.get("framework_name", ""),
                reading_time=metadata.get("estimated_reading_time", 5)
            )
            
            # Display learning objectives
            if "learning_objectives" in content:
                self._display_learning_objectives(content["learning_objectives"])
            
            # Display main content sections
            if "sections" in content:
                self._display_theory_sections(content["sections"], display_mode)
            
            # Display summary and next steps
            if "summary" in content:
                self._display_summary(content["summary"])
            
            if "next_steps" in content:
                self._display_next_steps(content["next_steps"])
            
            # Display glossary if available
            if "glossary" in content:
                self._display_glossary(content["glossary"])
                
        except Exception as e:
            logger.error(f"Error presenting theory content: {str(e)}")
            self.console.print(f"[{self.styles['error']}]Error displaying content: {str(e)}[/]")
    
    def present_code_example(
        self,
        code_content: Dict[str, Any],
        show_explanations: bool = True,
        show_exercises: bool = True
    ) -> None:
        """
        Present code examples with syntax highlighting and explanations.
        
        Parameters:
        ----------
        code_content : Dict[str, Any]
            Code content from CodeGenerator
        show_explanations : bool
            Whether to show code explanations
        show_exercises : bool
            Whether to show practice exercises
        """
        try:
            if self.is_logging:
                logger.info(f"Presenting code example: {code_content.get('concept', 'Unknown')}")
            
            content = code_content.get("code_content", {})
            metadata = code_content.get("metadata", {})
            
            # Display code header
            self._display_code_header(
                title=content.get("title", "Code Example"),
                concept=code_content.get("concept", ""),
                framework=code_content.get("framework_name", ""),
                language=code_content.get("language", "python"),
                completion_time=metadata.get("estimated_completion_time", 30)
            )
            
            # Display description
            if "description" in content:
                self.console.print(Panel(
                    content["description"],
                    title="Description",
                    border_style="cyan"
                ))
                self.console.print()
            
            # Display main code with syntax highlighting
            if "main_code" in content:
                self._display_code_block(
                    code=content["main_code"],
                    language=code_content.get("language", "python"),
                    title="Code Example"
                )
            
            # Display explanation if requested
            if show_explanations and "explanation" in content:
                self._display_code_explanation(content["explanation"])
            
            # Display line-by-line breakdown if available
            if show_explanations and "line_by_line" in content:
                self._display_line_by_line_explanation(content["line_by_line"])
            
            # Display expected output
            if "expected_output" in content:
                self._display_expected_output(content["expected_output"])
            
            # Display practice exercises if requested
            if show_exercises and "exercises" in code_content:
                self._display_practice_exercises(code_content["exercises"])
                
        except Exception as e:
            logger.error(f"Error presenting code example: {str(e)}")
            self.console.print(f"[{self.styles['error']}]Error displaying code: {str(e)}[/]")
    
    def present_quiz(
        self,
        quiz_data: Dict[str, Any],
        question_index: int = 0,
        show_all: bool = False
    ) -> None:
        """
        Present quiz questions with interactive formatting.
        
        Parameters:
        ----------
        quiz_data : Dict[str, Any]
            Quiz data from QuizGenerator
        question_index : int
            Index of current question to display
        show_all : bool
            Whether to show all questions at once
        """
        try:
            if self.is_logging:
                logger.info("Presenting quiz content")
            
            quiz_metadata = quiz_data.get("quiz_metadata", {})
            questions = quiz_data.get("questions", [])
            
            # Display quiz header
            self._display_quiz_header(quiz_metadata)
            
            if show_all:
                # Display all questions
                for i, question in enumerate(questions, 1):
                    self._display_single_question(question, i, len(questions))
                    if i < len(questions):
                        self.console.print("─" * 50)
            else:
                # Display single question
                if 0 <= question_index < len(questions):
                    self._display_single_question(
                        questions[question_index], 
                        question_index + 1, 
                        len(questions)
                    )
                else:
                    self.console.print(f"[{self.styles['error']}]Invalid question index[/]")
                    
        except Exception as e:
            logger.error(f"Error presenting quiz: {str(e)}")
            self.console.print(f"[{self.styles['error']}]Error displaying quiz: {str(e)}[/]")
    
    def present_progress_summary(
        self,
        progress_data: Dict[str, Any],
        include_details: bool = True
    ) -> None:
        """
        Present learning progress summary.
        
        Parameters:
        ----------
        progress_data : Dict[str, Any]
            Progress data to display
        include_details : bool
            Whether to include detailed breakdown
        """
        try:
            if self.is_logging:
                logger.info("Presenting progress summary")
            
            # Display progress header
            self._display_progress_header(progress_data)
            
            # Display progress metrics
            self._display_progress_metrics(progress_data)
            
            # Display detailed breakdown if requested
            if include_details:
                self._display_progress_details(progress_data)
                
        except Exception as e:
            logger.error(f"Error presenting progress: {str(e)}")
            self.console.print(f"[{self.styles['error']}]Error displaying progress: {str(e)}[/]")
    
    def _display_content_header(
        self,
        title: str,
        topic: str,
        framework: str,
        reading_time: int
    ) -> None:
        """Display content header with metadata."""
        header_text = f"[{self.styles['title']}]{title}[/]\n\n"
        if topic:
            header_text += f"📚 Topic: [{self.styles['concept']}]{topic}[/]\n"
        if framework:
            header_text += f"🔧 Framework: [{self.styles['concept']}]{framework}[/]\n"
        header_text += f"⏱️  Estimated reading time: [{self.styles['info']}]{reading_time} minutes[/]"
        
        self.console.print(Panel(
            header_text,
            title="Learning Content",
            border_style="blue"
        ))
        self.console.print()
    
    def _display_learning_objectives(self, objectives: List[str]) -> None:
        """Display learning objectives."""
        if not objectives:
            return
            
        objective_text = "\n".join([f"• {obj}" for obj in objectives])
        self.console.print(Panel(
            objective_text,
            title="Learning Objectives",
            border_style="green"
        ))
        self.console.print()
    
    def _display_theory_sections(
        self,
        sections: List[Dict[str, Any]],
        display_mode: str
    ) -> None:
        """Display theory content sections."""
        for i, section in enumerate(sections, 1):
            title = section.get("title", f"Section {i}")
            content = section.get("content", "")
            key_points = section.get("key_points", [])
            
            # Display section title
            self.console.print(f"[{self.styles['subtitle']}]{i}. {title}[/]\n")
            
            # Display content
            if display_mode == "summary":
                # Show only key points in summary mode
                if key_points:
                    for point in key_points:
                        self.console.print(f"  • {point}")
            else:
                # Show full content
                self.console.print(content)
                
                # Show key points if available
                if key_points:
                    self.console.print(f"\n[{self.styles['concept']}]Key Points:[/]")
                    for point in key_points:
                        self.console.print(f"  • {point}")
            
            # Display examples if available
            if "examples" in section:
                self._display_section_examples(section["examples"])
            
            self.console.print("\n" + "─" * 50 + "\n")
    
    def _display_section_examples(self, examples: List[Dict[str, str]]) -> None:
        """Display examples for a section."""
        if not examples:
            return
            
        self.console.print(f"\n[{self.styles['info']}]Examples:[/]")
        for example in examples:
            title = example.get("title", "Example")
            description = example.get("description", "")
            scenario = example.get("scenario", "")
            
            example_text = f"**{title}**\n{description}"
            if scenario:
                example_text += f"\n\n*Scenario:* {scenario}"
            
            self.console.print(Panel(
                Markdown(example_text),
                border_style="dim"
            ))
    
    def _display_code_header(
        self,
        title: str,
        concept: str,
        framework: str,
        language: str,
        completion_time: int
    ) -> None:
        """Display code example header."""
        header_text = f"[{self.styles['title']}]{title}[/]\n\n"
        if concept:
            header_text += f"💡 Concept: [{self.styles['concept']}]{concept}[/]\n"
        if framework:
            header_text += f"🔧 Framework: [{self.styles['concept']}]{framework}[/]\n"
        header_text += f"💻 Language: [{self.styles['concept']}]{language}[/]\n"
        header_text += f"⏱️  Estimated time: [{self.styles['info']}]{completion_time} minutes[/]"
        
        self.console.print(Panel(
            header_text,
            title="Code Example",
            border_style="cyan"
        ))
        self.console.print()
    
    def _display_code_block(
        self,
        code: str,
        language: str,
        title: str = "Code"
    ) -> None:
        """Display code with syntax highlighting."""
        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=True,
            word_wrap=True
        )
        
        self.console.print(Panel(
            syntax,
            title=title,
            border_style="bright_black"
        ))
        self.console.print()
    
    def _display_code_explanation(self, explanation: str) -> None:
        """Display code explanation."""
        self.console.print(Panel(
            explanation,
            title="Explanation",
            border_style="yellow"
        ))
        self.console.print()
    
    def _display_expected_output(self, output: str) -> None:
        """Display expected output."""
        self.console.print(Panel(
            f"[{self.styles['success']}]{output}[/]",
            title="Expected Output",
            border_style="green"
        ))
        self.console.print()
    
    def _display_quiz_header(self, metadata: Dict[str, Any]) -> None:
        """Display quiz header information."""
        title = metadata.get("topic", "Quiz")
        framework = metadata.get("framework_name", "")
        total_questions = metadata.get("total_questions", 0)
        total_points = metadata.get("total_points", 0)
        estimated_time = metadata.get("estimated_time", 0)
        
        header_text = f"[{self.styles['title']}]{title} Quiz[/]\n\n"
        if framework:
            header_text += f"🔧 Framework: [{self.styles['concept']}]{framework}[/]\n"
        header_text += f"📝 Questions: [{self.styles['info']}]{total_questions}[/]\n"
        header_text += f"🎯 Total Points: [{self.styles['info']}]{total_points}[/]\n"
        header_text += f"⏱️  Estimated Time: [{self.styles['info']}]{estimated_time} minutes[/]"
        
        self.console.print(Panel(
            header_text,
            title="Assessment",
            border_style="magenta"
        ))
        self.console.print()
    
    def _display_single_question(
        self,
        question: Dict[str, Any],
        question_num: int,
        total_questions: int
    ) -> None:
        """Display a single quiz question."""
        question_type = question.get("question_type", "unknown")
        points = question.get("points", 0)
        
        # Question header
        header = f"Question {question_num}/{total_questions} ({question_type.replace('_', ' ').title()}) - {points} points"
        self.console.print(f"[{self.styles['subtitle']}]{header}[/]\n")
        
        # Display question based on type
        if question_type == "multiple_choice":
            self._display_multiple_choice_question(question)
        elif question_type == "code_completion":
            self._display_code_completion_question(question)
        elif question_type == "conceptual":
            self._display_conceptual_question(question)
        else:
            # Generic question display
            self.console.print(question.get("question", "No question text available"))
        
        self.console.print()
    
    def _display_multiple_choice_question(self, question: Dict[str, Any]) -> None:
        """Display multiple choice question."""
        self.console.print(question.get("question", ""))
        self.console.print()
        
        options = question.get("options", [])
        for i, option in enumerate(options, 1):
            self.console.print(f"  {chr(64+i)}. {option}")
    
    def _display_code_completion_question(self, question: Dict[str, Any]) -> None:
        """Display code completion question."""
        self.console.print(question.get("question", ""))
        self.console.print()
        
        if "incomplete_code" in question:
            self._display_code_block(
                question["incomplete_code"],
                "python",  # Default to Python
                "Code to Complete"
            )
    
    def _display_conceptual_question(self, question: Dict[str, Any]) -> None:
        """Display conceptual question."""
        self.console.print(question.get("question", ""))
        
        if "key_points" in question:
            self.console.print(f"\n[{self.styles['info']}]Consider these aspects:[/]")
            for point in question["key_points"]:
                self.console.print(f"  • {point}")
    
    def _get_theory_template(self) -> Dict[str, str]:
        """Get theory content template."""
        return {
            "header": "theory_header",
            "content": "theory_content",
            "footer": "theory_footer"
        }
    
    def _get_code_template(self) -> Dict[str, str]:
        """Get code content template."""
        return {
            "header": "code_header",
            "content": "code_content",
            "footer": "code_footer"
        }
    
    def _get_quiz_template(self) -> Dict[str, str]:
        """Get quiz template."""
        return {
            "header": "quiz_header",
            "content": "quiz_content",
            "footer": "quiz_footer"
        }
    
    def _get_progress_template(self) -> Dict[str, str]:
        """Get progress template."""
        return {
            "header": "progress_header",
            "content": "progress_content",
            "footer": "progress_footer"
        } 