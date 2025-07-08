"""
Content generation package for the GAAPF framework.

This package provides AI-powered content generation capabilities including:
- Theory content generation
- Code example generation  
- Quiz creation and management
- Content presentation and formatting
"""

from .theory_generator import TheoryGenerator
from .code_generator import CodeGenerator
from .quiz_generator import QuizGenerator
from .presentation_manager import PresentationManager

__all__ = [
    'TheoryGenerator',
    'CodeGenerator', 
    'QuizGenerator',
    'PresentationManager'
] 