"""
Adaptive quiz generation module for the GAAPF framework.

This module provides AI-powered generation of quizzes and assessments
that adapt to user learning levels and progress.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import random

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

class QuizGenerator:
    """
    AI-powered adaptive quiz generator.
    
    Generates various types of assessments including:
    - Multiple choice questions
    - Code completion exercises
    - Conceptual questions
    - Practical application scenarios
    """
    
    def __init__(self, llm: BaseLanguageModel, is_logging: bool = False):
        """
        Initialize the quiz generator.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model for quiz generation
        is_logging : bool
            Enable detailed logging
        """
        self.llm = llm
        self.is_logging = is_logging
        
        # Quiz templates by type
        self.quiz_templates = {
            "multiple_choice": self._get_multiple_choice_template(),
            "code_completion": self._get_code_completion_template(),
            "conceptual": self._get_conceptual_template(),
            "practical_scenario": self._get_practical_scenario_template()
        }
        
        # Difficulty progression
        self.difficulty_levels = {
            "beginner": {"complexity": 1, "concept_depth": "basic"},
            "intermediate": {"complexity": 2, "concept_depth": "moderate"},
            "advanced": {"complexity": 3, "concept_depth": "deep"}
        }
        
        if self.is_logging:
            logger.info("QuizGenerator initialized")
    
    async def generate_adaptive_quiz(
        self,
        topic: str,
        framework_name: str,
        learning_level: str,
        concepts_covered: List[str],
        quiz_length: int = 5,
        question_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an adaptive quiz based on learning progress.
        
        Parameters:
        ----------
        topic : str
            Main topic for the quiz
        framework_name : str
            Target framework
        learning_level : str
            User's current learning level
        concepts_covered : List[str]
            Concepts that should be tested
        quiz_length : int
            Number of questions to generate
        question_types : List[str], optional
            Types of questions to include
            
        Returns:
        -------
        Dict[str, Any]
            Generated quiz with questions and metadata
        """
        try:
            if self.is_logging:
                logger.info(f"Generating adaptive quiz for topic: {topic}")
            
            # Default question types if not specified
            if question_types is None:
                question_types = ["multiple_choice", "conceptual", "code_completion"]
            
            # Generate questions
            questions = []
            for i in range(quiz_length):
                # Select question type (rotate through types)
                question_type = question_types[i % len(question_types)]
                
                # Select concept to test
                concept = concepts_covered[i % len(concepts_covered)]
                
                # Generate question
                question = await self._generate_question(
                    question_type=question_type,
                    concept=concept,
                    topic=topic,
                    framework_name=framework_name,
                    learning_level=learning_level,
                    difficulty=self.difficulty_levels[learning_level]
                )
                
                question["question_id"] = f"q_{i+1}"
                question["points"] = self._calculate_question_points(question_type, learning_level)
                questions.append(question)
            
            # Generate quiz metadata
            quiz_metadata = {
                "quiz_id": f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "topic": topic,
                "framework_name": framework_name,
                "learning_level": learning_level,
                "total_questions": quiz_length,
                "total_points": sum(q["points"] for q in questions),
                "estimated_time": quiz_length * 3,  # 3 minutes per question
                "concepts_tested": concepts_covered,
                "question_types": question_types,
                "generated_at": datetime.now().isoformat(),
                "adaptive_parameters": self.difficulty_levels[learning_level]
            }
            
            result = {
                "quiz_metadata": quiz_metadata,
                "questions": questions,
                "grading_criteria": self._generate_grading_criteria(questions),
                "feedback_templates": self._generate_feedback_templates(learning_level)
            }
            
            if self.is_logging:
                logger.info(f"Adaptive quiz generated with {quiz_length} questions")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating adaptive quiz: {str(e)}")
            raise
    
    async def generate_concept_assessment(
        self,
        concept: str,
        framework_name: str,
        learning_level: str,
        assessment_depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate focused assessment for a specific concept.
        
        Parameters:
        ----------
        concept : str
            Specific concept to assess
        framework_name : str
            Target framework
        learning_level : str
            User's learning level
        assessment_depth : str
            Depth of assessment (quick/standard/comprehensive)
            
        Returns:
        -------
        Dict[str, Any]
            Concept-focused assessment
        """
        try:
            # Determine question count based on depth
            question_counts = {
                "quick": 3,
                "standard": 5,
                "comprehensive": 8
            }
            
            question_count = question_counts.get(assessment_depth, 5)
            
            # Generate diverse question types for the concept
            assessment = await self.generate_adaptive_quiz(
                topic=concept,
                framework_name=framework_name,
                learning_level=learning_level,
                concepts_covered=[concept],
                quiz_length=question_count,
                question_types=["multiple_choice", "conceptual", "code_completion", "practical_scenario"]
            )
            
            # Enhance with concept-specific metadata
            assessment["assessment_type"] = "concept_focused"
            assessment["concept"] = concept
            assessment["assessment_depth"] = assessment_depth
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error generating concept assessment: {str(e)}")
            raise
    
    async def generate_progress_quiz(
        self,
        user_progress: Dict[str, Any],
        framework_name: str,
        weak_areas: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a quiz based on user's learning progress and weak areas.
        
        Parameters:
        ----------
        user_progress : Dict[str, Any]
            User's learning progress data
        framework_name : str
            Target framework
        weak_areas : List[str], optional
            Areas where user needs improvement
            
        Returns:
        -------
        Dict[str, Any]
            Progress-based adaptive quiz
        """
        try:
            # Analyze progress to determine focus areas
            focus_areas = weak_areas or self._identify_focus_areas(user_progress)
            learning_level = user_progress.get("current_level", "beginner")
            
            # Weight questions towards weak areas (60% weak areas, 40% general)
            quiz_length = 8
            weak_area_questions = int(quiz_length * 0.6)
            general_questions = quiz_length - weak_area_questions
            
            all_concepts = []
            
            # Add weak area concepts (repeated for higher probability)
            for area in focus_areas:
                all_concepts.extend([area] * 2)
            
            # Add general concepts from progress
            completed_concepts = user_progress.get("completed_concepts", [])
            all_concepts.extend(completed_concepts)
            
            # Generate adaptive quiz
            progress_quiz = await self.generate_adaptive_quiz(
                topic="Progress Assessment",
                framework_name=framework_name,
                learning_level=learning_level,
                concepts_covered=all_concepts[:quiz_length],
                quiz_length=quiz_length,
                question_types=["multiple_choice", "conceptual", "code_completion"]
            )
            
            # Add progress-specific metadata
            progress_quiz["quiz_type"] = "progress_assessment"
            progress_quiz["focus_areas"] = focus_areas
            progress_quiz["user_level_at_generation"] = learning_level
            
            return progress_quiz
            
        except Exception as e:
            logger.error(f"Error generating progress quiz: {str(e)}")
            raise
    
    async def _generate_question(
        self,
        question_type: str,
        concept: str,
        topic: str,
        framework_name: str,
        learning_level: str,
        difficulty: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a single question of the specified type."""
        try:
            template = self.quiz_templates.get(question_type, self.quiz_templates["multiple_choice"])
            
            chain = template | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "concept": concept,
                "topic": topic,
                "framework_name": framework_name,
                "learning_level": learning_level,
                "complexity": difficulty["complexity"],
                "concept_depth": difficulty["concept_depth"]
            })
            
            # Add question metadata
            result["question_type"] = question_type
            result["concept_tested"] = concept
            result["difficulty_level"] = learning_level
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            # Return a fallback question
            return {
                "question_type": question_type,
                "concept_tested": concept,
                "difficulty_level": learning_level,
                "question": f"What is {concept} in {framework_name}?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option A",
                "explanation": "This is a fallback question due to generation error."
            }
    
    def _identify_focus_areas(self, user_progress: Dict[str, Any]) -> List[str]:
        """Identify areas that need focus based on user progress."""
        try:
            # Simple heuristic: areas with low scores or frequent mistakes
            weak_areas = []
            
            quiz_history = user_progress.get("quiz_history", [])
            for quiz in quiz_history:
                for question in quiz.get("questions", []):
                    if not question.get("correct", False):
                        weak_areas.append(question.get("concept_tested", ""))
            
            # Get most frequent weak areas
            from collections import Counter
            area_counts = Counter(weak_areas)
            return [area for area, count in area_counts.most_common(3)]
            
        except Exception as e:
            logger.error(f"Error identifying focus areas: {str(e)}")
            return ["basic_concepts"]
    
    def _calculate_question_points(self, question_type: str, learning_level: str) -> int:
        """Calculate points for a question based on type and difficulty."""
        base_points = {
            "multiple_choice": 2,
            "conceptual": 3,
            "code_completion": 4,
            "practical_scenario": 5
        }
        
        level_multiplier = {
            "beginner": 1,
            "intermediate": 1.5,
            "advanced": 2
        }
        
        return int(base_points.get(question_type, 2) * level_multiplier.get(learning_level, 1))
    
    def _generate_grading_criteria(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate grading criteria for the quiz."""
        total_points = sum(q["points"] for q in questions)
        
        return {
            "total_points": total_points,
            "passing_score": int(total_points * 0.7),  # 70% to pass
            "excellent_score": int(total_points * 0.9),  # 90% for excellent
            "grading_scale": {
                "excellent": {"min_percentage": 90, "description": "Outstanding understanding"},
                "good": {"min_percentage": 80, "description": "Good grasp of concepts"},
                "satisfactory": {"min_percentage": 70, "description": "Meets requirements"},
                "needs_improvement": {"min_percentage": 0, "description": "Requires additional study"}
            }
        }
    
    def _generate_feedback_templates(self, learning_level: str) -> Dict[str, str]:
        """Generate feedback templates for different performance levels."""
        return {
            "excellent": f"Excellent work! You demonstrate {learning_level}-level mastery of these concepts.",
            "good": f"Good job! You show solid understanding at the {learning_level} level with minor gaps.",
            "satisfactory": f"You meet the {learning_level} requirements but could benefit from review.",
            "needs_improvement": f"Consider reviewing the {learning_level} fundamentals before proceeding."
        }
    
    def _get_multiple_choice_template(self) -> ChatPromptTemplate:
        """Get template for multiple choice questions."""
        return ChatPromptTemplate.from_template(
            """
            Generate a multiple choice question about "{concept}" in {framework_name}.
            
            Learning Level: {learning_level}
            Complexity: {complexity}/3
            Concept Depth: {concept_depth}
            
            Create a clear, well-structured multiple choice question with 4 options.
            
            Format as JSON:
            {{
                "question": "Clear question text",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option X",
                "explanation": "Why this answer is correct",
                "distractors_explanation": "Why other options are incorrect",
                "concept_focus": "Specific aspect of the concept being tested"
            }}
            """
        )
    
    def _get_code_completion_template(self) -> ChatPromptTemplate:
        """Get template for code completion questions."""
        return ChatPromptTemplate.from_template(
            """
            Generate a code completion question about "{concept}" in {framework_name}.
            
            Learning Level: {learning_level}
            Complexity: {complexity}/3
            
            Provide partial code and ask the user to complete it.
            
            Format as JSON:
            {{
                "question": "Complete the following code",
                "incomplete_code": "Partial code with blanks or missing parts",
                "correct_completion": "The correct code completion",
                "explanation": "Explanation of the solution",
                "common_mistakes": ["Common mistake 1", "Common mistake 2"],
                "context": "What this code accomplishes"
            }}
            """
        )
    
    def _get_conceptual_template(self) -> ChatPromptTemplate:
        """Get template for conceptual questions."""
        return ChatPromptTemplate.from_template(
            """
            Generate a conceptual question about "{concept}" in {framework_name}.
            
            Learning Level: {learning_level}
            Concept Depth: {concept_depth}
            
            Create an open-ended question that tests understanding.
            
            Format as JSON:
            {{
                "question": "Conceptual question text",
                "key_points": ["Key point 1", "Key point 2"],
                "sample_answer": "A good example answer",
                "evaluation_criteria": ["Criterion 1", "Criterion 2"],
                "depth_expected": "Expected depth of answer for this level"
            }}
            """
        )
    
    def _get_practical_scenario_template(self) -> ChatPromptTemplate:
        """Get template for practical scenario questions."""
        return ChatPromptTemplate.from_template(
            """
            Generate a practical scenario question about "{concept}" in {framework_name}.
            
            Learning Level: {learning_level}
            Complexity: {complexity}/3
            
            Present a real-world scenario and ask how to apply the concept.
            
            Format as JSON:
            {{
                "scenario": "Real-world situation description",
                "question": "What would you do in this scenario?",
                "ideal_approach": "Best approach to solve this",
                "alternative_approaches": ["Alternative 1", "Alternative 2"],
                "evaluation_points": ["What to look for in the answer"],
                "common_pitfalls": ["Pitfall 1", "Pitfall 2"]
            }}
            """
        ) 