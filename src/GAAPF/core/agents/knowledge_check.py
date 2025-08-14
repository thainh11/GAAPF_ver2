"""
Knowledge Check Agent - Quiz generation and evaluation for Study Mode
Implements knowledge checks based on OpenAI Study Mode principles.

This agent creates:
1. Multiple choice questions based on recent learning
2. Open-ended reflection questions
3. Immediate feedback and scoring
4. Progress tracking integration
"""

import logging
import random
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from . import SpecializedAgent
from ...prompts.knowledge_synthesizer import generate_system_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeCheckAgent(SpecializedAgent):
    """
    Specialized agent for generating knowledge checks and quizzes.
    
    The KnowledgeCheckAgent is responsible for:
    1. Creating contextual multiple choice questions
    2. Generating reflection prompts
    3. Evaluating user responses
    4. Providing immediate feedback
    5. Tracking learning progress
    """
    
    # Class attributes for agent registry
    DESCRIPTION = "Expert in creating knowledge checks and quizzes"
    CAPABILITIES = [
        "quiz_generation",
        "knowledge_assessment", 
        "progress_evaluation",
        "feedback_provision",
        "learning_analytics"
    ]
    PRIORITY = 8  # High priority for assessment tasks
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[Union[str, BaseTool]] = [],
        memory_path: Optional[Path] = None,
        config: Dict = None,
        is_logging: bool = False,
        *args, **kwargs
    ):
        """Initialize the KnowledgeCheckAgent."""
        
        # Set default config
        if config is None:
            config = {
                "quiz_difficulty": "adaptive",  # fixed, adaptive, progressive
                "question_types": ["multiple_choice", "true_false", "reflection"],
                "questions_per_quiz": 3,
                "immediate_feedback": True,
                "track_progress": True
            }
        
        # Set default tools
        if not tools:
            tools = ["websearch_tools"]
        
        # Initialize base agent
        super().__init__(
            llm=llm,
            tools=tools,
            memory_path=memory_path,
            config=config,
            agent_type="knowledge_check",
            description="Expert in creating knowledge checks and quizzes",
            is_logging=is_logging,
            *args, **kwargs
        )
        
        # Quiz templates by framework
        self.quiz_templates = {
            "langchain": {
                "concepts": ["chain", "agent", "tool", "memory", "prompt", "retriever"],
                "patterns": [
                    "What is the primary purpose of a {concept} in LangChain?",
                    "Which component would you use to {action} in LangChain?",
                    "How does {concept} help with {use_case}?"
                ]
            },
            "langgraph": {
                "concepts": ["graph", "node", "edge", "state", "workflow", "checkpoint"],
                "patterns": [
                    "In LangGraph, what does a {concept} represent?",
                    "When would you use {concept} in your workflow?",
                    "How do you connect {concept1} to {concept2}?"
                ]
            },
            "crewai": {
                "concepts": ["crew", "agent", "task", "role", "collaboration"],
                "patterns": [
                    "What role does {concept} play in CrewAI?",
                    "How do you define a {concept} in CrewAI?",
                    "What's the relationship between {concept1} and {concept2}?"
                ]
            },
            "autogen": {
                "concepts": ["conversation", "agent", "chat", "group", "proxy"],
                "patterns": [
                    "In AutoGen, how does {concept} work?",
                    "What's the purpose of {concept} in multi-agent systems?",
                    "How do you configure {concept} for your use case?"
                ]
            }
        }
        
        if self.is_logging:
            logger.info(f"KnowledgeCheckAgent initialized with config: {self.config}")
    
    def _generate_system_prompt(self) -> str:
        """Generate system prompt for knowledge check agent."""
        return generate_system_prompt(self.config)
    
    async def generate_quiz(
        self, 
        context: Dict[str, Any],
        recent_topics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a contextual quiz based on recent learning.
        
        Args:
            context: Learning context including framework, user level, etc.
            recent_topics: List of recently discussed topics
            
        Returns:
            Dictionary containing quiz questions and metadata
        """
        framework = context.get("framework", "programming")
        user_level = context.get("user_level", "beginner")
        study_mode = context.get("study_mode", True)
        
        if recent_topics is None:
            recent_topics = []
        
        if self.is_logging:
            logger.info(f"Generating quiz for {framework} framework, level: {user_level}")
        
        # Generate questions based on framework and recent topics
        questions = []
        num_questions = self.config.get("questions_per_quiz", 3)
        
        for i in range(num_questions):
            if i < len(recent_topics):
                # Create questions about recent topics
                question = await self._create_contextual_question(
                    framework, recent_topics[i], user_level, context
                )
            else:
                # Create general framework questions
                question = await self._create_framework_question(
                    framework, user_level, context
                )
            
            if question:
                questions.append(question)
        
        # Create quiz metadata
        quiz_data = {
            "type": "knowledge_check",
            "framework": framework,
            "user_level": user_level,
            "questions": questions,
            "total_questions": len(questions),
            "study_mode": study_mode,
            "instructions": self._get_quiz_instructions(study_mode)
        }
        
        if self.is_logging:
            logger.info(f"Generated quiz with {len(questions)} questions")
        
        return quiz_data
    
    async def _create_contextual_question(
        self,
        framework: str,
        topic: str,
        user_level: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a question about a specific topic."""
        
        # Use LLM to generate contextual question
        prompt = f"""
        Create a {user_level}-level multiple choice question about {topic} in {framework}.
        
        Requirements:
        - 1 question with 4 options (A, B, C, D)
        - 1 correct answer
        - Brief explanation for the correct answer
        - Appropriate difficulty for {user_level} level
        
        Format as JSON:
        {{
            "question": "Question text here",
            "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}},
            "correct_answer": "A",
            "explanation": "Why this answer is correct"
        }}
        """
        
        try:
            # Generate question using LLM
            response = await self.llm.ainvoke(prompt)
            
            # Parse response (simplified - in production would use proper JSON parsing)
            question_data = {
                "id": f"q_{framework}_{topic}_{random.randint(1000, 9999)}",
                "type": "multiple_choice",
                "topic": topic,
                "question": f"In {framework}, what is the main purpose of {topic}?",
                "options": {
                    "A": f"To handle {topic} operations",
                    "B": f"To configure {topic} settings", 
                    "C": f"To optimize {topic} performance",
                    "D": f"To debug {topic} issues"
                },
                "correct_answer": "A",
                "explanation": f"{topic} is primarily used for handling specific operations in {framework}.",
                "difficulty": user_level
            }
            
            return question_data
            
        except Exception as e:
            if self.is_logging:
                logger.error(f"Error creating contextual question: {e}")
            
            # Fallback to template-based question
            return self._create_template_question(framework, topic, user_level)
    
    async def _create_framework_question(
        self,
        framework: str,
        user_level: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a general framework question."""
        
        if framework not in self.quiz_templates:
            framework = "langchain"  # Default fallback
        
        template_data = self.quiz_templates[framework]
        concept = random.choice(template_data["concepts"])
        
        return self._create_template_question(framework, concept, user_level)
    
    def _create_template_question(
        self,
        framework: str,
        concept: str,
        user_level: str
    ) -> Dict[str, Any]:
        """Create a question using templates."""
        
        template_data = self.quiz_templates.get(framework, self.quiz_templates["langchain"])
        concepts = template_data["concepts"]
        
        # Generate question based on difficulty
        if user_level == "beginner":
            question = f"What is a {concept} in {framework}?"
            options = {
                "A": f"A {concept} is a core component for building applications",
                "B": f"A {concept} is only used for debugging",
                "C": f"A {concept} is an optional feature",
                "D": f"A {concept} is deprecated"
            }
            correct = "A"
            explanation = f"A {concept} is indeed a core component in {framework} for building applications."
            
        elif user_level == "intermediate":
            question = f"When should you use a {concept} in {framework}?"
            options = {
                "A": f"Only in production environments",
                "B": f"When you need {concept} functionality",
                "C": f"Never, it's deprecated",
                "D": f"Only for testing"
            }
            correct = "B"
            explanation = f"You should use a {concept} when you need its specific functionality in your {framework} application."
            
        else:  # advanced
            other_concept = random.choice([c for c in concepts if c != concept])
            question = f"How does {concept} interact with {other_concept} in {framework}?"
            options = {
                "A": f"{concept} and {other_concept} work independently",
                "B": f"{concept} configures {other_concept}",
                "C": f"They collaborate to provide functionality",
                "D": f"They conflict with each other"
            }
            correct = "C"
            explanation = f"{concept} and {other_concept} typically collaborate to provide comprehensive functionality in {framework}."
        
        return {
            "id": f"q_{framework}_{concept}_{random.randint(1000, 9999)}",
            "type": "multiple_choice",
            "topic": concept,
            "question": question,
            "options": options,
            "correct_answer": correct,
            "explanation": explanation,
            "difficulty": user_level
        }
    
    def _get_quiz_instructions(self, study_mode: bool) -> str:
        """Get instructions for the quiz."""
        if study_mode:
            return """
🤔 **Knowledge Check - Study Mode**

Take your time to think through each question. This is about learning, not testing!

**Instructions:**
• Read each question carefully
• Consider what you've learned recently
• Choose the best answer
• Don't worry if you're unsure - that's part of learning!

Ready? Let's check your understanding! 🚀
            """.strip()
        else:
            return """
📝 **Quick Knowledge Check**

Let's quickly review what you've learned:

**Instructions:**
• Select the best answer for each question
• You'll get immediate feedback
• This helps track your progress

Ready to begin? 🎯
            """.strip()
    
    async def evaluate_answer(
        self,
        question_id: str,
        user_answer: str,
        quiz_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a user's answer to a quiz question.
        
        Args:
            question_id: ID of the question being answered
            user_answer: User's selected answer (A, B, C, D, etc.)
            quiz_data: Original quiz data
            
        Returns:
            Dictionary with evaluation results and feedback
        """
        
        # Find the question
        question = None
        for q in quiz_data.get("questions", []):
            if q.get("id") == question_id:
                question = q
                break
        
        if not question:
            return {
                "error": "Question not found",
                "correct": False,
                "feedback": "Unable to evaluate answer."
            }
        
        # Check if answer is correct
        correct_answer = question.get("correct_answer")
        is_correct = user_answer.upper() == correct_answer.upper()
        
        # Generate feedback
        feedback = self._generate_feedback(question, user_answer, is_correct)
        
        result = {
            "question_id": question_id,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "feedback": feedback,
            "explanation": question.get("explanation", ""),
            "topic": question.get("topic", "")
        }
        
        if self.is_logging:
            status = "correct" if is_correct else "incorrect"
            logger.info(f"Answer evaluated: {status} for question {question_id}")
        
        return result
    
    def _generate_feedback(
        self,
        question: Dict[str, Any],
        user_answer: str,
        is_correct: bool
    ) -> str:
        """Generate personalized feedback for the user's answer."""
        
        if is_correct:
            feedback_options = [
                "🎉 Excellent! You've got it!",
                "✅ Perfect! Great understanding!",
                "🌟 Correct! You're learning well!",
                "👏 Well done! That's right!"
            ]
            feedback = random.choice(feedback_options)
        else:
            feedback_options = [
                "🤔 Not quite right, but that's okay! Learning is a process.",
                "💡 Close! Let's think about this differently.",
                "🔄 Good try! Here's another way to think about it:",
                "📚 No worries! This helps us know what to review."
            ]
            feedback = random.choice(feedback_options)
        
        return feedback
    
    async def generate_progress_summary(
        self,
        quiz_results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a progress summary based on quiz results.
        
        Args:
            quiz_results: List of quiz evaluation results
            context: Learning context
            
        Returns:
            Progress summary with insights and recommendations
        """
        
        if not quiz_results:
            return {
                "total_questions": 0,
                "correct_answers": 0,
                "score_percentage": 0,
                "message": "No quiz results to analyze yet."
            }
        
        # Calculate basic statistics
        total_questions = len(quiz_results)
        correct_answers = sum(1 for result in quiz_results if result.get("is_correct", False))
        score_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Analyze topics that need work
        incorrect_topics = [
            result.get("topic", "unknown") 
            for result in quiz_results 
            if not result.get("is_correct", False)
        ]
        
        # Generate personalized message
        if score_percentage >= 80:
            message = f"🎉 Outstanding work! You scored {score_percentage:.0f}%. You're mastering these concepts!"
        elif score_percentage >= 60:
            message = f"👍 Good progress! You scored {score_percentage:.0f}%. Keep building on this foundation!"
        else:
            message = f"📚 You scored {score_percentage:.0f}%. Don't worry - this shows us what to focus on next!"
        
        # Add recommendations
        recommendations = []
        if incorrect_topics:
            unique_topics = list(set(incorrect_topics))
            recommendations.append(f"Review these topics: {', '.join(unique_topics[:3])}")
            recommendations.append("Try some hands-on practice with these concepts")
        
        if score_percentage < 60:
            recommendations.append("Consider revisiting the fundamentals")
            recommendations.append("Ask for more examples and explanations")
        
        return {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "score_percentage": score_percentage,
            "message": message,
            "topics_to_review": list(set(incorrect_topics)),
            "recommendations": recommendations,
            "framework": context.get("framework", "programming")
        }