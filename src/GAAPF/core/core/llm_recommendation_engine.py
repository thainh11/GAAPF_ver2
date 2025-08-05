"""LLM-Based Recommendation Engine for GAAPF Architecture

This module provides intelligent recommendation capabilities that:
1. Suggests optimal constellation types based on learning context
2. Recommends learning paths and activities using AI intelligence
3. Provides personalized guidance beyond rule-based logic
4. Adapts recommendations based on user performance and preferences

This replaces fixed rule-based recommendation logic with dynamic LLM intelligence.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
import statistics
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMRecommendationEngine:
    """
    LLM-powered intelligent recommendation engine.
    
    This class uses an LLM to provide intelligent recommendations for:
    - Constellation type selection
    - Learning activity suggestions
    - Personalized learning paths
    - Performance optimization strategies
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        recommendation_cache_ttl: int = 3600,  # 1 hour
        max_recommendations: int = 5,
        is_logging: bool = False
    ):
        """
        Initialize the LLM recommendation engine.
        
        Parameters:
        ----------
        llm : BaseLanguageModel
            Language model to use for recommendations
        recommendation_cache_ttl : int, optional
            Cache time-to-live in seconds
        max_recommendations : int, optional
            Maximum number of recommendations to return
        is_logging : bool, optional
            Flag to enable detailed logging
        """
        self.llm = llm
        self.recommendation_cache_ttl = recommendation_cache_ttl
        self.max_recommendations = max_recommendations
        self.is_logging = is_logging
        
        # Caching and tracking
        self.recommendation_cache = {}
        self.user_interaction_history = {}
        self.recommendation_feedback = {}
        
        # Available constellation types (fallback reference)
        self.available_constellations = [
            "learning", "practice", "assessment", "project", "troubleshooting"
        ]
        
        # Learning activity categories
        self.activity_categories = {
            "exploration": ["introduction", "concept_overview", "guided_discovery"],
            "practice": ["exercises", "simulations", "hands_on_practice"],
            "assessment": ["quiz", "test", "evaluation", "self_assessment"],
            "application": ["project", "case_study", "real_world_application"],
            "review": ["summary", "recap", "reinforcement", "consolidation"]
        }
        
        # Performance improvement strategies
        self.improvement_strategies = {
            "struggling": [
                "break_down_concepts", "increase_practice", "provide_scaffolding",
                "use_analogies", "multi_modal_learning", "peer_support"
            ],
            "average": [
                "challenge_extension", "deeper_exploration", "cross_connections",
                "application_focus", "collaborative_learning"
            ],
            "advanced": [
                "independent_projects", "mentoring_others", "advanced_topics",
                "research_activities", "creative_applications"
            ]
        }
        
        if self.is_logging:
            logger.info("✅ LLM Recommendation Engine initialized")
    
    def recommend_constellation_type(
        self,
        user_id: str,
        learning_context: Dict,
        performance_data: Optional[Dict] = None,
        user_preferences: Optional[Dict] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Recommend optimal constellation type using LLM intelligence.
        
        Parameters:
        ----------
        user_id : str
            Unique identifier for the user
        learning_context : Dict
            Current learning context and session information
        performance_data : Dict, optional
            Recent performance metrics
        user_preferences : Dict, optional
            User preferences and learning style
        force_refresh : bool, optional
            Force new recommendation even if cached
            
        Returns:
        -------
        Dict[str, Any]
            Constellation recommendation with reasoning
        """
        start_time = time.time()
        
        if self.is_logging:
            logger.info(f"🎯 Generating constellation recommendation for user: {user_id}")
        
        # Check cache first
        cache_key = self._generate_cache_key("constellation", user_id, learning_context)
        if not force_refresh and cache_key in self.recommendation_cache:
            cached_result, timestamp = self.recommendation_cache[cache_key]
            if time.time() - timestamp < self.recommendation_cache_ttl:
                if self.is_logging:
                    logger.info("📋 Using cached constellation recommendation")
                return cached_result
        
        # Update interaction history
        self._update_interaction_history(user_id, "constellation_request", learning_context)
        
        try:
            # Generate recommendation using LLM
            recommendation = self._llm_recommend_constellation(
                user_id=user_id,
                learning_context=learning_context,
                performance_data=performance_data,
                user_preferences=user_preferences
            )
            
            if recommendation:
                # Validate and enhance recommendation
                validated_recommendation = self._validate_constellation_recommendation(recommendation)
                
                # Cache successful result
                self.recommendation_cache[cache_key] = (validated_recommendation, time.time())
                
                recommendation_time = time.time() - start_time
                if self.is_logging:
                    logger.info(f"✅ Constellation recommended successfully in {recommendation_time:.2f}s")
                
                return validated_recommendation
            else:
                raise ValueError("LLM returned empty recommendation")
                
        except Exception as e:
            if self.is_logging:
                logger.error(f"❌ Constellation recommendation failed: {e}")
            
            # Fallback to rule-based recommendation
            return self._get_fallback_constellation_recommendation(user_id, learning_context)
    
    def recommend_learning_activities(
        self,
        user_id: str,
        learning_context: Dict,
        constellation_type: str,
        performance_data: Optional[Dict] = None,
        session_goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Recommend specific learning activities for the current session.
        
        Parameters:
        ----------
        user_id : str
            User identifier
        learning_context : Dict
            Current learning context
        constellation_type : str
            Selected constellation type
        performance_data : Dict, optional
            Recent performance data
        session_goals : List[str], optional
            Specific goals for the current session
            
        Returns:
        -------
        Dict[str, Any]
            Activity recommendations with details
        """
        if self.is_logging:
            logger.info(f"🎯 Generating activity recommendations for {constellation_type} constellation")
        
        try:
            # Generate activity recommendations using LLM
            recommendations = self._llm_recommend_activities(
                user_id=user_id,
                learning_context=learning_context,
                constellation_type=constellation_type,
                performance_data=performance_data,
                session_goals=session_goals
            )
            
            if recommendations:
                return self._validate_activity_recommendations(recommendations)
            else:
                raise ValueError("LLM returned empty activity recommendations")
                
        except Exception as e:
            if self.is_logging:
                logger.error(f"❌ Activity recommendation failed: {e}")
            
            # Fallback to category-based recommendations
            return self._get_fallback_activity_recommendations(constellation_type, learning_context)
    
    def recommend_learning_path(
        self,
        user_id: str,
        learning_context: Dict,
        performance_history: Optional[List[Dict]] = None,
        learning_objectives: Optional[List[str]] = None,
        time_constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Recommend a personalized learning path.
        
        Parameters:
        ----------
        user_id : str
            User identifier
        learning_context : Dict
            Current learning context
        performance_history : List[Dict], optional
            Historical performance data
        learning_objectives : List[str], optional
            Specific learning objectives
        time_constraints : Dict, optional
            Time constraints and deadlines
            
        Returns:
        -------
        Dict[str, Any]
            Learning path recommendation with timeline
        """
        if self.is_logging:
            logger.info(f"🗺️ Generating learning path for user: {user_id}")
        
        try:
            # Generate learning path using LLM
            path_recommendation = self._llm_recommend_learning_path(
                user_id=user_id,
                learning_context=learning_context,
                performance_history=performance_history,
                learning_objectives=learning_objectives,
                time_constraints=time_constraints
            )
            
            if path_recommendation:
                return self._validate_learning_path_recommendation(path_recommendation)
            else:
                raise ValueError("LLM returned empty learning path")
                
        except Exception as e:
            if self.is_logging:
                logger.error(f"❌ Learning path recommendation failed: {e}")
            
            # Fallback to structured path
            return self._get_fallback_learning_path(learning_context, learning_objectives)
    
    def _llm_recommend_constellation(
        self,
        user_id: str,
        learning_context: Dict,
        performance_data: Optional[Dict],
        user_preferences: Optional[Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to recommend optimal constellation type.
        """
        # Prepare context for LLM
        context_analysis = self._prepare_constellation_context(
            user_id, learning_context, performance_data, user_preferences
        )
        
        prompt = f"""You are an expert educational AI specializing in personalized learning experiences.
Your task is to recommend the optimal constellation type for a learner based on comprehensive analysis.

**LEARNER ANALYSIS:**
{context_analysis}

**AVAILABLE CONSTELLATION TYPES:**

1. **Learning Constellation**
   - Purpose: Initial concept introduction and knowledge building
   - Best for: New topics, foundational concepts, exploration phase
   - Agents: Instructor, Tutor, Content Specialist, Progress Tracker
   - Focus: Understanding, comprehension, knowledge acquisition

2. **Practice Constellation**
   - Purpose: Skill development and knowledge reinforcement
   - Best for: Applying concepts, building fluency, skill practice
   - Agents: Practice Facilitator, Feedback Provider, Skill Assessor, Motivator
   - Focus: Application, repetition, skill building, confidence

3. **Assessment Constellation**
   - Purpose: Evaluation and progress measurement
   - Best for: Testing understanding, identifying gaps, certification
   - Agents: Assessor, Evaluator, Analytics Specialist, Report Generator
   - Focus: Measurement, evaluation, feedback, progress tracking

4. **Project Constellation**
   - Purpose: Real-world application and creative synthesis
   - Best for: Applying knowledge creatively, building portfolios
   - Agents: Project Guide, Mentor, Resource Coordinator, Reviewer
   - Focus: Creation, application, synthesis, real-world relevance

5. **Troubleshooting Constellation**
   - Purpose: Problem-solving and difficulty resolution
   - Best for: Addressing learning obstacles, debugging understanding
   - Agents: Problem Solver, Diagnostic Specialist, Support Coordinator, Adaptive Tutor
   - Focus: Problem resolution, gap identification, remediation

**RECOMMENDATION GUIDELINES:**

1. **Learning Stage Considerations:**
   - **Introduction/Exploration**: Learning Constellation
   - **Practice/Application**: Practice Constellation
   - **Evaluation/Testing**: Assessment Constellation
   - **Synthesis/Creation**: Project Constellation
   - **Difficulty/Confusion**: Troubleshooting Constellation

2. **Performance-Based Selection:**
   - **High Performance**: Project or Assessment Constellation
   - **Average Performance**: Practice or Learning Constellation
   - **Low Performance**: Troubleshooting or Learning Constellation
   - **Inconsistent Performance**: Assessment then Troubleshooting

3. **Context Factors:**
   - **New Module**: Learning Constellation
   - **Skill Building**: Practice Constellation
   - **Before Exam**: Assessment Constellation
   - **Capstone Work**: Project Constellation
   - **Struggling**: Troubleshooting Constellation

4. **User Preferences:**
   - **Hands-on Learners**: Practice or Project Constellation
   - **Theory-focused**: Learning or Assessment Constellation
   - **Goal-oriented**: Assessment or Project Constellation
   - **Support-seeking**: Troubleshooting or Learning Constellation

**YOUR TASK:**
Analyze all provided information and recommend the most appropriate constellation type that will:
- Best support the learner's current needs
- Align with their learning stage and objectives
- Consider their performance and preferences
- Optimize learning effectiveness

**RESPONSE FORMAT:**
Respond with a JSON object containing your recommendation:

{{
  "primary_recommendation": "<constellation_type>",
  "confidence": <0.0-1.0>,
  "reasoning": "<Detailed explanation of why this constellation is optimal>",
  "alternative_options": [
    {{
      "constellation_type": "<alternative_type>",
      "scenario": "<When this alternative would be better>",
      "confidence": <0.0-1.0>
    }}
  ],
  "session_focus": "<Primary focus for the upcoming session>",
  "expected_outcomes": ["<outcome1>", "<outcome2>", "<outcome3>"],
  "duration_estimate": "<Estimated session duration>",
  "preparation_needed": "<Any preparation the learner should do>"
}}

**IMPORTANT:**
- Choose the constellation type that best matches current learning needs
- Provide clear, actionable reasoning
- Consider both immediate and long-term learning goals
- Ensure recommendations are practical and achievable
"""
        
        try:
            messages = [
                SystemMessage(
                    content="You are an expert educational AI. Respond only with valid JSON containing constellation recommendations."
                ),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # Extract JSON from response
            recommendation = self._extract_json_from_response(content)
            
            if recommendation and self._validate_constellation_structure(recommendation):
                return recommendation
            else:
                raise ValueError("Invalid constellation recommendation structure")
                
        except Exception as e:
            if self.is_logging:
                logger.error(f"LLM constellation recommendation failed: {e}")
            return None
    
    def _llm_recommend_activities(
        self,
        user_id: str,
        learning_context: Dict,
        constellation_type: str,
        performance_data: Optional[Dict],
        session_goals: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to recommend specific learning activities.
        """
        # Prepare context for activity recommendations
        context_info = self._prepare_activity_context(
            user_id, learning_context, constellation_type, performance_data, session_goals
        )
        
        prompt = f"""You are an expert learning designer specializing in personalized educational activities.
Your task is to recommend specific learning activities for the current session.

**SESSION CONTEXT:**
{context_info}

**ACTIVITY DESIGN PRINCIPLES:**

1. **Learning Constellation Activities:**
   - Concept introductions, guided explorations, interactive tutorials
   - Knowledge building exercises, conceptual frameworks
   - Scaffolded learning experiences, foundational skill development

2. **Practice Constellation Activities:**
   - Skill drills, application exercises, simulation practice
   - Progressive difficulty challenges, repetitive practice
   - Performance-based activities, fluency building

3. **Assessment Constellation Activities:**
   - Quizzes, tests, self-assessments, peer evaluations
   - Progress checks, competency demonstrations
   - Reflection activities, portfolio reviews

4. **Project Constellation Activities:**
   - Creative projects, real-world applications, portfolio building
   - Collaborative work, research activities, presentations
   - Synthesis exercises, capstone experiences

5. **Troubleshooting Constellation Activities:**
   - Diagnostic assessments, gap analysis, remediation exercises
   - Problem-solving activities, debugging practice
   - Support sessions, adaptive interventions

**ACTIVITY SELECTION CRITERIA:**

1. **Performance-Based Selection:**
   - **High Performance**: Challenge activities, extension work, leadership roles
   - **Average Performance**: Standard practice, skill building, application focus
   - **Low Performance**: Remediation, scaffolded support, confidence building

2. **Learning Style Adaptations:**
   - **Visual**: Diagrams, charts, visual organizers, multimedia
   - **Auditory**: Discussions, explanations, verbal processing
   - **Kinesthetic**: Hands-on activities, simulations, movement-based learning
   - **Reading/Writing**: Text-based exercises, note-taking, written reflection

3. **Engagement Factors:**
   - **Interactive**: Games, simulations, collaborative activities
   - **Personalized**: Choice-based activities, interest-driven projects
   - **Challenging**: Problem-solving, critical thinking, creative tasks
   - **Supportive**: Guided practice, peer support, scaffolded experiences

**YOUR TASK:**
Design a sequence of 3-5 specific learning activities that will:
- Align with the constellation type and session goals
- Match the learner's performance level and preferences
- Provide appropriate challenge and support
- Build toward meaningful learning outcomes

**RESPONSE FORMAT:**
Respond with a JSON object containing activity recommendations:

{{
  "recommended_activities": [
    {{
      "activity_name": "<Descriptive activity name>",
      "activity_type": "<Type: exploration, practice, assessment, application, review>",
      "description": "<Detailed activity description>",
      "duration": "<Estimated time in minutes>",
      "difficulty_level": "<easy, moderate, challenging>",
      "learning_objectives": ["<objective1>", "<objective2>"],
      "materials_needed": ["<material1>", "<material2>"],
      "interaction_style": "<individual, collaborative, guided>",
      "success_criteria": "<How success will be measured>"
    }}
  ],
  "session_flow": "<How activities connect and build upon each other>",
  "total_duration": "<Total estimated session time>",
  "preparation_time": "<Time needed for setup>",
  "follow_up_suggestions": ["<suggestion1>", "<suggestion2>"],
  "adaptation_notes": "<How to modify activities based on learner response>"
}}

**IMPORTANT:**
- Design activities that are specific and actionable
- Ensure logical progression and skill building
- Consider practical constraints and resources
- Provide clear success criteria and assessment methods
"""
        
        try:
            messages = [
                SystemMessage(
                    content="You are an expert learning designer. Respond only with valid JSON containing activity recommendations."
                ),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # Extract JSON from response
            recommendations = self._extract_json_from_response(content)
            
            if recommendations and self._validate_activity_structure(recommendations):
                return recommendations
            else:
                raise ValueError("Invalid activity recommendation structure")
                
        except Exception as e:
            if self.is_logging:
                logger.error(f"LLM activity recommendation failed: {e}")
            return None
    
    def _llm_recommend_learning_path(
        self,
        user_id: str,
        learning_context: Dict,
        performance_history: Optional[List[Dict]],
        learning_objectives: Optional[List[str]],
        time_constraints: Optional[Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to recommend a comprehensive learning path.
        """
        # Prepare context for learning path recommendation
        path_context = self._prepare_learning_path_context(
            user_id, learning_context, performance_history, learning_objectives, time_constraints
        )
        
        prompt = f"""You are an expert educational strategist specializing in personalized learning path design.
Your task is to create a comprehensive, adaptive learning path for a specific learner.

**LEARNER PROFILE:**
{path_context}

**LEARNING PATH DESIGN PRINCIPLES:**

1. **Progressive Skill Building:**
   - Start with foundational concepts
   - Build complexity gradually
   - Ensure prerequisite mastery
   - Provide scaffolding and support

2. **Adaptive Pacing:**
   - Adjust based on performance
   - Allow for acceleration or remediation
   - Include checkpoints and assessments
   - Provide flexible timelines

3. **Engagement and Motivation:**
   - Include variety in activities and formats
   - Provide choice and autonomy
   - Celebrate milestones and achievements
   - Connect to real-world applications

4. **Comprehensive Coverage:**
   - Address all learning objectives
   - Include multiple learning modalities
   - Provide practice and application opportunities
   - Ensure knowledge transfer and retention

**PATH STRUCTURE COMPONENTS:**

1. **Learning Modules:**
   - Discrete learning units with clear objectives
   - Estimated duration and difficulty
   - Prerequisites and dependencies
   - Assessment and evaluation methods

2. **Constellation Sequences:**
   - Recommended constellation types for each module
   - Rationale for constellation selection
   - Transition strategies between constellations
   - Adaptation triggers and alternatives

3. **Milestone Checkpoints:**
   - Progress assessment points
   - Competency demonstrations
   - Path adjustment opportunities
   - Celebration and motivation moments

4. **Support and Resources:**
   - Additional learning materials
   - Help and support options
   - Peer collaboration opportunities
   - Expert guidance availability

**YOUR TASK:**
Design a comprehensive learning path that:
- Addresses all specified learning objectives
- Adapts to the learner's profile and constraints
- Provides clear progression and milestones
- Includes appropriate support and resources
- Optimizes for engagement and effectiveness

**RESPONSE FORMAT:**
Respond with a JSON object containing the learning path:

{{
  "learning_path": {{
    "path_name": "<Descriptive name for the learning path>",
    "total_duration": "<Estimated total time>",
    "difficulty_progression": "<How difficulty increases over time>",
    "learning_modules": [
      {{
        "module_id": "<unique_identifier>",
        "module_name": "<Descriptive module name>",
        "learning_objectives": ["<objective1>", "<objective2>"],
        "estimated_duration": "<Time estimate>",
        "difficulty_level": "<beginner, intermediate, advanced>",
        "recommended_constellation": "<constellation_type>",
        "key_activities": ["<activity1>", "<activity2>"],
        "prerequisites": ["<prerequisite1>", "<prerequisite2>"],
        "assessment_method": "<How progress will be measured>",
        "success_criteria": "<What constitutes successful completion>"
      }}
    ],
    "milestone_checkpoints": [
      {{
        "checkpoint_name": "<Milestone name>",
        "module_dependencies": ["<module_id1>", "<module_id2>"],
        "assessment_type": "<Type of assessment>",
        "success_threshold": "<Required performance level>",
        "adaptation_triggers": ["<trigger1>", "<trigger2>"]
      }}
    ],
    "support_resources": [
      {{
        "resource_type": "<Type of resource>",
        "description": "<Resource description>",
        "availability": "<When/how to access>",
        "usage_guidance": "<How to use effectively>"
      }}
    ],
    "adaptation_strategies": [
      {{
        "trigger_condition": "<What triggers adaptation>",
        "adaptation_action": "<How to adapt the path>",
        "alternative_options": ["<option1>", "<option2>"]
      }}
    ]
  }},
  "personalization_notes": "<How this path is tailored to the specific learner>",
  "success_indicators": ["<indicator1>", "<indicator2>"],
  "risk_mitigation": ["<risk1_mitigation>", "<risk2_mitigation>"]
}}

**IMPORTANT:**
- Design a path that is both comprehensive and achievable
- Ensure logical progression and skill building
- Include appropriate flexibility and adaptation
- Consider practical constraints and resources
- Provide clear guidance for implementation
"""
        
        try:
            messages = [
                SystemMessage(
                    content="You are an expert educational strategist. Respond only with valid JSON containing learning path recommendations."
                ),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # Extract JSON from response
            path_recommendation = self._extract_json_from_response(content)
            
            if path_recommendation and self._validate_learning_path_structure(path_recommendation):
                return path_recommendation
            else:
                raise ValueError("Invalid learning path recommendation structure")
                
        except Exception as e:
            if self.is_logging:
                logger.error(f"LLM learning path recommendation failed: {e}")
            return None
    
    def _prepare_constellation_context(
        self,
        user_id: str,
        learning_context: Dict,
        performance_data: Optional[Dict],
        user_preferences: Optional[Dict]
    ) -> str:
        """
        Prepare comprehensive context for constellation recommendation.
        """
        context_parts = []
        
        # Current learning context
        context_parts.append("**Current Learning Context:**")
        current_module = learning_context.get("current_module", "unknown")
        current_activity = learning_context.get("current_activity", "unknown")
        learning_stage = learning_context.get("learning_stage", "unknown")
        session_count = learning_context.get("session_count", 0)
        
        context_parts.append(f"- Module: {current_module}")
        context_parts.append(f"- Activity: {current_activity}")
        context_parts.append(f"- Learning Stage: {learning_stage}")
        context_parts.append(f"- Session Count: {session_count}")
        
        # User profile and preferences
        if user_preferences:
            context_parts.append(f"\n**User Preferences:**")
            learning_style = user_preferences.get("learning_style", "adaptive")
            pace_preference = user_preferences.get("pace_preference", "moderate")
            interaction_preference = user_preferences.get("interaction_preference", "balanced")
            
            context_parts.append(f"- Learning Style: {learning_style}")
            context_parts.append(f"- Pace Preference: {pace_preference}")
            context_parts.append(f"- Interaction Preference: {interaction_preference}")
        
        # Performance data
        if performance_data:
            context_parts.append(f"\n**Recent Performance:**")
            success_rate = performance_data.get("success_rate", 0.0)
            engagement_level = performance_data.get("engagement_level", 0.0)
            difficulty_comfort = performance_data.get("difficulty_comfort", 0.5)
            
            context_parts.append(f"- Success Rate: {success_rate:.1%}")
            context_parts.append(f"- Engagement Level: {engagement_level:.1%}")
            context_parts.append(f"- Difficulty Comfort: {difficulty_comfort:.1f}/1.0")
        
        # Interaction history
        if user_id in self.user_interaction_history:
            history = self.user_interaction_history[user_id]
            recent_interactions = history[-5:] if len(history) >= 5 else history
            
            context_parts.append(f"\n**Recent Interaction History:**")
            for interaction in recent_interactions:
                interaction_type = interaction.get("type", "unknown")
                timestamp = interaction.get("timestamp", 0)
                context_parts.append(f"- {interaction_type} ({self._format_timestamp(timestamp)})")
        
        # Learning objectives
        objectives = learning_context.get("learning_objectives", [])
        if objectives:
            context_parts.append(f"\n**Learning Objectives:**")
            for obj in objectives[:3]:  # Limit to top 3
                context_parts.append(f"- {obj}")
        
        return "\n".join(context_parts)
    
    def _prepare_activity_context(
        self,
        user_id: str,
        learning_context: Dict,
        constellation_type: str,
        performance_data: Optional[Dict],
        session_goals: Optional[List[str]]
    ) -> str:
        """
        Prepare context for activity recommendations.
        """
        context_parts = []
        
        # Constellation and session info
        context_parts.append(f"**Constellation Type:** {constellation_type}")
        context_parts.append(f"**Current Module:** {learning_context.get('current_module', 'unknown')}")
        context_parts.append(f"**Learning Stage:** {learning_context.get('learning_stage', 'unknown')}")
        
        # Session goals
        if session_goals:
            context_parts.append(f"\n**Session Goals:**")
            for goal in session_goals:
                context_parts.append(f"- {goal}")
        
        # Performance context
        if performance_data:
            context_parts.append(f"\n**Performance Context:**")
            success_rate = performance_data.get("success_rate", 0.0)
            areas_of_strength = performance_data.get("areas_of_strength", [])
            areas_for_improvement = performance_data.get("areas_for_improvement", [])
            
            context_parts.append(f"- Recent Success Rate: {success_rate:.1%}")
            if areas_of_strength:
                context_parts.append(f"- Strengths: {', '.join(areas_of_strength[:3])}")
            if areas_for_improvement:
                context_parts.append(f"- Areas for Improvement: {', '.join(areas_for_improvement[:3])}")
        
        # Time constraints
        available_time = learning_context.get("available_time", 60)  # Default 60 minutes
        context_parts.append(f"\n**Available Time:** {available_time} minutes")
        
        return "\n".join(context_parts)
    
    def _prepare_learning_path_context(
        self,
        user_id: str,
        learning_context: Dict,
        performance_history: Optional[List[Dict]],
        learning_objectives: Optional[List[str]],
        time_constraints: Optional[Dict]
    ) -> str:
        """
        Prepare context for learning path recommendations.
        """
        context_parts = []
        
        # Learning objectives
        if learning_objectives:
            context_parts.append("**Learning Objectives:**")
            for i, obj in enumerate(learning_objectives, 1):
                context_parts.append(f"{i}. {obj}")
        
        # Current context
        context_parts.append(f"\n**Current Learning Context:**")
        current_level = learning_context.get("current_level", "beginner")
        subject_area = learning_context.get("subject_area", "general")
        prior_knowledge = learning_context.get("prior_knowledge", "basic")
        
        context_parts.append(f"- Current Level: {current_level}")
        context_parts.append(f"- Subject Area: {subject_area}")
        context_parts.append(f"- Prior Knowledge: {prior_knowledge}")
        
        # Time constraints
        if time_constraints:
            context_parts.append(f"\n**Time Constraints:**")
            total_time = time_constraints.get("total_available_time", "flexible")
            sessions_per_week = time_constraints.get("sessions_per_week", "flexible")
            deadline = time_constraints.get("deadline", "none")
            
            context_parts.append(f"- Total Available Time: {total_time}")
            context_parts.append(f"- Sessions per Week: {sessions_per_week}")
            context_parts.append(f"- Deadline: {deadline}")
        
        # Performance history analysis
        if performance_history and len(performance_history) > 0:
            context_parts.append(f"\n**Performance History Analysis:**")
            
            # Calculate trends
            recent_scores = [entry.get("score", 0.0) for entry in performance_history[-5:]]
            if recent_scores:
                avg_score = statistics.mean(recent_scores)
                trend = "improving" if recent_scores[-1] > recent_scores[0] else "stable/declining"
                context_parts.append(f"- Recent Average Score: {avg_score:.2f}")
                context_parts.append(f"- Performance Trend: {trend}")
            
            # Identify patterns
            strong_areas = set()
            weak_areas = set()
            for entry in performance_history[-10:]:
                if entry.get("score", 0.0) > 0.8:
                    strong_areas.add(entry.get("topic", "unknown"))
                elif entry.get("score", 0.0) < 0.6:
                    weak_areas.add(entry.get("topic", "unknown"))
            
            if strong_areas:
                context_parts.append(f"- Strong Areas: {', '.join(list(strong_areas)[:3])}")
            if weak_areas:
                context_parts.append(f"- Areas Needing Support: {', '.join(list(weak_areas)[:3])}")
        
        return "\n".join(context_parts)
    
    def _extract_json_from_response(self, content: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response using multiple patterns.
        """
        import re
        
        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _validate_constellation_structure(self, recommendation: Dict) -> bool:
        """
        Validate constellation recommendation structure.
        """
        required_fields = ["primary_recommendation", "confidence", "reasoning"]
        
        for field in required_fields:
            if field not in recommendation:
                return False
        
        # Validate constellation type
        constellation_type = recommendation["primary_recommendation"]
        if constellation_type not in self.available_constellations:
            return False
        
        # Validate confidence
        confidence = recommendation["confidence"]
        if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
            return False
        
        return True
    
    def _validate_activity_structure(self, recommendations: Dict) -> bool:
        """
        Validate activity recommendation structure.
        """
        if "recommended_activities" not in recommendations:
            return False
        
        activities = recommendations["recommended_activities"]
        if not isinstance(activities, list) or len(activities) == 0:
            return False
        
        required_activity_fields = ["activity_name", "activity_type", "description", "duration"]
        
        for activity in activities:
            for field in required_activity_fields:
                if field not in activity:
                    return False
        
        return True
    
    def _validate_learning_path_structure(self, path_recommendation: Dict) -> bool:
        """
        Validate learning path recommendation structure.
        """
        if "learning_path" not in path_recommendation:
            return False
        
        learning_path = path_recommendation["learning_path"]
        required_fields = ["path_name", "learning_modules"]
        
        for field in required_fields:
            if field not in learning_path:
                return False
        
        modules = learning_path["learning_modules"]
        if not isinstance(modules, list) or len(modules) == 0:
            return False
        
        required_module_fields = ["module_id", "module_name", "learning_objectives"]
        
        for module in modules:
            for field in required_module_fields:
                if field not in module:
                    return False
        
        return True
    
    def _validate_constellation_recommendation(self, recommendation: Dict) -> Dict[str, Any]:
        """
        Validate and enhance constellation recommendation.
        """
        # Ensure required fields
        validated = {
            "primary_recommendation": recommendation.get("primary_recommendation", "learning"),
            "confidence": max(0.0, min(1.0, recommendation.get("confidence", 0.5))),
            "reasoning": recommendation.get("reasoning", "Default recommendation based on context"),
            "timestamp": time.time(),
            "recommendation_id": f"const_{int(time.time())}"
        }
        
        # Add optional fields if present
        optional_fields = [
            "alternative_options", "session_focus", "expected_outcomes",
            "duration_estimate", "preparation_needed"
        ]
        
        for field in optional_fields:
            if field in recommendation:
                validated[field] = recommendation[field]
        
        return validated
    
    def _validate_activity_recommendations(self, recommendations: Dict) -> Dict[str, Any]:
        """
        Validate and enhance activity recommendations.
        """
        validated = {
            "recommended_activities": recommendations.get("recommended_activities", []),
            "timestamp": time.time(),
            "recommendation_id": f"act_{int(time.time())}"
        }
        
        # Add optional fields
        optional_fields = [
            "session_flow", "total_duration", "preparation_time",
            "follow_up_suggestions", "adaptation_notes"
        ]
        
        for field in optional_fields:
            if field in recommendations:
                validated[field] = recommendations[field]
        
        return validated
    
    def _validate_learning_path_recommendation(self, path_recommendation: Dict) -> Dict[str, Any]:
        """
        Validate and enhance learning path recommendation.
        """
        validated = {
            "learning_path": path_recommendation.get("learning_path", {}),
            "timestamp": time.time(),
            "recommendation_id": f"path_{int(time.time())}"
        }
        
        # Add optional fields
        optional_fields = [
            "personalization_notes", "success_indicators", "risk_mitigation"
        ]
        
        for field in optional_fields:
            if field in path_recommendation:
                validated[field] = path_recommendation[field]
        
        return validated
    
    def _get_fallback_constellation_recommendation(self, user_id: str, learning_context: Dict) -> Dict[str, Any]:
        """
        Provide fallback constellation recommendation using rule-based logic.
        """
        # Simple rule-based fallback
        learning_stage = learning_context.get("learning_stage", "exploration")
        current_activity = learning_context.get("current_activity", "")
        
        if "assessment" in current_activity.lower() or "test" in current_activity.lower():
            constellation = "assessment"
            reasoning = "Assessment constellation selected for evaluation activities"
        elif "practice" in current_activity.lower() or "exercise" in current_activity.lower():
            constellation = "practice"
            reasoning = "Practice constellation selected for skill building activities"
        elif "project" in current_activity.lower() or "application" in current_activity.lower():
            constellation = "project"
            reasoning = "Project constellation selected for application activities"
        elif learning_stage == "exploration" or "introduction" in current_activity.lower():
            constellation = "learning"
            reasoning = "Learning constellation selected for exploration and introduction"
        else:
            constellation = "learning"
            reasoning = "Default learning constellation selected"
        
        return {
            "primary_recommendation": constellation,
            "confidence": 0.6,
            "reasoning": reasoning,
            "timestamp": time.time(),
            "recommendation_id": f"fallback_{int(time.time())}",
            "fallback": True
        }
    
    def _get_fallback_activity_recommendations(self, constellation_type: str, learning_context: Dict) -> Dict[str, Any]:
        """
        Provide fallback activity recommendations based on constellation type.
        """
        activities_by_constellation = {
            "learning": [
                {
                    "activity_name": "Concept Introduction",
                    "activity_type": "exploration",
                    "description": "Interactive introduction to key concepts",
                    "duration": "15",
                    "difficulty_level": "moderate"
                },
                {
                    "activity_name": "Guided Practice",
                    "activity_type": "practice",
                    "description": "Scaffolded practice with immediate feedback",
                    "duration": "20",
                    "difficulty_level": "moderate"
                }
            ],
            "practice": [
                {
                    "activity_name": "Skill Drills",
                    "activity_type": "practice",
                    "description": "Repetitive practice for skill fluency",
                    "duration": "25",
                    "difficulty_level": "moderate"
                },
                {
                    "activity_name": "Application Exercises",
                    "activity_type": "application",
                    "description": "Apply skills in varied contexts",
                    "duration": "20",
                    "difficulty_level": "challenging"
                }
            ],
            "assessment": [
                {
                    "activity_name": "Knowledge Check",
                    "activity_type": "assessment",
                    "description": "Quick assessment of understanding",
                    "duration": "15",
                    "difficulty_level": "moderate"
                },
                {
                    "activity_name": "Performance Evaluation",
                    "activity_type": "assessment",
                    "description": "Comprehensive skill demonstration",
                    "duration": "30",
                    "difficulty_level": "challenging"
                }
            ],
            "project": [
                {
                    "activity_name": "Creative Project",
                    "activity_type": "application",
                    "description": "Design and create original work",
                    "duration": "45",
                    "difficulty_level": "challenging"
                }
            ],
            "troubleshooting": [
                {
                    "activity_name": "Diagnostic Assessment",
                    "activity_type": "assessment",
                    "description": "Identify knowledge gaps and misconceptions",
                    "duration": "20",
                    "difficulty_level": "moderate"
                },
                {
                    "activity_name": "Remediation Practice",
                    "activity_type": "practice",
                    "description": "Targeted practice for identified gaps",
                    "duration": "25",
                    "difficulty_level": "easy"
                }
            ]
        }
        
        activities = activities_by_constellation.get(constellation_type, activities_by_constellation["learning"])
        
        return {
            "recommended_activities": activities,
            "session_flow": "Activities progress from introduction to application",
            "total_duration": str(sum(int(act["duration"]) for act in activities)),
            "timestamp": time.time(),
            "recommendation_id": f"fallback_act_{int(time.time())}",
            "fallback": True
        }
    
    def _get_fallback_learning_path(self, learning_context: Dict, learning_objectives: Optional[List[str]]) -> Dict[str, Any]:
        """
        Provide fallback learning path based on basic structure.
        """
        modules = [
            {
                "module_id": "intro",
                "module_name": "Introduction and Foundations",
                "learning_objectives": learning_objectives[:2] if learning_objectives else ["Understand basics"],
                "estimated_duration": "2-3 hours",
                "difficulty_level": "beginner",
                "recommended_constellation": "learning"
            },
            {
                "module_id": "practice",
                "module_name": "Practice and Application",
                "learning_objectives": learning_objectives[2:4] if learning_objectives and len(learning_objectives) > 2 else ["Apply knowledge"],
                "estimated_duration": "3-4 hours",
                "difficulty_level": "intermediate",
                "recommended_constellation": "practice"
            },
            {
                "module_id": "assessment",
                "module_name": "Assessment and Evaluation",
                "learning_objectives": ["Demonstrate mastery"],
                "estimated_duration": "1-2 hours",
                "difficulty_level": "intermediate",
                "recommended_constellation": "assessment"
            }
        ]
        
        return {
            "learning_path": {
                "path_name": "Standard Learning Path",
                "total_duration": "6-9 hours",
                "learning_modules": modules
            },
            "timestamp": time.time(),
            "recommendation_id": f"fallback_path_{int(time.time())}",
            "fallback": True
        }
    
    def _update_interaction_history(self, user_id: str, interaction_type: str, context: Dict):
        """
        Update user interaction history.
        """
        if user_id not in self.user_interaction_history:
            self.user_interaction_history[user_id] = []
        
        interaction = {
            "type": interaction_type,
            "timestamp": time.time(),
            "context": context
        }
        
        self.user_interaction_history[user_id].append(interaction)
        
        # Limit history size
        if len(self.user_interaction_history[user_id]) > 50:
            self.user_interaction_history[user_id].pop(0)
    
    def _generate_cache_key(self, recommendation_type: str, user_id: str, context: Dict) -> str:
        """
        Generate cache key for recommendations.
        """
        key_components = [
            recommendation_type,
            user_id,
            context.get("current_module", ""),
            context.get("learning_stage", ""),
            context.get("current_activity", "")
        ]
        return hash(tuple(key_components))
    
    def _format_timestamp(self, timestamp: float) -> str:
        """
        Format timestamp for display.
        """
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%H:%M")
    
    def record_recommendation_feedback(
        self,
        recommendation_id: str,
        feedback_type: str,
        feedback_data: Dict
    ):
        """
        Record feedback on recommendations for improvement.
        
        Parameters:
        ----------
        recommendation_id : str
            ID of the recommendation
        feedback_type : str
            Type of feedback (positive, negative, neutral)
        feedback_data : Dict
            Detailed feedback information
        """
        if recommendation_id not in self.recommendation_feedback:
            self.recommendation_feedback[recommendation_id] = []
        
        feedback_entry = {
            "type": feedback_type,
            "data": feedback_data,
            "timestamp": time.time()
        }
        
        self.recommendation_feedback[recommendation_id].append(feedback_entry)
        
        if self.is_logging:
            logger.info(f"📝 Recorded {feedback_type} feedback for recommendation {recommendation_id}")
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about recommendation performance.
        
        Returns:
        -------
        Dict[str, Any]
            Recommendation statistics and metrics
        """
        stats = {
            "total_recommendations": len(self.recommendation_cache),
            "total_users": len(self.user_interaction_history),
            "feedback_count": len(self.recommendation_feedback),
            "cache_utilization": 0.0,
            "recommendation_types": {}
        }
        
        # Analyze recommendation types
        for cache_key, (recommendation, timestamp) in self.recommendation_cache.items():
            rec_type = recommendation.get("primary_recommendation", "unknown")
            if rec_type not in stats["recommendation_types"]:
                stats["recommendation_types"][rec_type] = 0
            stats["recommendation_types"][rec_type] += 1
        
        # Calculate cache utilization
        current_time = time.time()
        active_cache_entries = sum(
            1 for _, (_, timestamp) in self.recommendation_cache.items()
            if current_time - timestamp < self.recommendation_cache_ttl
        )
        
        if len(self.recommendation_cache) > 0:
            stats["cache_utilization"] = active_cache_entries / len(self.recommendation_cache)
        
        return stats


# Convenience functions for easy integration

def create_llm_recommendation_engine(
    llm: BaseLanguageModel,
    **kwargs
) -> LLMRecommendationEngine:
    """
    Create an LLM recommendation engine instance.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    **kwargs
        Additional arguments for the engine
        
    Returns:
    -------
    LLMRecommendationEngine
        Configured recommendation engine instance
    """
    return LLMRecommendationEngine(llm=llm, **kwargs)


def recommend_constellation_for_context(
    llm: BaseLanguageModel,
    user_id: str,
    learning_context: Dict,
    **kwargs
) -> Dict[str, Any]:
    """
    Get constellation recommendation for a specific context.
    
    Parameters:
    ----------
    llm : BaseLanguageModel
        Language model to use
    user_id : str
        User identifier
    learning_context : Dict
        Current learning context
    **kwargs
        Additional arguments for recommendation
        
    Returns:
    -------
    Dict[str, Any]
        Constellation recommendation
    """
    engine = LLMRecommendationEngine(llm=llm)
    
    return engine.recommend_constellation_type(
        user_id=user_id,
        learning_context=learning_context,
        **kwargs
    )