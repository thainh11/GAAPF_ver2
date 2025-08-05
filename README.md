# GAAPF - Guidance AI Agent for Python Framework

🤖 **An Adaptive Multi-Agent Learning System for AI Framework Education**

GAAPF is a cutting-edge educational platform that uses the novel "Adaptive Learning Constellation" architecture to provide personalized, interactive learning experiences for Python AI frameworks. Built with LangChain, LangGraph, and advanced temporal optimization algorithms.

## 🌟 Key Features

### 🔗 Adaptive Learning Constellations
- **Dynamic Agent Networks**: Multi-agent systems that adapt in real-time based on user learning patterns
- **12 Specialized Agents**: Instructor, Code Assistant, Documentation Expert, Practice Facilitator, Assessment, Mentor, Research Assistant, Project Guide, Troubleshooter, Motivational Coach, Knowledge Synthesizer, Progress Tracker
- **Context-Aware Handoffs**: Intelligent agent coordination for seamless learning experiences
- **5 Constellation Types**: Knowledge Intensive, Hands-On Focused, Theory-Practice Balanced, Basic Learning, Guided Learning

### 📊 Temporal Learning Optimization
- **Effectiveness Tracking**: Continuous monitoring of learning outcomes and engagement
- **Pattern Recognition**: AI-powered analysis of optimal learning configurations
- **Personalized Recommendations**: Constellation selection based on individual learning patterns
- **Adaptive Engine**: Real-time learning path optimization

### 🎯 Comprehensive Framework Support
- **LangChain**: Complete learning path from basics to advanced agent systems
- **LangGraph**: Stateful multi-agent application development
- **Extensible Architecture**: Ready framework for adding CrewAI, AutoGen, LlamaIndex
- **Framework Information Collection**: Automated collection and curriculum generation for any supported framework

### 🚀 Modern Technology Stack
- **LangChain 0.3.x**: Latest LLM orchestration framework
- **LangGraph 0.4.x**: Advanced graph-based agent workflows
- **CLI Interface**: Real LLM integration with actual AI responses
- **Streamlit Demo**: Visual interface for system demonstration
- **Tavily Integration**: AI-powered search and documentation discovery for framework information collection
- **Pydantic 2.x**: Type-safe configuration and data validation
- **Modern Python**: Built for Python 3.10+

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │   CLI Interface │ │  Streamlit Web  │ │   FastAPI REST  │ │
│  │  (Real LLM)     │ │   (Demo Mode)   │ │   (Planned)     │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│              Learning Hub Core & Agent Management           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Adaptive Learning Constellation              │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐│ │
│  │  │Instructor │ │Code Assist│ │Doc Expert │ │Assessment ││ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘│ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐│ │
│  │  │ Mentor    │ │Practice   │ │Research   │ │Project    ││ │
│  │  │           │ │Facilitator│ │Assistant  │ │Guide      ││ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘│ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐│ │
│  │  │Troublesh. │ │Motiv.Coach│ │Knowledge  │ │Progress   ││ │
│  │  │           │ │           │ │Synthesizer│ │Tracker    ││ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘│ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│               Core Orchestration Layer                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │ Constellation   │ │ Temporal State  │ │ Intelligent     │ │
│  │ Manager         │ │ Manager         │ │ Agent Manager   │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │ Learning Hub    │ │ Analytics       │ │ Knowledge Graph │ │
│  │ Core            │ │ Engine          │ │ Manager         │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Tools & Integration Layer               │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │ Tavily Search   │ │   File Tools    │ │ Learning Tools  │ │
│  │ & Discovery     │ │ & Code Exec     │ │ & Assessment    │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    LLM Integration Layer                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │  Gemini 2.5     │ │   OpenAI GPT    │ │ Anthropic       │ │
│  │  Flash/Pro      │ │   3.5/4.0       │ │ Claude          │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                Memory & Storage Layer                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │ User Profiles   │ │Framework Configs│ │ Memory Systems  │ │
│  │ (JSON Files)    │ │ (JSON/Python)   │ │ (Multi-type)    │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │ Learning        │ │ Constellation   │ │ Temporal        │ │
│  │ Sessions        │ │ Memory          │ │ Patterns        │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 Agent Architecture - Adaptive Learning Constellation

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Constellation Architecture         │
└─────────────────────────────────────────────────────────────┘

                          ┌─────────────────────┐
                          │  ConstellationManager │
                          │  (LangGraph Orchestration) │
                          └──────────┬──────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     │               │               │
          ┌──────────▼─────────┐ ┌──▼──┐ ┌─────────▼─────────┐
          │  Knowledge Agents  │ │Base │ │   Practice Agents │
          └────────────────────┘ │Agent│ └───────────────────┘
          │ 📚 Instructor       │ │     │ │ ⚡ Code Assistant  │
          │ 📖 Doc Expert       │ │Core │ │ 🛠️ Practice Facilitator│
          │ 🔬 Research Assistant│ │     │ │ 🏗️ Project Guide   │
          │ 🧠 Knowledge Synth. │ │     │ │ 🔧 Troubleshooter  │
          └────────────────────┘ └──┬──┘ └───────────────────┘
                     │               │               │
          ┌──────────▼─────────┐    │    ┌─────────▼─────────┐
          │  Support Agents    │    │    │ Assessment Agents │
          └────────────────────┘    │    └───────────────────┘
          │ 🎯 Mentor          │    │    │ 📊 Assessment     │
          │ 💪 Motivational    │    │    │ 📈 Progress Tracker│
          │    Coach           │    │    └───────────────────┘
          └────────────────────┘    │
                                    │
                ┌───────────────────▼───────────────────┐
                │         Agent Communication           │
                │  ┌─────────────────────────────────┐  │
                │  │    Intelligent Handoff Logic   │  │
                │  │  • Content Analysis            │  │
                │  │  • Context Evaluation          │  │
                │  │  • Next Agent Suggestion       │  │
                │  │  • Confidence Scoring          │  │
                │  └─────────────────────────────────┘  │
                └───────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- At least one LLM API key:
  - **Google Gemini** (Recommended - free tier available)
  - **OpenAI GPT** (Pay per use)
  - **Anthropic Claude** (Pay per use)
- **Tavily API Key** (Optional - for enhanced search capabilities)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/gaapf-guidance-ai-agent.git
cd gaapf-guidance-ai-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp env.example .env
# Edit .env with your API keys
```

### ⚠️ IMPORTANT: Framework Initialization (REQUIRED)

**Before using GAAPF for learning, you MUST initialize the framework knowledge base:**

```bash
python setup_framework_knowledge.py
```

**What this does:**
- Initializes ALL supported frameworks (LangChain, LangGraph, CrewAI, AutoGen, Haystack, LlamaIndex)
- Stores framework information in vector database for semantic search
- Sets up caching and monitoring systems
- **Time required:** 10-30 minutes (one-time setup)

**📖 For detailed setup instructions, see:** [`SETUP_INSTRUCTIONS.md`](SETUP_INSTRUCTIONS.md)

### 4. **Run the CLI (Recommended)**
```bash
python run_optimized_cli.py
```

**Alternative interfaces:**

5a. **Launch the Streamlit interface** (Demo mode - mock responses)
```bash
streamlit run src/pyframeworks_assistant/interfaces/web/streamlit_app.py
```

### 🎯 CLI Interface (Real LLM Integration)

The CLI provides the full experience with **actual AI responses**:

- ✅ **Real LLM API calls** to Google Gemini, OpenAI GPT, or Anthropic Claude
- ✅ **Intelligent agent selection** based on your questions
- ✅ **Personalized learning paths** adapted to your skill level
- ✅ **Natural conversation** with specialized AI agents
- ✅ **Progress tracking** and temporal optimization
- ✅ **Advanced search** with Tavily integration for real-time framework discovery

**Quick CLI Demo:**
```bash
# Start the CLI
python run_cli.py

# Follow the interactive setup:
# 1. Profile creation (experience, skills, goals)
# 2. Framework selection (LangChain, LangGraph, etc.)
# 3. Start learning with real AI assistance!

# Example conversation:
You: What is LangChain and how do I get started?
🤖 Instructor: LangChain is a powerful framework for building applications with Large Language Models...

You: Show me a simple code example
🤖 Code Assistant: Here's a basic LangChain example to get you started:
```python
from langchain.llms import OpenAI
...
```
```

See [CLI_GUIDE.md](CLI_GUIDE.md) for complete CLI documentation.

### Configuration

Create a `.env` file with your API keys. The application will automatically detect and use available providers based on the keys you provide.

**.env Example:**
```env
# --- LLM Provider Priority (Optional) ---
# Comma-separated list of providers to try in order.
# Supported: together, google-genai, vertex-ai, openai
LLM_PROVIDER_PRIORITY=together,google-genai,vertex-ai,openai

# --- Provider API Keys (Choose at least one) ---

# Together AI (Recommended for high performance)
TOGETHER_API_KEY=your_together_api_key_here

# Google Gemini API
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Google Vertex AI (for GCP users)
# 1. Your Google Cloud project ID
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
# 2. Path to your service account credentials JSON file (optional, for non-default auth)
#    If not set, falls back to gcloud default credentials.
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-file.json

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# --- Other Services ---

# Search Enhancement (Optional)
TAVILY_API_KEY=your_tavily_api_key_here

# LangSmith Configuration (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=gaapf-guidance-ai-agent
```

## 🎯 Usage Examples

### Creating a User Profile

```python
from pyframeworks_assistant.config.user_profiles import (
    UserProfile, SkillLevel, LearningPace, LearningStyle
)

profile = UserProfile(
    user_id="user_123",
    programming_experience_years=3,
    python_skill_level=SkillLevel.INTERMEDIATE,
    learning_pace=LearningPace.MODERATE,
    preferred_learning_style=LearningStyle.HANDS_ON,
    learning_goals=["Learn LangChain", "Build RAG applications"]
)
```

### Using the Constellation System

```python
import asyncio
from pyframeworks_assistant.core.constellation import ConstellationManager
from pyframeworks_assistant.config.framework_configs import SupportedFrameworks
from pyframeworks_assistant.core.constellation_types import ConstellationType

async def learning_session():
    manager = ConstellationManager()
    
    # Create an adaptive constellation
    constellation = await manager.create_constellation(
        constellation_type=ConstellationType.HANDS_ON_FOCUSED,
        user_profile=profile,
        framework=SupportedFrameworks.LANGCHAIN,
        module_id="lc_basics",
        session_id="session_123"
    )
    
    # Run learning session
    result = await manager.run_session(
        session_id="session_123",
        user_message="I want to learn about LangChain chains",
        user_profile=profile,
        framework=SupportedFrameworks.LANGCHAIN,
        module_id="lc_basics"
    )
    
    return result

# Run the session
result = asyncio.run(learning_session())
```

### Temporal Optimization

```python
from pyframeworks_assistant.core.temporal_state import TemporalStateManager

temporal_manager = TemporalStateManager()

# Get optimal constellation for user
optimal_constellation, confidence = await temporal_manager.optimize_constellation_selection(
    user_profile=profile,
    framework=SupportedFrameworks.LANGCHAIN,
    module_id="lc_basics",
    session_context={}
)

print(f"Recommended: {optimal_constellation.value} (confidence: {confidence:.2f})")
```

## 📚 Framework Information Collection

The Framework Information Collection module automatically gathers comprehensive information about programming frameworks to provide personalized learning experiences:

- **Automated Information Collection**: Uses Tavily search and extraction to gather documentation, tutorials, API references, and examples
- **Concept Extraction**: Identifies key concepts and relationships within frameworks
- **Knowledge Graph Integration**: Builds a semantic network of framework concepts and relationships
- **Curriculum Generation**: Creates personalized learning paths based on user experience and goals
- **Background Processing**: Performs quick initialization during onboarding with more comprehensive collection in the background

### Supported Frameworks

- LangChain
- LangGraph  
- Microsoft AutoGen
- CrewAI
- Haystack by Deepset
- Hugging Face SmolAgents
- OpenAI Agents Python

### Example Usage

```python
from GAAPF.core.framework_onboarding import FrameworkOnboarding
from GAAPF.memory.long_term_memory import LongTermMemory
from GAAPF.core.knowledge_graph import KnowledgeGraph

# Initialize components
memory = LongTermMemory(user_id="user_123")
knowledge_graph = KnowledgeGraph()

# Initialize framework onboarding
onboarding = FrameworkOnboarding(
    memory=memory,
    knowledge_graph=knowledge_graph,
    tavily_api_key="your_tavily_api_key"
)

# Get user configuration
user_config = {
    "experience_level": "intermediate",
    "goals": ["build_production_app", "research"]
}

# Initialize framework and get curriculum
curriculum = await onboarding.initialize_framework(
    framework_name="LangChain",
    user_id="user_123",
    user_config=user_config
)

# Save curriculum
curriculum_path = onboarding.save_curriculum(curriculum, "user_123", "LangChain")
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- At least one LLM API key:
  - **Google Gemini** (Recommended - free tier available)
  - **OpenAI GPT** (Pay per use)
  - **Anthropic Claude** (Pay per use)
- **Tavily API Key** (Optional - for enhanced search capabilities)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/gaapf-guidance-ai-agent.git
cd gaapf-guidance-ai-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp env.example .env
# Edit .env with your API keys
```

4. **Run the CLI (Recommended)**
```bash
python run_cli.py
```

**Alternative interfaces:**

5a. **Launch the Streamlit interface** (Demo mode - mock responses)
```bash
streamlit run src/pyframeworks_assistant/interfaces/web/streamlit_app.py
```

### 🎯 CLI Interface (Real LLM Integration)

The CLI provides the full experience with **actual AI responses**:

- ✅ **Real LLM API calls** to Google Gemini, OpenAI GPT, or Anthropic Claude
- ✅ **Intelligent agent selection** based on your questions
- ✅ **Personalized learning paths** adapted to your skill level
- ✅ **Natural conversation** with specialized AI agents
- ✅ **Progress tracking** and temporal optimization
- ✅ **Advanced search** with Tavily integration for real-time framework discovery

**Quick CLI Demo:**
```bash
# Start the CLI
python run_cli.py

# Follow the interactive setup:
# 1. Profile creation (experience, skills, goals)
# 2. Framework selection (LangChain, LangGraph, etc.)
# 3. Start learning with real AI assistance!

# Example conversation:
You: What is LangChain and how do I get started?
🤖 Instructor: LangChain is a powerful framework for building applications with Large Language Models...

You: Show me a simple code example
🤖 Code Assistant: Here's a basic LangChain example to get you started:
```python
from langchain.llms import OpenAI
...
```
```

See [CLI_GUIDE.md](CLI_GUIDE.md) for complete CLI documentation.

### Configuration

Create a `.env` file with your API keys. The application will automatically detect and use available providers based on the keys you provide.

**.env Example:**
```env
# --- LLM Provider Priority (Optional) ---
# Comma-separated list of providers to try in order.
# Supported: together, google-genai, vertex-ai, openai
LLM_PROVIDER_PRIORITY=together,google-genai,vertex-ai,openai

# --- Provider API Keys (Choose at least one) ---

# Together AI (Recommended for high performance)
TOGETHER_API_KEY=your_together_api_key_here

# Google Gemini API
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Google Vertex AI (for GCP users)
# 1. Your Google Cloud project ID
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
# 2. Path to your service account credentials JSON file (optional, for non-default auth)
#    If not set, falls back to gcloud default credentials.
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-file.json

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# --- Other Services ---

# Search Enhancement (Optional)
TAVILY_API_KEY=your_tavily_api_key_here

# LangSmith Configuration (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=gaapf-guidance-ai-agent
```

## 🎯 Usage Examples

### Creating a User Profile

```python
from pyframeworks_assistant.config.user_profiles import (
    UserProfile, SkillLevel, LearningPace, LearningStyle
)

profile = UserProfile(
    user_id="user_123",
    programming_experience_years=3,
    python_skill_level=SkillLevel.INTERMEDIATE,
    learning_pace=LearningPace.MODERATE,
    preferred_learning_style=LearningStyle.HANDS_ON,
    learning_goals=["Learn LangChain", "Build RAG applications"]
)
```

### Using the Constellation System

```python
import asyncio
from pyframeworks_assistant.core.constellation import ConstellationManager
from pyframeworks_assistant.config.framework_configs import SupportedFrameworks
from pyframeworks_assistant.core.constellation_types import ConstellationType

async def learning_session():
    manager = ConstellationManager()
    
    # Create an adaptive constellation
    constellation = await manager.create_constellation(
        constellation_type=ConstellationType.HANDS_ON_FOCUSED,
        user_profile=profile,
        framework=SupportedFrameworks.LANGCHAIN,
        module_id="lc_basics",
        session_id="session_123"
    )
    
    # Run learning session
    result = await manager.run_session(
        session_id="session_123",
        user_message="I want to learn about LangChain chains",
        user_profile=profile,
        framework=SupportedFrameworks.LANGCHAIN,
        module_id="lc_basics"
    )
    
    return result

# Run the session
result = asyncio.run(learning_session())
```

### Temporal Optimization

```python
from pyframeworks_assistant.core.temporal_state import TemporalStateManager

temporal_manager = TemporalStateManager()

# Get optimal constellation for user
optimal_constellation, confidence = await temporal_manager.optimize_constellation_selection(
    user_profile=profile,
    framework=SupportedFrameworks.LANGCHAIN,
    module_id="lc_basics",
    session_context={}
)

print(f"Recommended: {optimal_constellation.value} (confidence: {confidence:.2f})")
```

## 📚 Framework Support

### Currently Supported

| Framework | Version | Status | Learning Features |
|-----------|---------|--------|------------------|
| LangChain | 0.3.25+ | ✅ Full Support | Complete curriculum with 12 specialized agents |
| LangGraph | 0.4.7+ | ✅ Full Support | Stateful multi-agent workflows and patterns |

### Extensible Architecture Ready For

| Framework | Status | Architecture Support |
|-----------|--------|---------------------|
| CrewAI | 🏗️ Architecture Ready | Agent definitions and constellation patterns prepared |
| AutoGen | 🏗️ Architecture Ready | Multi-agent framework integration planned |
| LlamaIndex | 🏗️ Architecture Ready | RAG-focused learning paths designed |

## 🔄 System Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     GAAPF Learning Workflow                    │
└─────────────────────────────────────────────────────────────────┘

1. USER ONBOARDING
   👤 User starts → Profile Creation → Framework Selection

2. CONSTELLATION FORMATION
   🔍 Analyze Context → 🎯 Select Optimal Agents → ⭐ Form Constellation

3. LEARNING SESSION
   💬 User Question → 🤖 Agent Processing → 🔄 Intelligent Handoffs
   
4. ADAPTIVE OPTIMIZATION  
   📊 Track Effectiveness → 🧠 Learn Patterns → 🎯 Optimize Future Sessions

5. PROGRESS SYNTHESIS
   📈 Update Progress → 💾 Save Session → 🚀 Plan Next Learning

┌─────────────────────────────────────────────────────────────────┐
│                    Detailed User Journey                       │
└─────────────────────────────────────────────────────────────────┘

START → User launches CLI
   ↓
PROFILE → Create/Load user profile
   ↓
FRAMEWORK → Choose: LangChain, LangGraph, etc.
   ↓
CONSTELLATION → System forms optimal agent team
   ↓
LEARNING LOOP:
   ├─ User asks question
   ├─ Primary agent responds (with real LLM)
   ├─ Handoff to specialist if needed
   ├─ Practice exercises generated (with file tools)
   ├─ Progress tracked and analyzed
   └─ Loop continues...
   ↓
ADAPTATION → System learns user patterns
   ↓
SYNTHESIS → Session summary & next steps
   ↓
END → Save progress & exit gracefully
```

## 🏛️ Constellation Types

GAAPF offers 5 specialized constellation types currently implemented:

1. **Knowledge Intensive** 📚
   - Focus: Theoretical understanding
   - Primary Agents: Instructor, Documentation Expert, Knowledge Synthesizer
   - Support Agents: Research Assistant, Progress Tracker
   - Best for: Conceptual learning, theoretical foundations

2. **Hands-On Focused** ⚡
   - Focus: Practical implementation
   - Primary Agents: Code Assistant, Practice Facilitator, Project Guide
   - Support Agents: Troubleshooter, Mentor
   - Best for: Learning by doing, practical skills

3. **Theory-Practice Balanced** ⚖️
   - Focus: Balanced approach
   - Primary Agents: Instructor, Code Assistant, Practice Facilitator
   - Support Agents: Documentation Expert, Mentor
   - Best for: Comprehensive understanding

4. **Basic Learning** 🌟
   - Focus: Gentle introduction
   - Primary Agents: Instructor, Code Assistant
   - Support Agents: Mentor, Practice Facilitator
   - Best for: Beginners, foundational learning

5. **Guided Learning** 🎯
   - Focus: Structured guidance
   - Primary Agents: Instructor, Mentor
   - Support Agents: Code Assistant, Practice Facilitator
   - Best for: Users needing extra support

Additional constellation types (Research Intensive, Quick Learning, Deep Exploration, Project Oriented, Assessment Focused) are planned for future releases.

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Basic functionality test
python test_basic.py

# Unit tests (when available)
pytest tests/unit/

# Integration tests (when available)
pytest tests/integration/
```

## 🛠️ Development

### Project Structure

```
gaapf-guidance-ai-agent/
├── src/
│   └── pyframeworks_assistant/
│       ├── config/           # User profiles and framework configurations
│       ├── core/             # Core constellation and learning systems
│       │   ├── constellation.py         # Main constellation manager
│       │   ├── temporal_state.py        # Temporal optimization
│       │   ├── learning_hub.py          # Central learning coordination
│       │   ├── knowledge_graph.py       # Concept relationships
│       │   └── analytics_engine.py      # Learning analytics
│       ├── agents/           # 12 individual agent implementations
│       ├── memory/           # Multiple memory systems
│       ├── tools/            # Tavily search, file tools, learning tools
│       └── interfaces/       # CLI and Streamlit interfaces
├── tests/                    # Test suite
├── user_profiles/           # User data storage
├── generated_code/          # Practice session files
├── requirements.txt         # Dependencies
└── pyproject.toml          # Project configuration
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_basic.py

# Run linting (if configured)
ruff check src/
black src/

# Type checking (if configured)
mypy src/
```

## 📊 Performance & Metrics

### Learning Effectiveness Metrics

- **Comprehension Score**: Understanding level measurement via LLM analysis
- **Engagement Score**: User interaction and interest tracking
- **Completion Rate**: Task and module completion tracking
- **Time Efficiency**: Learning speed optimization
- **Retention Estimate**: Knowledge retention prediction
- **Agent Handoff Efficiency**: Smooth transitions between specialists

### System Performance

- **Constellation Adaptation**: < 2 seconds response time
- **Pattern Recognition**: Real-time learning optimization
- **Memory Management**: Efficient session state handling
- **LLM Integration**: Multi-provider support with failover

## 🔧 Configuration Options

### User Profile Customization

```python
# Skill levels: none, beginner, intermediate, advanced, expert
# Learning paces: slow, moderate, fast, intensive
# Learning styles: visual, hands_on, theoretical, mixed
# Difficulty progression: gradual, moderate, aggressive
```

### System Configuration

```python
# Constellation settings
max_concurrent_agents = 16
constellation_timeout = 300  # seconds
role_morphing_enabled = True

# Temporal optimization
effectiveness_tracking_enabled = True
pattern_analysis_enabled = True
optimization_auto_apply = False

# Memory management
max_memory_sessions = 1000
memory_cleanup_interval = 3600  # seconds
```

## 🤝 Community & Support

- **Documentation**: [CLI_GUIDE.md](CLI_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/your-username/gaapf-guidance-ai-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/gaapf-guidance-ai-agent/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain Team** for the incredible LLM orchestration framework
- **LangGraph Team** for advanced graph-based agent capabilities  
- **Streamlit Team** for the beautiful web interface framework
- **Tavily Team** for AI-powered search and discovery capabilities
- **The AI Community** for continuous inspiration and collaboration

## 🔮 Roadmap

### Phase 1: Core System (Current) ✅
- ✅ Adaptive learning constellations with 12 specialized agents
- ✅ LangChain & LangGraph support with real curriculum
- ✅ Temporal optimization with pattern recognition
- ✅ CLI interface with real LLM integration
- ✅ Streamlit demo interface
- ✅ Tavily-powered search and discovery

### Phase 2: Enhanced Features (Q2 2025)
- 🔄 Additional framework support (CrewAI, AutoGen)
- 🔄 Advanced analytics dashboard
- 🔄 API-first architecture
- 🔄 Enhanced memory systems

### Phase 3: Advanced Learning (Q3 2025)
- 📋 Multi-language support
- 📋 Team collaboration features
- 📋 Advanced assessment tools
- 📋 Custom constellation creation

### Phase 4: AI Enhancement (Q4 2025)
- 📋 Autonomous curriculum generation
- 📋 Cross-framework learning paths
- 📋 AI-powered content creation
- 📋 Predictive learning analytics

---

<div align="center">

**Built with ❤️ for the AI learning community**

[⭐ Star us on GitHub](https://github.com/your-username/gaapf-guidance-ai-agent) | [🐛 Report Issues](https://github.com/your-username/gaapf-guidance-ai-agent/issues) | [💡 Request Features](https://github.com/your-username/gaapf-guidance-ai-agent/discussions)

</div>