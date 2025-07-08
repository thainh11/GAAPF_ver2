# Framework Information Collection Module

## Overview

The Framework Information Collection module is designed to automatically gather comprehensive information about programming frameworks to provide personalized learning experiences. It leverages web search and content extraction technologies to collect documentation, tutorials, API references, and examples for various Python AI frameworks.

## Key Features

- **Automated Information Collection**: Uses Tavily search and extraction to gather documentation, tutorials, API references, and examples
- **Concept Extraction**: Identifies key concepts and relationships within frameworks using heuristic techniques
- **Knowledge Graph Integration**: Builds a semantic network of framework concepts and relationships
- **Curriculum Generation**: Creates personalized learning paths based on user experience and goals
- **Background Processing**: Performs quick initialization during onboarding with more comprehensive collection in the background

## Supported Frameworks

- LangChain
- LangGraph  
- Microsoft AutoGen
- CrewAI
- Haystack by Deepset
- Hugging Face SmolAgents
- OpenAI Agents Python

## Implementation Details

The module consists of several key components:

1. **FrameworkCollector**: Core class responsible for collecting framework information using web search and extraction
2. **Framework Onboarding**: Integration with the onboarding flow to initialize framework knowledge
3. **Knowledge Graph Integration**: Connects framework concepts in a semantic network
4. **Curriculum Generator**: Creates personalized learning paths based on collected information

## Technical Implementation

### Data Collection Strategy

The module uses a two-phase approach to minimize user waiting time:

1. **Quick Initialization** (during onboarding):
   - Collects basic framework information using limited search queries
   - Creates an initial curriculum based on this information
   - Takes only a few seconds to complete

2. **Comprehensive Collection** (background task):
   - Runs asynchronously after onboarding is complete
   - Collects more detailed information about the framework
   - Updates the knowledge graph with comprehensive concept relationships
   - Can take several minutes to complete

### Information Storage

- **Cache**: Framework information is cached in JSON files to avoid redundant collection
- **Vector Database**: Concepts and documentation are stored in ChromaDB for semantic search
- **Knowledge Graph**: Relationships between concepts are stored in a graph structure

### Customization

The curriculum generation process takes into account:

- User experience level (beginner, intermediate, advanced)
- Learning goals (e.g., build production app, research)
- Framework complexity

## Usage Example

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

## Running the Example

To try the Framework Information Collection module:

1. Set up your environment with the necessary API keys:
   ```
   TAVILY_API_KEY=your_tavily_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

2. Run the example script:
   ```bash
   python examples/framework_collection_example.py
   ```

3. Follow the interactive prompts to select a framework and configure your learning experience.

## Integration with the Learning System

The Framework Information Collection module integrates with the broader learning system:

1. During onboarding, users select a framework to learn
2. The system quickly initializes basic information about the framework
3. A personalized curriculum is generated based on user preferences
4. More comprehensive information is collected in the background
5. The knowledge graph is continuously updated with new concepts and relationships
6. Specialized agents use this knowledge to provide personalized learning guidance 