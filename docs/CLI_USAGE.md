# GAAPF CLI Interface v2.0

An intelligent, async-powered command-line interface for the GAAPF (Guidance AI Agent for Python Framework) learning system.

## ✨ Key Features

- **🤖 Intelligent Agent Selection**: Automatically selects the best AI agent for your learning needs
- **⚡ Async Performance**: Modern async/await architecture for responsive interactions
- **🎯 Personalized Learning**: Adapts to your experience level and learning style
- **💾 Session Management**: Automatic session saving and progress tracking
- **🌈 Rich UI**: Beautiful terminal interface with colors and formatting
- **🔄 Real-time Conversations**: Natural conversations with specialized AI agents

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with asyncio support
2. **At least one LLM provider** configured (see Setup section)
3. **Dependencies installed** via pip or poetry

### Installation

```bash
# Clone and navigate to the project
git clone <repository-url>
cd GAAPF-main

# Install dependencies
pip install -r requirements.txt
# OR using poetry
poetry install
```

### Setup LLM Provider

Set up at least one LLM provider by creating a `.env` file:

```bash
# Recommended: Together AI (cost-effective)
TOGETHER_API_KEY=your_together_api_key_here

# Alternative: Google Gemini (free tier available)
GOOGLE_API_KEY=your_google_api_key_here

# Alternative: OpenAI
OPENAI_API_KEY=your_openai_api_key_here
```

### Run the CLI

```bash
# Simple start
python run_cli.py

# With debug mode for detailed logging
python run_cli.py --debug

# With verbose logging
python run_cli.py --verbose

# Custom directories
python run_cli.py --memory custom_memory --user-profiles custom_profiles
```

## 🎮 Usage

### First Time Setup

1. **User Profile**: Create or select a user profile with your experience level
2. **Framework Selection**: Choose from supported AI agent frameworks:
   - LangChain
   - LangGraph
   - Microsoft AutoGen
   - CrewAI
   - Haystack
3. **Learning Session**: The system automatically initializes an AI constellation

### Interactive Commands

Once in a learning session, you can:

- **Ask Questions**: Type any question about your chosen framework
- **Use Commands**: Type `/help` to see available commands
  - `/help` - Show help information
  - `/status` - View current session status
  - `/agents` - List available AI agents
  - `/clear` - Clear conversation history
  - `/quit` or `/exit` - End session
- **Ctrl+C**: Quick exit with automatic session saving

### Example Conversation

```
You: What are the core components of LangChain?

🤖 Instructor:
LangChain has several core components that work together:

1. **LLMs (Language Models)**: The foundation models that power your applications
2. **Prompts**: Templates and strategies for effective model interaction
3. **Chains**: Sequences of operations that combine different components
4. **Memory**: Systems for maintaining conversation context
5. **Agents**: AI systems that can use tools and make decisions
6. **Tools**: External functions and APIs that agents can utilize

Would you like me to dive deeper into any of these components?

📚 Concepts covered: LLMs, Prompts, Chains
```

## 🤖 AI Agent System

The CLI features an intelligent multi-agent constellation that automatically selects the best agent for your needs:

- **🎓 Instructor Agent**: Explains concepts and theory
- **💻 Code Assistant**: Helps with practical coding examples
- **📚 Documentation Expert**: Provides detailed documentation references
- **🔧 Practice Facilitator**: Creates hands-on exercises
- **📊 Assessment Agent**: Generates quizzes and evaluations
- **🤝 Mentor Agent**: Provides learning guidance and motivation
- **🔍 Research Assistant**: Finds latest information and resources
- **🎯 Project Guide**: Helps with real-world project development
- **🛠️ Troubleshooter**: Assists with debugging and problem-solving
- **🧠 Knowledge Synthesizer**: Connects concepts across topics
- **📈 Progress Tracker**: Monitors and reports learning progress
- **💪 Motivational Coach**: Keeps you engaged and motivated

## 🔧 Configuration

### Command Line Options

```bash
python run_cli.py [OPTIONS]

Options:
  --user-profiles PATH    User profiles directory (default: user_profiles)
  --frameworks PATH       Frameworks directory (default: frameworks)
  --memory PATH          Memory directory (default: memory)
  --verbose              Enable verbose logging
  --debug                Enable debug mode with detailed output
  --help                 Show help message
```

### Environment Variables

```bash
# LLM Provider Priority (comma-separated)
LLM_PROVIDER_PRIORITY=together,google-genai,openai

# API Keys
TOGETHER_API_KEY=your_key
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key

# Debug Mode
DEBUG_MODE=true
```

## 🎯 Supported Frameworks

The CLI supports learning these AI agent frameworks:

1. **LangChain** - Building applications with language models
2. **LangGraph** - Orchestrating agentic workflows
3. **Microsoft AutoGen** - Multi-agent conversations
4. **CrewAI** - Role-playing agent orchestration
5. **Haystack** - NLP pipelines and RAG applications

## 💾 Data Storage

The CLI creates and manages several directories:

- `user_profiles/` - User learning profiles and progress
- `frameworks/` - Framework configurations and metadata
- `memory/` - Long-term memory and conversation history
- `cli_debug.log` - Debug logging (when enabled)

## 🔍 Troubleshooting

### Common Issues

**"No LLM provider configured"**
- Set up at least one API key in your `.env` file
- Check that your API key is valid and has sufficient credits

**"Module not found" errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the correct directory

**"ChromaDB errors"**
- The system uses fallback embeddings if Google API fails
- This is normal and doesn't affect functionality

### Debug Mode

Run with `--debug` for detailed logging:

```bash
python run_cli.py --debug
```

This shows:
- Function entry/exit timing
- Detailed error messages
- Agent selection reasoning
- Session management details

## 🚀 Advanced Features

### Session Management
- Automatic session saving on exit
- Progress tracking across sessions
- Learning context preservation

### Intelligent Agent Selection
- Context-aware agent routing
- User intent recognition
- Learning velocity optimization

### Memory System
- Long-term conversation memory
- Framework-specific knowledge retention
- User preference learning

## 🤝 Contributing

The CLI is part of the larger GAAPF system. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python run_cli.py --debug`
5. Submit a pull request

## 📝 License

This project is licensed under the same terms as the main GAAPF project.

---

**Happy Learning! 🎓**

For more information about the GAAPF system, see the main README.md file. 