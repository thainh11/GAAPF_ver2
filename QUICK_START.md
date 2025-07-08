# 🚀 GAAPF Quick Start Guide

## Running the CLI

You have several options to run the GAAPF GAAPF Learning System:

### Option 1: Main CLI Runner (Recommended)
```bash
python run_cli.py
```

With options:
```bash
python run_cli.py --verbose           # Enable verbose logging
python run_cli.py --debug            # Enable debug mode
python run_cli.py --memory custom    # Use custom memory directory
python run_cli.py --help             # Show all options
```

### Option 2: Simple Launcher
```bash
python start_GAAPF.py
```

### Option 3: Direct Package Import
```bash
python -m GAAPF.interfaces.cli.cli
```

### Option 4: Using Poetry (if installed)
```bash
poetry run python run_cli.py
```

## Prerequisites

Make sure you have the required dependencies installed:

```bash
# Using pip
pip install -r requirements.txt

# Using poetry
poetry install
```

## Environment Setup

1. Copy the environment example:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```
# At least one LLM provider is required

# Together AI (Recommended - cost-effective)
TOGETHER_API_KEY=your_together_api_key_here

# Google Gemini (Free tier available)  
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Optional providers
OPENAI_API_KEY=your_openai_key_here

# Optional for enhanced search
TAVILY_API_KEY=your_tavily_key_here

# LLM Provider Priority
LLM_PROVIDER_PRIORITY=together,google-genai,vertex-ai,openai
```

## First Run

1. **Start the CLI:**
   ```bash
   python run_cli.py
   ```

2. **Create a user profile** when prompted

3. **Select a framework** to learn (e.g., LangChain)

4. **Start learning!** The system will guide you through the learning process

## Testing the Installation

Before running the full CLI, you can test if everything is set up correctly:

```bash
python test_setup.py
```

This comprehensive test will verify:
- Python version compatibility
- Required dependencies installation
- Package imports functionality  
- Environment configuration
- Directory structure

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root
cd /path/to/GAAPF-main

# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py
```

### API Key Issues
- Make sure your `.env` file contains valid API keys
- At least one LLM provider key is required
- Check that your API keys have sufficient credits/quota

### Memory/Database Issues
```bash
# Reset the memory database if needed
python scripts/maintenance/reset_chromadb.py
```

## Getting Help

- Type `help` or `h` in the CLI for available commands
- Use `--help` flag with any script for detailed options
- Check the full documentation in `docs/` directory

## Quick Commands in CLI

Once the CLI is running:
- `help` - Show available commands
- `status` - Show current session status  
- `agents` - List available agents
- `clear` - Clear conversation history
- `quit` or `exit` - Exit the CLI

Happy Learning! 🎓 