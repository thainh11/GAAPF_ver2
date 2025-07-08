# GAAPF Testing Suite

This directory contains comprehensive testing for the GAAPF (Guidance AI Agent for Python Framework) project. The tests cover all functionality according to the specified scenario requirements.

## 🎯 Test Scenario Coverage

The testing suite validates the following scenario requirements:

### Onboarding Flow
- ✅ User configuration and framework selection
- ✅ Framework selection from supported list:
  - LangChain
  - LangGraph  
  - Microsoft AutoGen
  - CrewAI
  - Haystack by Deepset
  - Hugging Face SmolAgents
  - OpenAI Agents Python
- ✅ Tavily tools integration for information collection
- ✅ Initial curriculum generation based on framework difficulty

### Main Learning Flow
- ✅ Agent coordination through constellation graph
- ✅ Theory and code generation in parallel
- ✅ Quiz creation and progress tracking
- ✅ Learning parameter adjustment based on progress
- ✅ Framework syntax compliance in generated code

### Agent Requirements
- ✅ Tool execution (not just action descriptions)
- ✅ LangGraph state management and graph execution
- ✅ Agent handoffs and collaboration

## 📁 Test Files

### 1. `comprehensive_test.py`
**Complete end-to-end testing of all GAAPF functionality**

Tests all components together:
- Core component initialization
- Onboarding flow with Tavily integration
- Curriculum creation and approval workflow
- Agent coordination and handoffs
- Tool execution verification
- Learning flow orchestration
- LangGraph integration

### 2. `test_onboarding_flow.py`
**Focused testing of the onboarding process**

Specific tests:
- Framework selection validation
- User configuration scenarios
- Tavily API integration
- Curriculum generation for different frameworks
- Complete onboarding flow end-to-end

### 3. `test_agent_coordination.py`
**Agent coordination and tool execution testing**

Covers:
- Agent initialization with tools
- Constellation graph setup and compilation
- Basic agent interactions
- Agent handoff scenarios
- Tool execution verification
- LangGraph state management

### 4. `run_tests.py`
**Test runner and orchestrator**

Features:
- Environment validation
- Test discovery and execution
- Comprehensive reporting
- Multiple execution modes

## 🚀 Quick Start

### Prerequisites

1. **Environment Setup**
   ```bash
   # Copy environment template
   cp env_example.txt .env
   
   # Edit .env with your API keys
   nano .env
   ```

2. **Required API Keys**
   - At least one LLM API key:
     - `GOOGLE_API_KEY` (Google Gemini)
     - `OPENAI_API_KEY` (OpenAI GPT)
   - `TAVILY_API_KEY` (for web search tools)

3. **Dependencies**
   ```bash
   # Install project dependencies
   pip install -r ../requirements.txt
   ```

### Running Tests

#### Check Environment
```bash
python run_tests.py env
```

#### List Available Tests
```bash
python run_tests.py list
```

#### Run All Tests
```bash
python run_tests.py all
```

#### Run Specific Tests
```bash
# Comprehensive test suite
python run_tests.py comprehensive

# Onboarding flow only
python run_tests.py onboarding

# Agent coordination only
python run_tests.py agents
```

#### Run Individual Test Files
```bash
# Direct execution
python comprehensive_test.py
python test_onboarding_flow.py
python test_agent_coordination.py
```

## 📊 Test Reports

Tests generate detailed JSON reports with:
- Execution timestamps and durations
- Success/failure status for each test
- Error messages and stack traces
- Environment information
- Performance metrics

Reports are saved as:
- `test_run_report_<timestamp>.json` (from test runner)
- `test_report_<timestamp>.json` (from comprehensive test)
- `onboarding_result_<timestamp>.json` (from onboarding test)
- `agent_coordination_results_<timestamp>.json` (from agent test)

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | One of LLM keys | Google Gemini API key |
| `OPENAI_API_KEY` | One of LLM keys | OpenAI GPT API key |
| `TAVILY_API_KEY` | Yes | Tavily search API key |
| `TOGETHER_API_KEY` | No | Together AI API key |
| `ANTHROPIC_API_KEY` | No | Anthropic Claude API key |

### Test Configuration

Tests can be configured through environment variables:
- `TEST_MODE=true` - Enable test mode
- `LOG_LEVEL=INFO` - Set logging level

## 🧪 Test Details

### Onboarding Flow Tests
1. **Framework Selection**: Validates all required frameworks are supported
2. **User Configuration**: Tests different user experience levels and goals
3. **Tavily Integration**: Verifies web search and information extraction
4. **Curriculum Generation**: Tests curriculum creation for different frameworks

### Agent Coordination Tests
1. **Agent Initialization**: Validates all specialized agents are created
2. **Constellation Graph**: Tests LangGraph compilation and setup
3. **Basic Interactions**: Tests individual agent responses
4. **Handoff Scenarios**: Tests agent-to-agent handoffs
5. **Tool Execution**: Verifies tools are executed, not just described
6. **State Management**: Tests LangGraph state persistence

### Comprehensive Tests
Combines all test scenarios and validates:
- Complete system integration
- End-to-end workflows
- Performance and reliability
- Error handling and recovery

## 🐛 Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```
   Error: No LLM available
   Solution: Set GOOGLE_API_KEY or OPENAI_API_KEY
   ```

2. **Import Errors**
   ```
   Error: Module not found
   Solution: Run from project root directory
   ```

3. **Tool Execution Failures**
   ```
   Error: Tavily API key missing
   Solution: Set TAVILY_API_KEY environment variable
   ```

4. **Memory Errors**
   ```
   Error: ChromaDB initialization failed
   Solution: Ensure sufficient disk space and permissions
   ```

### Debug Mode

For detailed debugging:
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Run with Python debugging
python -u comprehensive_test.py
```

### Test Isolation

Each test creates isolated:
- Memory collections
- Agent instances
- Session states
- Result files

## 📈 Performance Expectations

### Typical Test Durations
- **Onboarding Flow**: 30-60 seconds
- **Agent Coordination**: 45-90 seconds  
- **Comprehensive Test**: 2-5 minutes

### Resource Requirements
- **Memory**: 2-4 GB RAM
- **Storage**: 500 MB for test data
- **Network**: Active internet for API calls

## 🔍 Validation Criteria

Tests validate:
- ✅ Framework information collection using Tavily
- ✅ Curriculum generation and approval workflow
- ✅ Agent coordination through LangGraph
- ✅ Tool execution (not descriptions)
- ✅ Progress tracking and adjustment
- ✅ Framework syntax compliance
- ✅ State management and persistence

## 📝 Contributing

To add new tests:
1. Create test file in `test_project/`
2. Follow existing naming convention
3. Add to `run_tests.py` test discovery
4. Update this README

## 🎉 Success Criteria

A successful test run should show:
- ✅ All supported frameworks recognized
- ✅ Tavily tools executing searches
- ✅ Agents coordinating through constellation graph
- ✅ Curriculum generation working
- ✅ Progress tracking functional
- ✅ LangGraph state management working

The testing suite comprehensively validates that the GAAPF project meets all scenario requirements for an intelligent, tool-enabled learning system. 