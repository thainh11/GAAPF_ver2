# GAAPF AgentBench Evaluation

This directory contains scripts and configurations for evaluating the GAAPF (Guidance AI Agent for Python Framework) system using AgentBench.

## Overview

The evaluation setup consists of:
- **FastAPI Server**: `start_gaapf_api_server.py` - Provides HTTP API interface for GAAPF
- **Benchmark Runner**: `run_gaapf_benchmark.py` - Orchestrates the complete evaluation process
- **AgentBench Integration**: Modified configurations to work with GAAPF API

## Prerequisites

1. **Python Environment**: Python 3.8 or higher
2. **Dependencies**: Install required packages
   ```bash
   pip install -r requirements.txt
   ```

3. **Google Cloud Project**: Set up a Google Cloud project with Vertex AI API enabled
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```
   
4. **Vertex AI Authentication**: Configure authentication (see [Vertex AI Setup Guide](VERTEX_AI_SETUP.md))
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   ```
   Or on Windows:
   ```cmd
   set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
   ```

5. **AgentBench**: Ensure AgentBench is properly set up in the `agentbench/` directory

## Quick Start

### Option 1: Run Complete Evaluation

```bash
python run_gaapf_benchmark.py
```

This will:
1. Start the GAAPF API server
2. Test the API connection
3. Run AgentBench evaluation
4. Display results
5. Clean up (stop the server)

### Option 2: Test API Only

```bash
python run_gaapf_benchmark.py --test-only
```

### Option 3: Manual Process

1. **Start the API server manually**:
   ```bash
   python start_gaapf_api_server.py
   ```

2. **In another terminal, run AgentBench**:
   ```bash
   cd agentbench
   export GAAPF_API_URL="http://127.0.0.1:8000"
   python -m agentbench.main --config configs/assignments/gaapf-lightweight-benchmark.yaml
   ```

## Configuration Options

### Benchmark Runner Options

```bash
python run_gaapf_benchmark.py --help
```

Options:
- `--host`: API server host (default: 127.0.0.1)
- `--port`: API server port (default: 8000)
- `--config`: AgentBench configuration file (default: gaapf-lightweight-benchmark.yaml)
- `--test-only`: Only test API connection, don't run benchmark

### Available Benchmark Configurations

- `gaapf-lightweight-benchmark.yaml`: Quick evaluation with minimal tasks
- `gaapf-full-benchmark.yaml`: Complete evaluation with all available tasks

### API Server Options

```bash
python start_gaapf_api_server.py --help
```

Options:
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development

## API Endpoints

### Health Check
```
GET /health
```
Returns server status and available agents.

### Process Question
```
POST /
Content-Type: application/json

{
  "question": "Your question here",
  "context": {},  // Optional
  "user_id": "benchmark_user"  // Optional
}
```

Response:
```json
{
  "response": "Agent's response",
  "agent_type": "instructor",
  "status": "success"
}
```

## Results

Evaluation results are saved to:
```
agentbench/outputs/latest_run/
```

The results include:
- Task completion rates
- Response quality metrics
- Performance statistics
- Detailed logs

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure GAAPF is properly installed and the Python path is correct
2. **API Connection Failed**: Check if the server started successfully and the port is available
3. **Vertex AI Authentication Errors**: Verify your Google Cloud credentials are properly configured
4. **AgentBench Not Found**: Ensure AgentBench is properly installed in the `agentbench/` directory

### Debugging

1. **Check server logs**: The API server provides detailed logging
2. **Test API manually**: Use curl or a REST client to test the API endpoints
3. **Verify configuration**: Check that the YAML configurations are valid

### Manual API Testing

```bash
# Test health endpoint
curl http://127.0.0.1:8000/health

# Test question endpoint
curl -X POST http://127.0.0.1:8000/ \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello, can you help me learn Python?"}'
```

## Architecture

```
┌─────────────────┐    HTTP     ┌──────────────────┐    Python    ┌─────────────────┐
│   AgentBench    │ ──────────► │  FastAPI Server  │ ───────────► │  GAAPF System   │
│   Evaluation    │             │                  │              │  (Constellation) │
└─────────────────┘             └──────────────────┘              └─────────────────┘
```

1. **AgentBench** sends HTTP requests with questions
2. **FastAPI Server** receives requests and formats them for GAAPF
3. **GAAPF System** processes questions through agent constellation
4. **Responses** flow back through the same path

## Extending the Evaluation

### Adding New Tasks

1. Create new task configurations in `agentbench/configs/tasks/`
2. Update assignment configurations to include new tasks
3. Modify the GAAPF system to handle task-specific requirements

### Customizing Agent Behavior

1. Modify the constellation configuration in the API server
2. Add new specialized agents to the GAAPF system
3. Update the learning context for specific evaluation scenarios

## Performance Considerations

- **Model Selection**: The system uses `gemini-1.5-flash-001` by default for optimal speed/performance balance. You can modify this in the server configuration:
  - `gemini-1.5-pro-001`: More capable but slower
  - `gemini-1.5-flash-001`: Faster, good balance (default)
  - `gemini-1.0-pro`: Legacy model
- **Concurrency**: The FastAPI server supports async processing <mcreference link="https://fastapi.tiangolo.com/benchmarks/" index="3">3</mcreference>
- **Memory Usage**: Monitor memory usage during long evaluations
- **API Quotas**: Be aware of Vertex AI quotas and pricing when running large-scale evaluations
- **Logging Overhead**: Comprehensive logging may impact performance. Adjust log levels if needed
- **Timeout Settings**: Adjust timeouts for complex questions

## License

This evaluation setup is part of the GAAPF project. Please refer to the main project license.