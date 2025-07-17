#!/usr/bin/env python3
"""
GAAPF AgentBench Evaluation Runner

This script orchestrates the complete evaluation process:
1. Starts the GAAPF API server
2. Runs AgentBench evaluation
3. Collects and reports results
"""

import os
import sys
import time
import subprocess
import signal
import requests
from pathlib import Path
import argparse
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GAAPFBenchmarkRunner:
    """
    Manages the complete GAAPF benchmark evaluation process.
    """
    
    def __init__(self, 
                 api_host: str = "127.0.0.1", 
                 api_port: int = 8000,
                 benchmark_config: str = "gaapf-lightweight-benchmark.yaml"):
        self.api_host = api_host
        self.api_port = api_port
        self.api_url = f"http://{api_host}:{api_port}"
        self.benchmark_config = benchmark_config
        self.api_process: Optional[subprocess.Popen] = None
        self.benchmark_dir = Path(__file__).parent
        self.agentbench_dir = self.benchmark_dir / "agentbench"
        
    def check_dependencies(self) -> bool:
        """
        Check if all required dependencies are available.
        """
        logger.info("Checking dependencies...")
        
        # Check if AgentBench directory exists
        if not self.agentbench_dir.exists():
            logger.error(f"AgentBench directory not found: {self.agentbench_dir}")
            return False
            
        # Check if Python packages are available
        required_packages = ['fastapi', 'uvicorn', 'requests', 'langchain_google_vertexai']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        # Check Vertex AI environment variables
        logger.info("Checking Vertex AI configuration...")
        
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            logger.warning("GOOGLE_CLOUD_PROJECT environment variable not set")
            logger.info("Set with: export GOOGLE_CLOUD_PROJECT=your-project-id")
        else:
            logger.info(f"Google Cloud Project: {project_id}")
        
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path:
            if os.path.exists(credentials_path):
                logger.info(f"Service account credentials found: {credentials_path}")
            else:
                logger.warning(f"Service account file not found: {credentials_path}")
                logger.info("Skipping Vertex AI authentication check for external server mode")
        else:
            logger.info("Using Application Default Credentials (ADC)")
            logger.info("If authentication fails, set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'")
            logger.info("Skipping Vertex AI authentication check for external server mode")
        
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        logger.info(f"Vertex AI location: {location}")
            
        logger.info("All dependencies and configuration checks passed")
        return True
    
    def start_api_server(self) -> bool:
        """
        Start the GAAPF API server in a subprocess.
        """
        logger.info(f"Starting GAAPF API server on {self.api_url}...")
        
        try:
            # Start the API server
            server_script = self.benchmark_dir / "start_gaapf_api_server.py"
            
            self.api_process = subprocess.Popen([
                sys.executable, str(server_script),
                "--host", self.api_host,
                "--port", str(self.api_port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get(f"{self.api_url}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("GAAPF API server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(1)
                logger.info(f"Waiting for server to start... ({i+1}/{max_retries})")
            
            logger.error("Failed to start API server within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            return False
    
    def stop_api_server(self):
        """
        Stop the GAAPF API server.
        """
        if self.api_process:
            logger.info("Stopping GAAPF API server...")
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("API server didn't stop gracefully, killing...")
                self.api_process.kill()
            self.api_process = None
    
    def test_api_connection(self) -> bool:
        """
        Test the API connection with a simple request.
        """
        logger.info("Testing API connection...")
        
        try:
            # Test question endpoint directly (no health endpoint available)
            test_request = {
                "question": "Hello, this is a test question. Can you respond?",
                "user_id": "benchmark_test"
            }
            
            response = requests.post(self.api_url, json=test_request, timeout=30)
            if response.status_code != 200:
                logger.error(f"Test question failed: {response.status_code}")
                return False
            
            result = response.json()
            logger.info(f"API test successful. Response: {result.get('response', '')[:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    def run_agentbench(self) -> bool:
        """
        Run the AgentBench evaluation.
        """
        logger.info(f"Running AgentBench with config: {self.benchmark_config}")
        
        try:
            # Set environment variable for API URL
            env = os.environ.copy()
            env['GAAPF_API_URL'] = self.api_url
            
            # Prepare AgentBench command
            config_path = self.agentbench_dir / "configs" / "assignments" / self.benchmark_config
            
            if not config_path.exists():
                logger.error(f"Benchmark config not found: {config_path}")
                return False
            
            # Run AgentBench
            cmd = [
                sys.executable, "-m", "src.assigner",
                "--config", str(config_path)
            ]
            
            logger.info(f"Executing: {' '.join(cmd)}")
            logger.info(f"Working directory: {self.agentbench_dir}")
            
            result = subprocess.run(
                cmd,
                cwd=self.agentbench_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("AgentBench evaluation completed successfully")
                logger.info("STDOUT:")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"AgentBench failed with return code: {result.returncode}")
                logger.error("STDERR:")
                logger.error(result.stderr)
                logger.error("STDOUT:")
                logger.error(result.stdout)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("AgentBench evaluation timed out")
            return False
        except Exception as e:
            logger.error(f"Error running AgentBench: {e}")
            return False
    
    def run_evaluation(self, skip_server_start: bool = False) -> bool:
        """
        Run the complete evaluation process.
        """
        logger.info("Starting GAAPF AgentBench evaluation...")
        
        try:
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Start API server only if not skipping
            if not skip_server_start:
                if not self.start_api_server():
                    return False
            
            # Test API connection
            if not self.test_api_connection():
                return False
            
            # Run benchmark
            success = self.run_agentbench()
            
            return success
            
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return False
        finally:
            # Stop the API server only if we started it
            if not skip_server_start:
                self.stop_api_server()
    
    def show_results(self):
        """
        Display evaluation results.
        """
        results_dir = self.agentbench_dir / "outputs" / "latest_run"
        
        if results_dir.exists():
            logger.info(f"Results saved to: {results_dir}")
            
            # List result files
            result_files = list(results_dir.glob("**/*"))
            if result_files:
                logger.info("Result files:")
                for file in result_files:
                    if file.is_file():
                        logger.info(f"  - {file.relative_to(results_dir)}")
        else:
            logger.warning("No results directory found")

def main():
    parser = argparse.ArgumentParser(description="Run GAAPF AgentBench Evaluation")
    parser.add_argument("--host", default="127.0.0.1", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--config", default="gaapf-lightweight-benchmark.yaml", 
                       help="AgentBench configuration file")
    parser.add_argument("--test-only", action="store_true", 
                       help="Only test API connection, don't run benchmark")
    parser.add_argument("--external-server", action="store_true", 
                       help="Connect to existing external server instead of starting own server")
    
    args = parser.parse_args()
    
    runner = GAAPFBenchmarkRunner(
        api_host=args.host,
        api_port=args.port,
        benchmark_config=args.config
    )
    
    if args.test_only:
        # Just test the API
        logger.info("Running API test only...")
        if args.external_server:
            # Test external server
            if runner.check_dependencies():
                success = runner.test_api_connection()
                sys.exit(0 if success else 1)
            else:
                sys.exit(1)
        else:
            # Start own server for testing
            if runner.check_dependencies() and runner.start_api_server():
                success = runner.test_api_connection()
                runner.stop_api_server()
                sys.exit(0 if success else 1)
            else:
                sys.exit(1)
    
    # Run full evaluation
    success = runner.run_evaluation(skip_server_start=args.external_server)
    
    if success:
        runner.show_results()
        logger.info("Evaluation completed successfully!")
        sys.exit(0)
    else:
        logger.error("Evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()