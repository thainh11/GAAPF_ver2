"""
Sandbox Runner - Secure code execution for GAAPF Phase 2
Provides isolated environment for running generated code and tests.
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

class SandboxRunner:
    """
    Lightweight sandbox for executing code securely.
    Uses subprocess with resource limits (Unix) or timeout (Windows).
    """
    
    def __init__(self, timeout: int = 10, memory_limit_mb: int = 400):
        """
        Initialize SandboxRunner.
        
        Args:
            timeout: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB (Unix only)
        """
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.is_windows = sys.platform.startswith('win')
        
        if not self.is_windows:
            try:
                import resource
                self.resource = resource
            except ImportError:
                logger.warning("resource module not available, memory limits disabled")
                self.resource = None
        else:
            try:
                import psutil
                self.psutil = psutil
            except ImportError:
                logger.warning("psutil not available on Windows, process monitoring disabled")
                self.psutil = None

    def _setup_unix_limits(self):
        """Setup resource limits on Unix systems."""
        if not self.resource:
            return
            
        try:
            # Memory limit (in bytes)
            memory_bytes = self.memory_limit_mb * 1024 * 1024
            self.resource.setrlimit(self.resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # CPU time limit
            self.resource.setrlimit(self.resource.RLIMIT_CPU, (self.timeout, self.timeout))
            
            # Prevent fork bombs
            self.resource.setrlimit(self.resource.RLIMIT_NPROC, (10, 10))
            
        except Exception as e:
            logger.warning(f"Failed to set resource limits: {e}")

    def _kill_process_tree(self, process):
        """Kill process and all its children."""
        if self.is_windows and self.psutil:
            try:
                parent = self.psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                pass
        else:
            try:
                # Unix: kill process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(0.5)
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    def run_in_sandbox(self, cmd: str, timeout: int = None, cwd: Union[Path, str] = ".") -> Dict[str, Any]:
        """
        Execute command in sandbox environment.
        
        Args:
            cmd: Command to execute
            timeout: Override default timeout
            cwd: Working directory
            
        Returns:
            Dict with stdout, stderr, exit_code, execution_time
        """
        if timeout is None:
            timeout = self.timeout
            
        cwd = Path(cwd).resolve()
        if not cwd.exists():
            return {
                "stdout": "",
                "stderr": f"Working directory does not exist: {cwd}",
                "exit_code": 1,
                "execution_time": 0.0
            }

        start_time = time.time()
        
        try:
            # Setup process arguments
            if self.is_windows:
                # Windows: use shell for better compatibility
                process_args = {
                    'shell': True,
                    'cwd': str(cwd),
                    'capture_output': True,
                    'text': True,
                    'timeout': timeout
                }
                preexec_fn = None
            else:
                # Unix: setup process group and limits
                process_args = {
                    'shell': True,
                    'cwd': str(cwd),
                    'capture_output': True,
                    'text': True,
                    'preexec_fn': self._setup_unix_limits,
                    'start_new_session': True  # Create new process group
                }
            
            # Execute command
            try:
                result = subprocess.run(cmd, **process_args)
                execution_time = time.time() - start_time
                
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode,
                    "execution_time": execution_time
                }
                
            except subprocess.TimeoutExpired as e:
                execution_time = time.time() - start_time
                return {
                    "stdout": e.stdout or "",
                    "stderr": f"Command timed out after {timeout}s",
                    "exit_code": -1,
                    "execution_time": execution_time
                }
                
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "stdout": "",
                "stderr": f"Sandbox execution error: {str(e)}",
                "exit_code": -2,
                "execution_time": execution_time
            }

    def validate_code(self, code_path: Union[Path, str], test_command: str = None) -> Dict[str, Any]:
        """
        Validate code by running tests and linting.
        
        Args:
            code_path: Path to code file or directory
            test_command: Custom test command (default: pytest)
            
        Returns:
            Dict with test results and quality metrics
        """
        code_path = Path(code_path)
        if not code_path.exists():
            return {
                "tests_passed": False,
                "quality_score": 0,
                "lint_errors": ["File not found"],
                "test_output": ""
            }
        
        results = {
            "tests_passed": False,
            "quality_score": 0,
            "lint_errors": [],
            "test_output": ""
        }
        
        # Run tests
        if test_command is None:
            test_command = f"python -m pytest {code_path} -v"
        
        test_result = self.run_in_sandbox(test_command, cwd=code_path.parent if code_path.is_file() else code_path)
        results["test_output"] = test_result["stdout"] + test_result["stderr"]
        results["tests_passed"] = test_result["exit_code"] == 0
        
        # Run linting
        lint_result = self.run_in_sandbox(f"python -m ruff check {code_path}", cwd=code_path.parent if code_path.is_file() else code_path)
        if lint_result["exit_code"] != 0 and lint_result["stderr"]:
            results["lint_errors"] = lint_result["stderr"].split('\n')[:5]  # Limit to 5 errors
        
        # Calculate quality score (simple heuristic)
        score = 0
        if results["tests_passed"]:
            score += 70  # 70% for passing tests
        if not results["lint_errors"]:
            score += 30  # 30% for clean linting
        else:
            # Partial credit for few lint errors
            error_count = len([e for e in results["lint_errors"] if e.strip()])
            score += max(0, 30 - error_count * 5)
        
        results["quality_score"] = min(100, score)
        
        return results


# Convenience function for simple usage
def run_in_sandbox(cmd: str, timeout: int = 10, cwd: Union[Path, str] = ".") -> Dict[str, Any]:
    """
    Convenience function to run command in sandbox.
    
    Args:
        cmd: Command to execute
        timeout: Maximum execution time
        cwd: Working directory
        
    Returns:
        Dict with stdout, stderr, exit_code, execution_time
    """
    runner = SandboxRunner(timeout=timeout)
    return runner.run_in_sandbox(cmd, timeout=timeout, cwd=cwd)