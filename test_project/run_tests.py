#!/usr/bin/env python3
"""
Test Runner for GAAPF Project

This script runs all the different test suites and generates a comprehensive report.
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()


class GAAPFTestRunner:
    """Test runner for all GAAPF tests"""
    
    def __init__(self):
        self.test_project_dir = Path(__file__).parent  # Current directory
        self.test_results = {}
        self.start_time = datetime.now()
        print("🚀 GAAPF Test Runner initialized")
    
    def check_environment(self):
        """Check if required environment variables are set"""
        print(f"\n{'='*50}")
        print("CHECKING ENVIRONMENT")
        print(f"{'='*50}")
        
        required_vars = {
            "GOOGLE_API_KEY": "Google Gemini API key",
            "OPENAI_API_KEY": "OpenAI API key",
            "TAVILY_API_KEY": "Tavily search API key"
        }
        
        optional_vars = {
            "TOGETHER_API_KEY": "Together API key",
            "ANTHROPIC_API_KEY": "Anthropic API key"
        }
        
        missing_required = []
        missing_optional = []
        
        print("\n📋 Required Environment Variables:")
        for var, description in required_vars.items():
            if os.getenv(var):
                print(f"   ✅ {var}: {description}")
            else:
                print(f"   ❌ {var}: {description} (MISSING)")
                missing_required.append(var)
        
        print("\n📋 Optional Environment Variables:")
        for var, description in optional_vars.items():
            if os.getenv(var):
                print(f"   ✅ {var}: {description}")
            else:
                print(f"   ⚠️  {var}: {description} (not set)")
                missing_optional.append(var)
        
        # Check if at least one LLM API key is available
        has_llm_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not has_llm_key:
            print(f"\n❌ CRITICAL: No LLM API key found!")
            print("   You need at least one of: GOOGLE_API_KEY or OPENAI_API_KEY")
            return False
        
        if missing_required:
            print(f"\n⚠️  Missing required variables: {', '.join(missing_required)}")
            print("   Some tests may fail or be skipped")
        
        return True
    
    def list_available_tests(self):
        """List all available test files"""
        print(f"\n{'='*50}")
        print("AVAILABLE TESTS")
        print(f"{'='*50}")
        
        test_files = [
            {
                "file": "comprehensive_test.py",
                "name": "Comprehensive Test Suite",
                "description": "Complete end-to-end testing of all GAAPF functionality"
            },
            {
                "file": "test_onboarding_flow.py",
                "name": "Onboarding Flow Test",
                "description": "Framework selection, user config, and Tavily integration"
            },
            {
                "file": "test_agent_coordination.py",
                "name": "Agent Coordination Test",
                "description": "Agent handoffs, tool execution, and LangGraph integration"
            }
        ]
        
        available_tests = []
        for test in test_files:
            test_path = self.test_project_dir / test["file"]
            if test_path.exists():
                available_tests.append(test)
                print(f"   ✅ {test['name']}")
                print(f"      File: {test['file']}")
                print(f"      Description: {test['description']}")
            else:
                print(f"   ❌ {test['name']} (file not found)")
        
        return available_tests
    
    async def run_test_file(self, test_file: str, test_name: str):
        """Run a specific test file"""
        print(f"\n{'='*60}")
        print(f"RUNNING: {test_name}")
        print(f"{'='*60}")
        
        test_path = self.test_project_dir / test_file
        
        if not test_path.exists():
            print(f"❌ Test file not found: {test_path}")
            return {"status": "failed", "error": "Test file not found"}
        
        try:
            # Change to test directory
            original_cwd = os.getcwd()
            os.chdir(self.test_project_dir.parent)
            
            # Run the test
            start_time = datetime.now()
            
            # Execute the test file
            result = subprocess.run([
                sys.executable, str(test_path)
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Restore original directory
            os.chdir(original_cwd)
            
            test_result = {
                "status": "success" if result.returncode == 0 else "failed",
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            if result.returncode == 0:
                print(f"✅ {test_name} completed successfully")
                print(f"   Duration: {duration:.2f} seconds")
            else:
                print(f"❌ {test_name} failed")
                print(f"   Duration: {duration:.2f} seconds")
                print(f"   Return code: {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name} timed out")
            return {"status": "timeout", "error": "Test execution timed out"}
        except Exception as e:
            print(f"💥 Error running {test_name}: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def run_all_tests(self):
        """Run all available tests"""
        print(f"\n{'='*60}")
        print("RUNNING ALL TESTS")
        print(f"{'='*60}")
        
        available_tests = self.list_available_tests()
        
        if not available_tests:
            print("❌ No test files found!")
            return
        
        for test in available_tests:
            result = await self.run_test_file(test["file"], test["name"])
            self.test_results[test["name"]] = result
    
    async def run_specific_test(self, test_name: str):
        """Run a specific test by name"""
        available_tests = self.list_available_tests()
        
        for test in available_tests:
            if test["name"].lower() == test_name.lower() or test["file"] == test_name:
                result = await self.run_test_file(test["file"], test["name"])
                self.test_results[test["name"]] = result
                return result
        
        print(f"❌ Test not found: {test_name}")
        return {"status": "failed", "error": "Test not found"}
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print(f"\n{'='*60}")
        print("GENERATING TEST REPORT")
        print(f"{'='*60}")
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r.get("status") == "success"])
        failed_tests = len([r for r in self.test_results.values() if r.get("status") == "failed"])
        timeout_tests = len([r for r in self.test_results.values() if r.get("status") == "timeout"])
        
        # Create report
        report = {
            "test_run_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration": total_duration,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "timeout_tests": timeout_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "test_results": self.test_results,
            "environment_info": {
                "python_version": sys.version,
                "project_root": str(project_root),
                "test_project_dir": str(self.test_project_dir),
                "has_google_api_key": bool(os.getenv("GOOGLE_API_KEY")),
                "has_openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
                "has_tavily_api_key": bool(os.getenv("TAVILY_API_KEY"))
            }
        }
        
        # Save report
        report_file = self.test_project_dir / f"test_run_report_{int(self.start_time.timestamp())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n📊 TEST RUN SUMMARY")
        print(f"   Total duration: {total_duration:.2f} seconds")
        print(f"   Total tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Timeout: {timeout_tests}")
        print(f"   Success rate: {report['test_run_summary']['success_rate']:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            if result.get("status") == "success":
                icon = "✅"
            elif result.get("status") == "timeout":
                icon = "⏰"
            else:
                icon = "❌"
            
            duration = result.get("duration", 0)
            print(f"   {icon} {test_name}: {result.get('status', 'unknown')} ({duration:.2f}s)")
            
            if result.get("status") == "failed":
                error = result.get("error", "Unknown error")
                print(f"      Error: {error}")
        
        print(f"\n📁 Full report saved to: {report_file}")
        
        return report
    
    def show_help(self):
        """Show help information"""
        print(f"\n{'='*60}")
        print("GAAPF TEST RUNNER HELP")
        print(f"{'='*60}")
        
        print("\nUsage:")
        print("  python run_tests.py [command] [options]")
        
        print("\nCommands:")
        print("  all                    Run all available tests")
        print("  list                   List available tests")
        print("  comprehensive          Run comprehensive test suite")
        print("  onboarding            Run onboarding flow test")
        print("  agents                Run agent coordination test")
        print("  env                   Check environment variables")
        print("  help                  Show this help message")
        
        print("\nExamples:")
        print("  python run_tests.py all")
        print("  python run_tests.py comprehensive")
        print("  python run_tests.py env")
        
        print("\nEnvironment Variables:")
        print("  GOOGLE_API_KEY        Google Gemini API key")
        print("  OPENAI_API_KEY        OpenAI API key")
        print("  TAVILY_API_KEY        Tavily search API key")


async def main():
    """Main execution function"""
    runner = GAAPFTestRunner()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        command = "help"
    else:
        command = sys.argv[1].lower()
    
    if command == "help":
        runner.show_help()
        return
    
    if command == "env":
        runner.check_environment()
        return
    
    if command == "list":
        runner.list_available_tests()
        return
    
    # Check environment before running tests
    if not runner.check_environment():
        print("\n❌ Environment check failed. Please set required API keys.")
        return
    
    if command == "all":
        await runner.run_all_tests()
        runner.generate_report()
    elif command == "comprehensive":
        await runner.run_specific_test("comprehensive_test.py")
        runner.generate_report()
    elif command == "onboarding":
        await runner.run_specific_test("test_onboarding_flow.py")
        runner.generate_report()
    elif command == "agents":
        await runner.run_specific_test("test_agent_coordination.py")
        runner.generate_report()
    else:
        print(f"❌ Unknown command: {command}")
        runner.show_help()


if __name__ == "__main__":
    asyncio.run(main()) 