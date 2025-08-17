import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

# Rich imports for modern UI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.theme import Theme
from rich.table import Table
from rich.text import Text

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import simplified components
from ...core.core.simple_hub import SimpleLearningHub
from ...core.agents.instructor import InstructorAgent
from ...core.agents.code_assistant import CodeAssistantAgent  
from ...core.agents.practice_facilitator import PracticeFacilitatorAgent
from ...core.agents.socratic_instructor import SocraticInstructorAgent

# Simple theme for clean interface
SIMPLE_THEME = Theme({
    "primary": "bright_blue",
    "success": "bright_green", 
    "warning": "yellow",
    "error": "bright_red",
    "study_mode": "bright_magenta",
    "info": "cyan",
    "muted": "dim white"
})

class SimpleCLI:
    """
    Simplified CLI with essential features only:
    1. Basic conversation loop
    2. Study mode toggle (Phase 2 ready)
    3. Framework selection
    4. Simple commands
    5. Clean user experience
    
    This replaces the 3757-line complex CLI with ~500 lines (87% reduction)
    while maintaining all essential functionality.
    """
    
    def __init__(
        self,
        llm,
        frameworks_path: Path = Path("frameworks"),
        is_logging: bool = False
    ):
        """
        Initialize simplified CLI.
        
        Args:
            llm: Language model for AI interactions
            frameworks_path: Path to framework configurations
            is_logging: Enable detailed logging
        """
        self.llm = llm
        self.console = Console(theme=SIMPLE_THEME)
        self.is_logging = is_logging
        
        # Defer agent creation to setup_session to avoid double init/logs
        self.agents = {}
        # Initialize simplified hub with empty agents (will be set later)
        self.hub = SimpleLearningHub(llm, self.agents, is_logging)
        
        # Session state
        self.current_framework = "langchain"
        self.study_mode = False
        self.session_start = datetime.now()
        self.interaction_count = 0
        self.user_id = "default"
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        if self.is_logging:
            print(f"SimpleCLI initialized with {len(self.agents)} agents")
    
    async def start(self):
        """Start the simplified CLI experience."""
        try:
            self.show_welcome()
            await self.setup_session()
            await self.conversation_loop()
        except KeyboardInterrupt:
            self.console.print("\n👋 Thanks for using GAAPF! Goodbye!", style="info")
        except Exception as e:
            self.console.print(f"\n❌ Unexpected error: {e}", style="error")
            if self.is_logging:
                import traceback
                traceback.print_exc()
    
    def show_welcome(self):
        """Show welcome message with available commands."""
        welcome_text = """
🤖 **Welcome to GAAPF - Simplified Learning System**

**Available Commands:**
• `/study`     - Enable Study Mode (learn through questions) 🤔
• `/direct`    - Enable Direct Mode (get direct answers) 📚  
• `/framework` - Change learning framework
• `/info`      - Show session information
• `/help`      - Show this help message
• `/quit`      - Exit GAAPF

**Learning Modes:**
• **Study Mode**: Learn through guided questions (Socratic method)
• **Direct Mode**: Get immediate answers and explanations

**Supported Frameworks:**
• LangChain, LangGraph, CrewAI, AutoGen
        """
        
        self.console.print(Panel(welcome_text, title="GAAPF Learning System", style="primary"))
    
    async def setup_session(self):
        """Simple session setup with framework and mode selection."""
        self.console.print("\n🚀 Let's set up your learning session!", style="info")
        
        # Framework selection
        frameworks = ["langchain", "langgraph", "crewai", "autogen"]
        
        self.console.print("\n📚 **Select a framework to learn:**")
        for i, fw in enumerate(frameworks, 1):
            self.console.print(f"  {i}. {fw.title()}")
        
        choice = Prompt.ask("Choose framework (1-4)", default="1")
        try:
            self.current_framework = frameworks[int(choice) - 1]
            self.console.print(f"✅ Selected: **{self.current_framework.title()}**", style="success")
        except (ValueError, IndexError):
            self.current_framework = "langchain"
            self.console.print("✅ Defaulted to: **LangChain**", style="success")
        
        # Set framework in hub
        self.hub.set_framework(self.current_framework)
        # Ensure vectordb is populated for RAG
        try:
            from ...core.tools.framework_collector import FrameworkCollector
            collector = FrameworkCollector(is_logging=self.is_logging)
            stats = collector.ensure_ingested(self.current_framework, persistent_dir="data/frameworks/vectordb")
            if self.is_logging:
                self.console.print(f"(RAG init) {stats}", style="muted")
        except Exception as e:
            if self.is_logging:
                self.console.print(f"(RAG init failed: {e})", style="muted")
        # Rebuild agents with per-framework memory to keep context separated
        from pathlib import Path as _Path
        mem_file = _Path(f"templates/memory_{self.current_framework}.json")
        self.agents = {
            "instructor": InstructorAgent(self.llm, is_logging=self.is_logging, memory_path=mem_file),
            "code_assistant": CodeAssistantAgent(self.llm, is_logging=self.is_logging, memory_path=mem_file),
            "practice": PracticeFacilitatorAgent(self.llm, is_logging=self.is_logging, memory_path=mem_file),
            "socratic_instructor": SocraticInstructorAgent(self.llm, is_logging=self.is_logging, memory_path=mem_file),
        }
        # Update hub agents reference
        self.hub.agents = self.agents
        
        # Ask for user id to keep session consistent
        try:
            entered_id = Prompt.ask("\n👤 Enter your user id", default=self.user_id or "default").strip()
            if entered_id:
                self.user_id = entered_id
        except Exception:
            pass
        
        # Study mode preference
        use_study_mode = Confirm.ask(
            "\n🤔 **Enable Study Mode?** (Learn through guided questions)", 
            default=True
        )
        
        if use_study_mode:
            self.hub.enable_study_mode()
            self.study_mode = True
            self.console.print("🤔 **Study Mode activated!** - You'll learn through discovery", style="study_mode")
        else:
            self.console.print("📚 **Direct Mode activated!** - You'll get direct answers", style="primary")
    
    async def conversation_loop(self):
        """Main conversation loop with user."""
        # Show guided opener once if history exists
        try:
            if self.hub.has_history(self.user_id):
                opener = self.hub.generate_history_aware_opener(self.user_id, self.current_framework, self.study_mode)
                if opener:
                    self.console.print(Panel(opener, title="Welcome back", style="info"))
        except Exception:
            pass

        self.console.print(f"\n💭 **Ask me anything about {self.current_framework.title()}!**")
        self.console.print("💡 Type a command (starting with /) or ask a question", style="muted")
        
        while True:
            try:
                # Show current mode indicator
                mode_indicator = "🤔" if self.study_mode else "📚"
                
                # Get user input
                user_input = Prompt.ask(f"\n[{mode_indicator}] **You**").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    should_continue = await self.handle_command(user_input)
                    if not should_continue:
                        break
                    continue
                
                # Process regular query
                await self.process_user_query(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n👋 Goodbye!", style="info")
                break
            except Exception as e:
                self.console.print(f"\n❌ Error: {e}", style="error")
                if self.is_logging:
                    import traceback
                    traceback.print_exc()
    
    async def process_user_query(self, query: str):
        """Process user query through the learning hub."""
        self.interaction_count += 1
        
        # Show processing indicator
        with self.console.status(f"[cyan]Processing your question..."):
            context = {
                "framework": self.current_framework,
                "study_mode": self.study_mode,
                "interaction_count": self.interaction_count,
            }
            
            # Process query through hub
            response = await self.hub.process_query(query, context)
        
        # Display response
        self.display_response(response)
    
    def display_response(self, response: Dict[str, Any]):
        """Display AI response with proper formatting."""
        agent_name = response.get("agent_used", "Agent")
        content = response.get("content", "")
        study_mode = response.get("study_mode", False)
        
        # Choose style based on mode
        style = "study_mode" if study_mode else "primary"
        mode_text = "Study Guide" if study_mode else "Assistant"
        
        # Display agent name and mode
        header = f"🤖 **{agent_name}** ({mode_text})"
        self.console.print(f"\n{header}", style=style)
        
        # Display main content
        self.console.print(content)
        
        # Show suggestions if available
        suggestions = response.get("suggestions", [])
        if suggestions:
            self.console.print("\n💡 **Suggestions:**", style="warning")
            for suggestion in suggestions[:3]:  # Max 3 suggestions
                self.console.print(f"  • {suggestion}")
        
        # Show error if any
        if response.get("error") and self.is_logging:
            self.console.print(f"\n🔧 Debug: {response['error']}", style="muted")

        # Do not auto-print last exchange after each response to reduce noise
    
    async def handle_command(self, command: str) -> bool:
        """
        Handle CLI commands.
        
        Args:
            command: Command string starting with /
            
        Returns:
            True to continue conversation, False to quit
        """
        cmd = command.lower().strip()
        
        if cmd == "/study":
            self.hub.enable_study_mode()
            self.study_mode = True
            self.console.print(Panel(
                "🤔 **Study Mode Activated!**\n\n"
                "• I'll guide you with questions instead of direct answers\n"
                "• Think through problems step by step\n"
                "• Ask for hints if you get stuck\n"
                "• Type `/direct` to switch back to direct answers",
                style="study_mode",
                title="Study Mode"
            ))
            
        elif cmd == "/direct":
            self.hub.disable_study_mode()
            self.study_mode = False
            self.console.print(Panel(
                "📚 **Direct Mode Activated!**\n\n"
                "• I'll provide direct answers and explanations\n"
                "• Faster learning for quick questions\n"
                "• Type `/study` to switch to guided learning",
                style="primary",
                title="Direct Mode"
            ))
            
        elif cmd == "/framework":
            await self.setup_session()
            
        elif cmd == "/info":
            self.show_session_info()
            
        elif cmd == "/help":
            self.show_welcome()
        elif cmd == "/history":
            # Show last 3 exchanges if available
            try:
                # Prefer transcript storage from any agent that has memory
                # We'll look up via instructor if present
                agent = self.hub.agents.get("instructor") or next(iter(self.hub.agents.values()))
                recent = []
                if getattr(agent, "memory", None):
                    # Pull last 6 messages to display as 3 exchanges
                    recent = agent.memory.get_recent_chat(user_id="default", framework=self.current_framework, k=6)
                if not recent:
                    self.console.print("(No recent history)", style="muted")
                else:
                    lines = []
                    for item in recent[-6:]:
                        role = item.get("role", "?")
                        content = (item.get("content", "") or "")[:180]
                        prefix = "You" if role == "user" else "AI"
                        lines.append(f"{prefix}: {content}")
                    self.console.print(Panel("\n".join(lines), title="Recent history", style="muted"))
            except Exception as e:
                if self.is_logging:
                    self.console.print(f"(history unavailable: {e})", style="muted")
            
        elif cmd in ["/quit", "/exit", "/q"]:
            self.show_goodbye()
            return False
            
        else:
            self.console.print(f"❌ Unknown command: `{command}`", style="error")
            self.console.print("💡 Type `/help` to see available commands", style="muted")
        
        return True

    # Language detection is disabled; enforce English-only UI and prompts
    def _detect_lang(self, text: str) -> str:
        return "en"
    
    def show_session_info(self):
        """Display current session information."""
        duration = datetime.now() - self.session_start
        
        # Create info table
        table = Table(title="Session Information", style="info")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Framework", self.current_framework.title())
        table.add_row("Learning Mode", "Study Mode 🤔" if self.study_mode else "Direct Mode 📚")
        table.add_row("Interactions", str(self.interaction_count))
        table.add_row("Session Duration", str(duration).split('.')[0])  # Remove microseconds
        table.add_row("Available Agents", ", ".join(self.hub.get_available_agents()))
        
        self.console.print(table)
    
    def show_goodbye(self):
        """Show goodbye message with session summary."""
        duration = datetime.now() - self.session_start
        
        goodbye_text = f"""
🎉 **Learning Session Complete!**

**Session Summary:**
• Framework: {self.current_framework.title()}
• Mode: {"Study Mode 🤔" if self.study_mode else "Direct Mode 📚"}
• Interactions: {self.interaction_count}
• Duration: {str(duration).split('.')[0]}

**Keep Learning!** 🚀
        """
        
        self.console.print(Panel(goodbye_text, style="success", title="Goodbye!"))

# Entry point function for easy import
async def start_simple_cli(llm, is_logging: bool = False):
    """
    Start the simplified CLI interface.
    
    Args:
        llm: Language model instance
        is_logging: Enable detailed logging
    """
    cli = SimpleCLI(llm, is_logging=is_logging)
    await cli.start()

if __name__ == "__main__":
    # For direct execution (testing purposes)
    print("SimpleCLI: Use start_simple_cli() function to run with an LLM instance")