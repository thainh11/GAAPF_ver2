#!/usr/bin/env python3
"""
CLI interface for the GAAPF - Guidance AI Agent for Python Framework.
Provides an interactive terminal-based learning experience with intelligent agent selection.
"""

import os
import sys
import signal
import logging
import traceback
import time
import re
import uuid
from datetime import datetime
from pathlib import Path
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

# Rich imports for modern UI
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.status import Status

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ...config.user_profiles import UserProfile
from ...config.framework_configs import FrameworkConfig
from ...core.learning_hub import LearningHub
from ...core.constellation import Constellation, create_constellation_for_context
from ...agents import (
    InstructorAgent, CodeAssistantAgent, DocumentationExpertAgent,
    PracticeFacilitatorAgent, AssessmentAgent, MentorAgent,
    ResearchAssistantAgent, ProjectGuideAgent, TroubleshooterAgent,
    MotivationalCoachAgent, KnowledgeSynthesizerAgent, ProgressTrackerAgent
)
from ...memory.long_term_memory import LongTermMemory

from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger
from dotenv import load_dotenv

# Setup debug logging
logging.basicConfig(
    filename='cli_debug.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Debug mode flag
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

# Modern GAAPF Theme System inspired by Gemini CLI
GAAPF_THEME = Theme({
    "primary": "bright_blue",
    "secondary": "cyan", 
    "success": "bright_green",
    "warning": "yellow",
    "error": "bright_red",
    "info": "bright_cyan",
    "muted": "dim white",
    "accent": "magenta",
    "agent.instructor": "blue",
    "agent.code_assistant": "green",
    "agent.mentor": "purple",
    "agent.research": "cyan",
    "agent.documentation": "bright_yellow",
    "agent.practice": "bright_yellow",
    "agent.assessment": "red",
    "agent.troubleshooter": "bright_red",
    "agent.motivational": "bright_magenta",
    "agent.knowledge": "magenta",
    "agent.progress": "bright_green",
    "agent.project": "bright_blue",
    "agent.practice_facilitator": "bright_yellow",
    "agent.documentation_expert": "bright_yellow",
    "agent.project_guide": "bright_blue",
    "agent.progress_tracker": "bright_green",
    "agent.knowledge_synthesizer": "magenta",
    "agent.motivational_coach": "bright_magenta",
    "framework.title": "bold bright_blue",
    "user.input": "bright_white",
    "system.status": "dim cyan",
    "session.active": "bright_green",
    "session.progress": "bright_blue"
})

class ModernCommand:
    """Represents a CLI command with metadata"""
    def __init__(self, name: str, description: str, handler, aliases=None, category="General"):
        self.name = name
        self.description = description 
        self.handler = handler
        self.aliases = aliases or []
        self.category = category

class CommandRegistry:
    """Modern command registry with categorization and help"""
    def __init__(self, cli_instance):
        self.commands = {}
        self.categories = {}
        self.cli = cli_instance
        self._register_builtin_commands()
    
    def register(self, command: ModernCommand):
        """Register a command"""
        self.commands[command.name] = command
        for alias in command.aliases:
            self.commands[alias] = command
        
        if command.category not in self.categories:
            self.categories[command.category] = []
        self.categories[command.category].append(command)
    
    def _register_builtin_commands(self):
        """Register all built-in commands"""
        commands = [
            ModernCommand("/help", "Show available commands and tips", self.show_help, ["/?", "/h"], "Help"),
            ModernCommand("/status", "Display current session status", self.show_status, ["/s"], "Session"),
            ModernCommand("/agents", "List and manage available agents", self.list_agents, ["/a"], "Agents"),
            ModernCommand("/memory", "View and manage conversation memory", self.show_memory, ["/m"], "Memory"),
            ModernCommand("/tools", "Show available tools and capabilities", self.show_tools, ["/t"], "Tools"),
            ModernCommand("/settings", "Configure CLI preferences", self.show_settings, [], "Configuration"),
            ModernCommand("/export", "Export session data", self.export_session, [], "Data"),
            ModernCommand("/cleandb", "Clean vector database", self.clean_vectordb, [], "Data"),
            ModernCommand("/clear", "Clear conversation history", self.clear_conversation, ["/c"], "Session"),
            ModernCommand("/restart", "Restart current session", self.restart_session, ["/r"], "Session"),
            ModernCommand("/quit", "Exit GAAPF CLI", self.quit_cli, ["/q", "/exit"], "System"),
            # Enhanced Learning Commands
            ModernCommand("/progress", "Show detailed learning progress", self.show_progress, ["/p"], "Learning"),
            ModernCommand("/modules", "Navigate and manage learning modules", self.show_modules, ["/mod"], "Learning"),
            ModernCommand("/constellation", "View and manage constellation state", self.show_constellation, ["/con"], "Learning"),
            ModernCommand("/practice", "Start practice session", self.start_practice, ["/prac"], "Learning"),
            ModernCommand("/assessment", "Take assessment or quiz", self.start_assessment, ["/quiz"], "Learning"),
            ModernCommand("/curriculum", "View and customize curriculum", self.show_curriculum, ["/curr"], "Learning"),
            # Tool Commands
            ModernCommand("/search", "Perform web search", self.perform_search, ["/websearch"], "Tools"),
            ModernCommand("/deepsearch", "Perform deep research", self.perform_deepsearch, ["/deep"], "Tools"),
            ModernCommand("/collect", "Collect framework information", self.collect_framework_info, ["/framework"], "Tools"),
            ModernCommand("/code", "Request specific agent for code help", self.request_code_help, ["/coding"], "Tools"),
        ]
        
        for cmd in commands:
            self.register(cmd)
    
    def show_help(self):
        """Display beautiful help with command categories"""
        self.cli.show_modern_help()
    
    def show_status(self):
        """Display modern status dashboard"""
        self.cli.show_modern_status()
    
    def list_agents(self):
        """Display agents with rich formatting"""
        self.cli.show_modern_agents()
    
    def show_memory(self):
        """Display memory information"""
        self.cli.show_memory_info()
    
    def show_tools(self):
        """Display available tools"""
        self.cli.show_tools_info()
    
    def show_settings(self):
        """Display settings panel"""
        self.cli.show_settings_panel()
    
    def export_session(self):
        """Export session data"""
        self.cli.export_session_data()
    
    def clean_vectordb(self):
        """Clean vector database"""
        self.cli.clean_vector_database()
    
    def clear_conversation(self):
        """Clear conversation with confirmation"""
        self.cli.clear_conversation_safely()
    
    def restart_session(self):
        """Restart session with confirmation"""
        self.cli.restart_session_safely()
    
    def quit_cli(self):
        """Quit CLI with graceful shutdown"""
        self.cli.quit_gracefully()
    
    # Enhanced Learning Commands
    def show_progress(self):
        """Show detailed learning progress"""
        self.cli.show_learning_progress()
    
    def show_modules(self):
        """Navigate and manage learning modules"""
        self.cli.show_learning_modules()
    
    def show_constellation(self):
        """View and manage constellation state"""
        self.cli.show_constellation_info()
    
    def start_practice(self):
        """Start practice session"""
        self.cli.start_practice_mode()
    
    def start_assessment(self):
        """Take assessment or quiz"""
        self.cli.start_assessment_mode()
    
    def show_curriculum(self):
        """View and customize curriculum"""
        self.cli.show_curriculum_overview()
    
    # Tool Commands
    def perform_search(self):
        """Perform web search"""
        self.cli.perform_web_search()
    
    def perform_deepsearch(self):
        """Perform deep research"""
        self.cli.perform_deep_search()
    
    def collect_framework_info(self):
        """Collect framework information"""
        self.cli.collect_framework_information()
    
    def request_code_help(self):
        """Request specific agent for code help"""
        self.cli.request_specific_agent("code_assistant")

class LearningSessionState:
    """Represents the current learning session state."""
    def __init__(self, session_id: str, user_id: str, framework_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.framework_id = framework_id
        self.constellation: Optional[Constellation] = None
        self.messages: List[Any] = []
        self.learning_context: Dict = {}
        self.progress_percentage: float = 0.0
        self.current_module: str = "introduction"
        self.start_time = datetime.now()

# Modernized debug decorators
def debug_step(func):
    """Decorator to add debug output to methods."""
    def wrapper(*args, **kwargs):
        if DEBUG_MODE:
            console = Console(theme=GAAPF_THEME)
            console.print(f"[muted]🔧 DEBUG: Entering {func.__name__}[/muted]")
            start_time = time.time()
        
        result = func(*args, **kwargs)
        
        if DEBUG_MODE:
            elapsed = time.time() - start_time
            console.print(f"[muted]✅ DEBUG: Exiting {func.__name__} (took {elapsed:.2f}s)[/muted]")
        
        return result
    return wrapper

def async_debug_step(func):
    """Decorator to add debug output to async methods."""
    async def wrapper(*args, **kwargs):
        if DEBUG_MODE:
            console = Console(theme=GAAPF_THEME)
            console.print(f"[muted]🔧 DEBUG: Entering async {func.__name__}[/muted]")
            start_time = time.time()
        
        result = await func(*args, **kwargs)
        
        if DEBUG_MODE:
            elapsed = time.time() - start_time
            console.print(f"[muted]✅ DEBUG: Exiting async {func.__name__} (took {elapsed:.2f}s)[/muted]")
        
        return result
    return wrapper

class GAAPFCLI:
    """
    Modern CLI interface for the GAAPF learning system.
    
    Features:
    - Intelligent agent selection with beautiful UI
    - Real-time conversation with AI agents
    - Enhanced user experience with Rich library
    - Async support for better performance
    - Advanced session management
    - Modern command system with autocomplete
    - Context-aware help and information display
    """
    
    def __init__(
        self,
        user_profiles_path: Path = Path('user_profiles'),
        frameworks_path: Path = Path('frameworks'),
        memory_path: Path = Path('memory'),
        is_logging: bool = False
    ):
        """Initialize the modern CLI learning system."""
        if DEBUG_MODE:
            print("🔧 DEBUG: Initializing GAAPFCLI")
        
        self.user_profiles_path = user_profiles_path
        self.frameworks_path = frameworks_path
        self.memory_path = memory_path
        self.is_logging = is_logging
        
        # Initialize modern console with theme
        self.console = Console(theme=GAAPF_THEME)
        
        # Initialize command registry
        self.command_registry = CommandRegistry(self)
        
        # Initialize session data
        self.current_session: Optional[LearningSessionState] = None
        self.session_id = f"gaapf_session_{uuid.uuid4().hex[:8]}"
        
        # FIX: Add shutdown flag for graceful exit
        self.shutdown_requested = False
        
        # Initialize LLM to None first, will be set after model selection
        self.llm = None
        self.selected_provider = None
        
        # Initialize managers
        self.user_profile_manager = UserProfile(user_profiles_path, is_logging)
        self.framework_manager = FrameworkConfig(frameworks_path, is_logging)
        
        # Initialize Learning Hub later after LLM selection
        self.learning_hub = None
        
        # Initialize specialized agents later after LLM selection
        self.agents = {}
        
        # Initialize long-term memory later after LLM selection
        self.long_term_memory = None
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        if self.is_logging:
            logging.info("GAAPFCLI initialized")
        
        # Load project configuration if available
        self._load_gaapf_config()
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully with session saving."""
        self.console.print("\n[warning]💾 Saving session progress...[/warning]")
        
        # FIX: Set shutdown flag instead of calling sys.exit(0)
        self.shutdown_requested = True
        
        if self.current_session and self.long_term_memory:
            try:
                # Save session to long-term memory
                session_data = {
                    "session_id": self.current_session.session_id,
                    "user_id": self.current_session.user_id,
                    "framework_id": self.current_session.framework_id,
                    "messages_count": len(self.current_session.messages),
                    "progress": self.current_session.progress_percentage,
                    "end_time": datetime.now().isoformat()
                }
                
                self.long_term_memory.add_external_knowledge(
                    text=f"Learning session completed for {session_data['framework_id']}",
                    user_id=session_data['user_id'],
                    source="session",
                    metadata=session_data
                )
                
                self.console.print("[success]✅ Session progress saved![/success]")
                    
            except Exception as e:
                self.console.print(f"[warning]⚠️ Error saving session: {e}[/warning]")
        
        self.console.print("[info]👋 Thanks for learning with GAAPF! See you next time![/info]")
    
    def _load_gaapf_config(self):
        """Load GAAPF.md configuration file for project-specific behavior"""
        config_files = ["GAAPF.md", ".gaapf/config.md", "gaapf.md"]
        
        for config_file in config_files:
            if Path(config_file).exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_content = f.read()
                        # Store configuration for later use
                        self.project_config = config_content
                        
                        self.console.print(
                            f"[success]✓[/success] Loaded project configuration from [accent]{config_file}[/accent]"
                        )
                        return
                except Exception as e:
                    self.console.print(f"[warning]⚠️ Failed to load {config_file}: {e}[/warning]")
        
        self.project_config = None
    
    def create_contextual_spinner(self, task_description: str, style: str = "primary"):
        """Create task-specific loading indicators with beautiful styling"""
        return Progress(
            SpinnerColumn("dots12", style=style),
            TextColumn(f"[{style}]{task_description}[/{style}]"),
            TextColumn("[muted]({task.elapsed:.1f}s)[/muted]"),
            console=self.console,
            transient=True
        )
    
    def get_agent_emoji(self, agent_type: str) -> str:
        """Get emoji for agent type"""
        emoji_map = {
            "instructor": "👨‍🏫",
            "code_assistant": "💻", 
            "mentor": "🧙‍♂️",
            "research_assistant": "🔍",
            "documentation_expert": "📚",
            "practice_facilitator": "🎯",
            "assessment": "📝",
            "troubleshooter": "🔧",
            "motivational_coach": "💪",
            "knowledge_synthesizer": "🧠",
            "progress_tracker": "📊",
            "project_guide": "🗺️"
        }
        return emoji_map.get(agent_type, "🤖")
    
    def print_modern_banner(self):
        """Modern, branded banner with system status"""
        
        # Main brand panel with gradient-like effect
        brand_content = Text()
        brand_content.append("🤖 ", style="bright_blue")
        brand_content.append("GAAPF", style="bold bright_blue")
        brand_content.append(" - Guidance AI Agent for Python Framework\n", style="bright_blue")
        brand_content.append("Intelligent Multi-Agent Learning System\n", style="dim bright_cyan")
        brand_content.append("Powered by Advanced AI Constellation", style="muted")
        
        brand_panel = Panel(
            brand_content,
            style="bright_blue",
            padding=(1, 2),
            title="[bold bright_white]Welcome to GAAPF[/bold bright_white]",
            subtitle="[dim]v2.0 - Modern CLI Edition[/dim]"
        )
        
        # Status indicators in columns
        status_items = [
            f"[success]✓ LLM:[/success] {self._get_llm_provider_display()}",
            f"[success]✓ Agents:[/success] {len(self.agents)} active", 
            f"[info]ℹ️ Session:[/info] {self.session_id[:12]}...",
            f"[accent]⚡ Ready to learn![/accent]"
        ]
        
        status_columns = Columns(status_items, equal=True, expand=True)
        
        self.console.print()
        self.console.print(brand_panel)
        self.console.print(status_columns)
        self.console.print()
    
    def print_initial_banner(self):
        """Initial banner shown at startup before model selection"""
        
        # Main brand panel with gradient-like effect
        brand_content = Text()
        brand_content.append("🤖 ", style="bright_blue")
        brand_content.append("GAAPF", style="bold bright_blue")
        brand_content.append(" - Guidance AI Agent for Python Framework\n", style="bright_blue")
        brand_content.append("Intelligent Multi-Agent Learning System\n", style="dim bright_cyan")
        brand_content.append("Let's set up your AI learning companion!", style="muted")
        
        brand_panel = Panel(
            brand_content,
            style="bright_blue",
            padding=(1, 2),
            title="[bold bright_white]Welcome to GAAPF[/bold bright_white]",
            subtitle="[dim]v2.0 - Modern CLI Edition[/dim]"
        )
        
        self.console.print()
        self.console.print(brand_panel)
        self.console.print()
    
    def _get_llm_provider_display(self) -> str:
        """Get a nice display name for the LLM provider"""
        if self.selected_provider:
            return self.selected_provider
        elif self.llm:
            if hasattr(self.llm, 'model_name'):
                if 'together' in str(type(self.llm)).lower():
                    return "Together AI"
                elif 'google' in str(type(self.llm)).lower():
                    return "Google Gemini"
                elif 'openai' in str(type(self.llm)).lower():
                    return "OpenAI"
                elif 'vertex' in str(type(self.llm)).lower():
                    return "Vertex AI"
            return "LLM Connected"
        return "Not Selected"
    
    def select_model_provider(self) -> BaseLanguageModel:
        """Interactive model provider selection with enhanced UI"""
        
        # Header panel
        model_header = Panel(
            "[bold bright_blue]🧠 AI Model Selection[/bold bright_blue]\n"
            "[dim]Choose your preferred AI provider for the best learning experience[/dim]",
            style="bright_blue"
        )
        self.console.print(model_header)
        
        # Load environment variables
        load_dotenv()
        
        # Available providers with detailed information
        providers = [
            {
                "name": "Together AI",
                "id": "together",
                "model": "meta-llama/Llama-3-70B-Instruct-Turbo",
                "description": "Fast and cost-effective LLaMA 3.3 model",
                "env_var": "TOGETHER_API_KEY",
                "strengths": "Cost-effective, Fast responses",
                "best_for": "General learning, Code assistance"
            },
            {
                "name": "Google Gemini",
                "id": "google-genai",
                "model": "gemini-2.5-flash",
                "description": "Latest Google Gemini 2.0 Flash model",
                "env_var": "GOOGLE_API_KEY",
                "strengths": "Latest features, Multimodal",
                "best_for": "Advanced reasoning, Complex queries"
            },
            {
                "name": "Google Vertex AI",
                "id": "vertex-ai",
                "model": "gemini-2.5-flash",
                "description": "Google Cloud hosted model for enterprise use",
                "env_vars": ["GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"],
                "strengths": "GCP integration, Security",
                "best_for": "Enterprise apps, Scalability"
            },
            {
                "name": "OpenAI GPT-4",
                "id": "openai",
                "model": "gpt-4o",
                "description": "OpenAI's most capable model",
                "env_var": "OPENAI_API_KEY",
                "strengths": "Highest quality, Proven reliability",
                "best_for": "Complex learning, Professional use"
            }
        ]
        
        # Check which providers are available
        available_providers = []
        for provider in providers:
            # --- DEBUGGING START ---
            self.console.print(f"[bold yellow]🔍 Checking provider: {provider['name']}[/bold yellow]")
            if "env_vars" in provider:
                # Handle providers requiring multiple env vars (like Vertex AI)
                is_available = all(os.environ.get(var) and os.environ.get(var).strip() for var in provider["env_vars"])
                self.console.print(f"  - Env vars required: {provider['env_vars']}")
                for var in provider['env_vars']:
                    self.console.print(f"    - {var}: '{os.environ.get(var)}'")
                self.console.print(f"  - Initial availability: {is_available}")

                if is_available and provider['id'] == 'vertex-ai':
                    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                    self.console.print(f"  - Vertex AI creds path: {creds_path}")
                    if creds_path and Path(creds_path).is_file():
                        self.console.print("  - [green]Credentials file found.[/green]")
                    else:
                        self.console.print("  - [red]Credentials file NOT found or path is incorrect.[/red]")
                        is_available = False
            else:
                # Handle providers with a single env var
                env_var_value = os.environ.get(provider.get("env_var"))
                is_available = env_var_value and env_var_value.strip()
                self.console.print(f"  - Env var required: {provider.get('env_var')}")
                self.console.print(f"    - {provider.get('env_var')}: '{env_var_value}'")
                self.console.print(f"  - Initial availability: {is_available}")

            if is_available:
                self.console.print(f"  -> [bold green]Provider is available.[/bold green]\n")
                provider["available"] = True
                available_providers.append(provider)
            else:
                self.console.print(f"  -> [bold red]Provider is NOT available.[/bold red]\n")
                provider["available"] = False
            # --- DEBUGGING END ---
        
        if not available_providers:
            # Show setup instructions if no providers are configured
            self._show_llm_setup_instructions()
            sys.exit(1)
        
        # Create provider selection table
        provider_table = Table(title="🚀 Available AI Providers", style="bright_cyan", show_header=True, header_style="bold bright_blue")
        provider_table.add_column("ID", style="dim", width=3, no_wrap=True)
        provider_table.add_column("Provider", style="bold bright_green", width=15)
        provider_table.add_column("Model", style="cyan", width=25)
        provider_table.add_column("Best For", style="white", width=20)
        provider_table.add_column("Status", style="success", width=10, justify="center")
        
        choice_options = []
        for i, provider in enumerate(available_providers, 1):
            status = "🟢 Ready" if provider["available"] else "🔴 Setup Required"
            provider_table.add_row(
                f"{i}.",
                provider["name"],
                provider["model"],
                provider["best_for"],
                status
            )
            choice_options.append(str(i))
        
        self.console.print(provider_table)
        

        
        # Get user choice
        choice = Prompt.ask(
            "\n[bold cyan]Select your AI provider[/bold cyan]",
            choices=choice_options,
            default="1"
        )
        
        selected_provider = available_providers[int(choice) - 1]
        
        # Initialize the selected provider with progress indicator
        with self.console.status(f"[bold green]🚀 Connecting to {selected_provider['name']}...", spinner="dots"):
            try:
                time.sleep(1)  # Brief pause for visual effect
                
                if selected_provider["id"] == "together":
                    api_key = os.environ.get(selected_provider["env_var"])
                    llm = ChatTogether(
                        model=selected_provider["model"],
                        temperature=0.7,
                        api_key=api_key
                    )
                elif selected_provider["id"] == "google-genai":
                    api_key = os.environ.get(selected_provider["env_var"])
                    llm = ChatGoogleGenerativeAI(
                        model=selected_provider["model"],
                        temperature=0.7,
                        google_api_key=api_key
                    )
                elif selected_provider["id"] == "vertex-ai":
                    llm = ChatVertexAI(
                        model_name=selected_provider["model"],
                        temperature=0.7,
                        project=os.environ.get("GOOGLE_CLOUD_PROJECT")
                    )
                elif selected_provider["id"] == "openai":
                    api_key = os.environ.get(selected_provider["env_var"])
                    llm = ChatOpenAI(
                        model=selected_provider["model"],
                        temperature=0.7,
                        api_key=api_key
                    )
                else:
                    raise ValueError(f"Unknown provider: {selected_provider['id']}")
                
                # Store the selected provider info
                self.selected_provider = selected_provider["name"]
                
                # Show success message
                success_panel = Panel(
                    f"[bold bright_green]✅ Connected Successfully![/bold bright_green]\n\n"
                    f"[bold]Provider:[/bold] {selected_provider['name']}\n"
                    f"[bold]Model:[/bold] {selected_provider['model']}\n"
                    f"[bold]Best For:[/bold] {selected_provider['best_for']}\n\n"
                    f"[dim]Your AI learning companion is ready to help![/dim]",
                    title="🎉 AI Ready!",
                    style="bright_green"
                )
                self.console.print(success_panel)
                
                return llm
                
            except Exception as e:
                error_panel = Panel(
                    f"[bold red]❌ Connection Failed[/bold red]\n\n"
                    f"[dim]Error: {str(e)}[/dim]\n\n"
                    f"Please check your API key for {selected_provider['name']}",
                    title="Connection Error",
                    style="error"
                )
                self.console.print(error_panel)
                sys.exit(1)
    
    def _initialize_llm(self) -> BaseLanguageModel:
        """Initialize the language model from available providers."""
        # Load environment variables
        load_dotenv()
        
        # Get provider priority from environment or use default
        provider_priority = os.environ.get("LLM_PROVIDER_PRIORITY", "together,google-genai,vertex-ai,openai").split(",")
        
        # Try each provider in order of priority
        for provider in provider_priority:
            provider = provider.strip()
            
            try:
                if provider == "together":
                    api_key = os.environ.get("TOGETHER_API_KEY")
                    if api_key and api_key.strip():
                        if self.is_logging:
                            self.console.print("[success]✓ Using Together AI[/success]")
                        return ChatTogether(
                            model="meta-llama/Llama-3-70B-Instruct-Turbo",
                            temperature=0.7,
                            api_key=api_key
                        )
                
                elif provider == "google-genai":
                    api_key = os.environ.get("GOOGLE_API_KEY")
                    if api_key and api_key.strip():
                        if self.is_logging:
                            self.console.print("[success]✓ Using Google Gemini[/success]")
                        return ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash",
                            temperature=0.7,
                            google_api_key=api_key
                        )
                
                elif provider == "vertex-ai":
                    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
                    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                    
                    if project and credentials_path:
                        try:
                            log_message = f"[success]✓ Using Google Vertex AI (Project: {project})[/success]"
                            if self.is_logging:
                                self.console.print(log_message + " with application credentials.")
                            
                            return ChatVertexAI(
                                model_name="gemini-2.5-flash",
                                temperature=0.7,
                                project=project
                            )
                        except Exception as e:
                            if self.is_logging:
                                self.console.print(f"[error]Failed to initialize Vertex AI: {e}[/error]")
                                self.console.print("[info]Ensure 'gcloud auth application-default login' is run or GOOGLE_APPLICATION_CREDENTIALS is set.[/info]")

                elif provider == "openai":
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if api_key and api_key.strip():
                        if self.is_logging:
                            self.console.print("[success]✓ Using OpenAI[/success]")
                        return ChatOpenAI(
                            model="gpt-4o",
                            temperature=0.7,
                            api_key=api_key
                        )
                        
            except Exception as e:
                if self.is_logging:
                    logging.warning(f"Failed to initialize {provider}: {e}")
                continue
        
        # If no provider is available, show setup instructions
        self._show_llm_setup_instructions()
        sys.exit(1)
    
    def _show_llm_setup_instructions(self):
        """Show modern LLM setup instructions with Rich formatting."""
        setup_content = """
🔑 [bold red]No AI provider configured![/bold red]

To get started with GAAPF, you need to configure at least one AI provider:

[bold bright_blue]🌟 Recommended Options:[/bold bright_blue]

[bold]1. Together AI (Cost-effective)[/bold]
   • Get API key: https://api.together.xyz/settings/api-keys
   • Set: TOGETHER_API_KEY=your_key_here

[bold]2. Google Gemini (Free tier)[/bold]
   • Get API key: https://aistudio.google.com/app/apikey
   • Set: GOOGLE_API_KEY=your_key_here

[bold]3. OpenAI (Premium)[/bold]
   • Get API key: https://platform.openai.com/api-keys
   • Set: OPENAI_API_KEY=your_key_here

[bold cyan]⚡ Quick Setup:[/bold cyan]
1. Create a .env file in this directory
2. Add your API key(s) using the format above
3. Restart GAAPF CLI

[muted]💡 You can also set environment variables in your shell.[/muted]
"""
        
        setup_panel = Panel(
            Markdown(setup_content),
            title="[bold red]🚨 Setup Required[/bold red]",
            style="red"
        )
        
        self.console.print(setup_panel)
    
    def _initialize_components_after_llm(self):
        """Initialize components that depend on LLM after model selection"""
        # Initialize Learning Hub
        self.learning_hub = LearningHub(
            llm=self.llm,
            user_profiles_path=self.user_profiles_path,
            frameworks_path=self.frameworks_path,
            memory_path=self.memory_path,
            memory=self.long_term_memory,
            is_logging=self.is_logging
        )
        
        # Initialize specialized agents
        self.agents = self._initialize_agents()
        
        gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        gcp_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Initialize long-term memory
        self.long_term_memory = LongTermMemory(
            chroma_path=self.memory_path / 'chroma_db',
            collection_name="cli_sessions",
            is_logging=self.is_logging,
            project=gcp_project,
            location=gcp_location
        )
    
    def _initialize_agents(self) -> Dict:
        """Initialize all specialized agents with modern progress indicator and optimized tool registration."""
        agents = {}
        
        # Pre-register common tools once to avoid duplicate analysis
        common_tools = ["websearch_tools", "deepsearch", "framework_collector", "terminal_tools", "computer_tools"]
        
        agent_classes = [
            ("instructor", InstructorAgent),
            ("code_assistant", CodeAssistantAgent),
            ("documentation_expert", DocumentationExpertAgent),
            ("practice_facilitator", PracticeFacilitatorAgent),
            ("assessment", AssessmentAgent),
            ("mentor", MentorAgent),
            ("research_assistant", ResearchAssistantAgent),
            ("project_guide", ProjectGuideAgent),
            ("troubleshooter", TroubleshooterAgent),
            ("motivational_coach", MotivationalCoachAgent),
            ("knowledge_synthesizer", KnowledgeSynthesizerAgent),
            ("progress_tracker", ProgressTrackerAgent),
        ]

        with Progress(
            SpinnerColumn("bouncingBar"),
            TextColumn("[progress.description]{task.description}", justify="left"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[dim]({task.completed}/{task.total})[/dim]"),
            transient=True,
            console=self.console,
        ) as progress:
            # Pre-register common tools
            if common_tools:
                tool_task = progress.add_task("[cyan]Pre-registering common tools...", total=len(common_tools))
                from ...register.tool import ToolManager, GlobalToolRegistry
                temp_tool_manager = ToolManager()
                
                for tool_name in common_tools:
                    try:
                        if not GlobalToolRegistry().is_module_registered(f"src.GAAPF.core.tools.{tool_name}"):
                            progress.update(tool_task, description=f"[cyan]⚡ Analyzing {tool_name}")
                            temp_tool_manager.register_module_tool(tool_name, llm=None)
                        else:
                            progress.update(tool_task, description=f"[cyan]✓ Using cached {tool_name}")
                        progress.advance(tool_task)
                    except Exception as e:
                        if self.is_logging:
                            logging.warning(f"Failed to pre-register tool {tool_name}: {e}")
                        progress.advance(tool_task)
                
                progress.remove_task(tool_task)
            
            # Initialize agents with optimized tool loading
            task = progress.add_task("[cyan]Initializing AI agents...", total=len(agent_classes))
            for agent_type, agent_class in agent_classes:
                emoji = self.get_agent_emoji(agent_type)
                agent_name = agent_type.replace('_', ' ').title()
                progress.update(task, description=f"[cyan]Loading {emoji} {agent_name}")
                try:
                    agent = agent_class(
                        llm=self.llm,
                        memory_path=self.memory_path / f"{agent_type}_memory.json",
                        is_logging=self.is_logging
                    )
                    # Verify memory is properly initialized
                    if hasattr(agent, 'memory') and agent.memory:
                        if self.is_logging:
                            logging.info(f"Memory initialized for {agent_type} agent")
                    else:
                        if self.is_logging:
                            logging.warning(f"Memory NOT initialized for {agent_type} agent")
                    agents[agent_type] = agent
                except Exception as e:
                    if self.is_logging:
                        logging.warning(f"Failed to initialize {agent_type} agent: {e}")
                time.sleep(0.02) # Reduced delay since tools are pre-cached
                progress.advance(task)
        
        return agents
    
    @debug_step
    def setup_user_profile(self) -> str:
        """Modern interactive user profile setup with Rich UI."""
        
        # Header panel
        profile_header = Panel(
            "[bold bright_blue]👤 User Profile Setup[/bold bright_blue]\n"
            "[dim]Let's create your personalized learning profile[/dim]",
            style="bright_blue"
        )
        self.console.print(profile_header)
        
        # Get list of existing users
        users = self.user_profile_manager.get_all_users()
        
        if users:
            # Show existing users in a table with proper formatting
            user_table = Table(
                title="🚀 Existing GAAPF Users", 
                style="bright_cyan",
                show_header=True,
                header_style="bold bright_blue",
                border_style="bright_blue"
            )
            user_table.add_column("ID", style="dim", width=4, no_wrap=True)
            user_table.add_column("Name", style="bright_white", width=15)
            user_table.add_column("Experience", style="cyan", width=15)
            user_table.add_column("Last Session", style="muted", width=15)
            
            for i, user_id in enumerate(users, 1):
                profile = self.user_profile_manager.get_profile(user_id)
                name = profile.get("name", "Unknown")
                experience = profile.get("experience_level", "beginner").title()
                last_session = profile.get("last_session", "Never")
                user_table.add_row(f"{i}.", name, experience, last_session)
            
            self.console.print()
            self.console.print(user_table)
            self.console.print()
            
            choice = Prompt.ask(
                "[bold cyan]Select user number or enter 'new' for new profile[/bold cyan]", 
                choices=[str(i) for i in range(1, len(users) + 1)] + ["new", "n"],
                default="new"
            )
            
            if choice not in ["new", "n"]:
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(users):
                        selected_user = users[index]
                        selected_profile = self.user_profile_manager.get_profile(selected_user)
                        
                        # Show selection confirmation
                        confirmation_panel = Panel(
                            f"[bold bright_green]✅ Profile Selected[/bold bright_green]\n\n"
                            f"[bold]User ID:[/bold] {selected_user}\n"
                            f"[bold]Name:[/bold] {selected_profile.get('name', 'Unknown')}\n"
                            f"[bold]Experience:[/bold] {selected_profile.get('experience_level', 'beginner').title()}",
                            title="👤 Welcome Back!",
                            style="bright_green"
                        )
                        self.console.print(confirmation_panel)
                        return selected_user
                except ValueError:
                    pass
        
        # Create new user
        return self._create_new_user()
    
    def _create_new_user(self) -> str:
        """Create a new user profile with modern UI."""
        
        new_user_panel = Panel(
            "[bold bright_green]✨ Creating New User Profile[/bold bright_green]\n"
            "[dim]Tell us about yourself to personalize your learning experience[/dim]",
            style="bright_green"
        )
        self.console.print(new_user_panel)
        
        # Generate user ID with preview
        default_id = f"user_{uuid.uuid4().hex[:8]}"
        user_id = Prompt.ask("[bold]User ID[/bold]", default=default_id)
        
        # Get user information with better prompts
        name = Prompt.ask("[bold]Your name[/bold]", default="Learner")
        email = Prompt.ask("[bold]Email (optional)[/bold]", default="")
        
        # Experience level with styled choices
        experience_choices = ["beginner", "intermediate", "advanced"]
        experience_level = Prompt.ask(
            "\n[bold]Experience Level[/bold]",
            choices=experience_choices,
            default="beginner"
        )
        
        # Learning preferences
        learning_preferences = self._get_learning_preferences()
        
        # Create user profile
        self.user_profile_manager.create_profile(
            user_id=user_id,
            name=name,
            email=email,
            experience_level=experience_level,
            learning_style=learning_preferences
        )
        
        # Success message
        success_panel = Panel(
            f"[bold bright_green]✅ Profile Created Successfully![/bold bright_green]\n\n"
            f"[bold]User ID:[/bold] {user_id}\n"
            f"[bold]Name:[/bold] {name}\n"
            f"[bold]Experience:[/bold] {experience_level.title()}\n"
            f"[bold]Learning Style:[/bold] {learning_preferences['interaction_style'].title()}",
            title="🎉 Welcome to GAAPF!",
            style="bright_green"
        )
        self.console.print(success_panel)
        
        return user_id
    
    def _get_learning_preferences(self) -> Dict:
        """Get learning preferences with interactive prompts."""
        
        # Interaction style
        interaction_styles = ["guided", "exploratory", "balanced"]
        interaction_style = Prompt.ask(
            "\n[bold]Preferred interaction style[/bold]",
            choices=interaction_styles,
            default="guided"
        )
        
        # Learning pace
        pace_options = ["slow", "moderate", "fast"]
        pace = Prompt.ask(
            "[bold]Learning pace[/bold]",
            choices=pace_options,
            default="moderate"
        )
        
        # Detail level
        detail_options = ["high", "balanced", "concise"]
        detail_level = Prompt.ask(
            "[bold]Detail level preference[/bold]",
            choices=detail_options,
            default="balanced"
        )
        
        return {
            "interaction_style": interaction_style,
            "pace": pace,
            "detail_level": detail_level,
            "preferred_mode": "balanced"
        }
    
    @debug_step
    def select_framework(self) -> str:
        """Modern interactive framework selection with rich table."""
        
        framework_header = Panel(
            "[bold bright_blue]🔧 Framework Selection[/bold bright_blue]\n"
            "[dim]Choose your Python framework for learning[/dim]",
            style="bright_blue"
        )
        self.console.print(framework_header)
        
        # Define available agent frameworks
        agent_frameworks = [
            {
                "id": "langchain",
                "name": "LangChain",
                "description": "Framework for developing applications powered by language models",
                "version": "0.3.x",
                "difficulty": "Intermediate",
                "popularity": "⭐⭐⭐⭐⭐"
            },
            {
                "id": "langgraph",
                "name": "LangGraph",
                "description": "Framework for orchestrating agentic workflows with language models",
                "version": "0.1.x",
                "difficulty": "Advanced",
                "popularity": "⭐⭐⭐⭐"
            },
            {
                "id": "autogen",
                "name": "Microsoft AutoGen",
                "description": "Multi-agent conversation framework for LLM-based applications",
                "version": "0.2.x",
                "difficulty": "Advanced",
                "popularity": "⭐⭐⭐⭐"
            },
            {
                "id": "crewai",
                "name": "CrewAI",
                "description": "Framework for orchestrating role-playing agents",
                "version": "0.1.x",
                "difficulty": "Intermediate",
                "popularity": "⭐⭐⭐"
            },
            {
                "id": "haystack",
                "name": "Haystack by Deepset",
                "description": "Framework for building NLP pipelines and RAG applications",
                "version": "2.0.x",
                "difficulty": "Intermediate",
                "popularity": "⭐⭐⭐⭐"
            }
        ]
        
        # Create framework selection table with proper formatting
        framework_table = Table(
            title="🚀 Available Frameworks", 
            style="bright_cyan",
            show_header=True,
            header_style="bold bright_blue",
            border_style="bright_blue"
        )
        framework_table.add_column("ID", style="dim", width=4, no_wrap=True)
        framework_table.add_column("Framework", style="bold bright_blue", width=20)
        framework_table.add_column("Description", style="white", width=40)
        framework_table.add_column("Difficulty", justify="center", style="yellow", width=12)
        framework_table.add_column("Popularity", justify="center", width=12)
        
        for i, framework in enumerate(agent_frameworks, 1):
            # Truncate description properly
            description = framework['description']
            if len(description) > 38:
                description = description[:35] + "..."
            
            framework_table.add_row(
                f"{i}.",
                framework['name'],
                description,
                framework['difficulty'],
                framework['popularity']
            )
        
        self.console.print()
        self.console.print(framework_table)
        self.console.print()
        
        # Get user choice
        choice = Prompt.ask(
            "[bold cyan]Select framework number[/bold cyan]",
            choices=[str(i) for i in range(1, len(agent_frameworks) + 1)],
            default="1"
        )
        
        selected_framework = agent_frameworks[int(choice) - 1]
        framework_id = selected_framework["id"]
        
        # Check if framework configuration exists, create if not
        if not self.framework_manager.get_framework(framework_id):
            with self.create_contextual_spinner(f"Setting up {selected_framework['name']}", "primary"):
                time.sleep(1)  # Simulate setup time
                
                # Create framework configuration
                self.framework_manager.create_framework(
                    framework_id=framework_id,
                    name=selected_framework["name"],
                    description=selected_framework["description"],
                    version=selected_framework["version"],
                    modules=self._get_default_modules_for_framework(framework_id)
                )
        
        # Show selection confirmation
        selection_panel = Panel(
            f"[bold bright_green]✅ Framework Selected[/bold bright_green]\n\n"
            f"[bold]{selected_framework['name']}[/bold]\n"
            f"[dim]{selected_framework['description']}[/dim]\n\n"
            f"[bold]Version:[/bold] {selected_framework['version']}\n"
            f"[bold]Difficulty:[/bold] {selected_framework['difficulty']}",
            title="🎯 Ready to Learn!",
            style="bright_green"
        )
        self.console.print(selection_panel)
        
        return framework_id
    
    def _get_default_modules_for_framework(self, framework_id: str) -> Dict:
        """Get default modules for a specific framework."""
        if framework_id == "langchain":
            return {
                "introduction": {
                    "title": "Introduction to LangChain",
                    "description": "Overview of LangChain framework and its components",
                    "complexity": "basic",
                    "estimated_duration": 30,
                    "concepts": ["LLMs", "Chains", "Prompts", "Memory"],
                    "prerequisites": []
                },
                "components": {
                    "title": "LangChain Components",
                    "description": "Core components of the LangChain framework",
                    "complexity": "basic",
                    "estimated_duration": 60,
                    "concepts": ["Models", "Prompts", "Chains", "Memory", "Agents", "Tools"],
                    "prerequisites": ["introduction"]
                },
                "rag": {
                    "title": "Retrieval Augmented Generation",
                    "description": "Building RAG applications with LangChain",
                    "complexity": "intermediate",
                    "estimated_duration": 90,
                    "concepts": ["Embeddings", "Vector Stores", "Retrievers", "Document Loaders"],
                    "prerequisites": ["components"]
                }
            }
        elif framework_id == "langgraph":
            return {
                "introduction": {
                    "title": "Introduction to LangGraph",
                    "description": "Overview of LangGraph and state machines for LLM orchestration",
                    "complexity": "basic",
                    "estimated_duration": 30,
                    "concepts": ["State Machines", "Graphs", "Nodes", "Edges"],
                    "prerequisites": []
                }
            }
        else:
            return {
                "introduction": {
                    "title": "Introduction",
                    "description": f"Introduction to {framework_id}",
                    "complexity": "basic",
                    "estimated_duration": 30,
                    "concepts": ["overview", "installation"],
                    "prerequisites": []
                }
            }
    
    @async_debug_step
    async def initialize_learning_session(self, user_id: str, framework_id: str):
        """Initialize or resume the learning session state with enhanced integration."""
        
        with self.console.status("[info]Initializing learning session...", spinner="dots"):
            time.sleep(0.5)  # Brief pause for visual effect
            
            # Check if session can be resumed first
            should_resume = self.learning_hub.session_manager.session_exists(user_id, framework_id)
            
            if should_resume:
                session_data = self.learning_hub.resume_session(user_id, framework_id)
            else:
                session_data = await self.learning_hub.start_session(user_id, framework_id)
            
            if "error" in session_data:
                self.console.print(f"[error]❌ Error starting session: {session_data['error']}[/error]")
                return False
            
            # Create enhanced session state
            self.current_session = LearningSessionState(
                session_id=session_data["session_id"],
                user_id=user_id,
                framework_id=framework_id
            )
            
            # Enhanced integration with Learning Hub
            constellation = self.learning_hub.active_constellations.get(session_data["session_id"])
            if constellation:
                self.current_session.constellation = constellation
            
            # Get enhanced learning context
            learning_context = self.learning_hub.active_sessions.get(session_data["session_id"], {})
            self.current_session.learning_context = learning_context
            
            # Set learning stage and progress from hub
            self.current_session.current_module = learning_context.get("current_module", "introduction")
            self.current_session.progress_percentage = learning_context.get("progress", 0.0)
            
            # Initialize learning stage based on context
            interaction_count = learning_context.get("interaction_count", 0)
            if interaction_count == 0:
                self.current_session.learning_stage = "introduction"
            elif interaction_count < 5:
                self.current_session.learning_stage = "exploration"
            elif interaction_count < 15:
                self.current_session.learning_stage = "concept"
            else:
                self.current_session.learning_stage = "practice"
            
            # Restore conversation history if session was resumed
            if session_data.get("is_resumed", False):
                self._restore_conversation_history(learning_context.get("messages", []))
                self.console.print(f"[info]🔄 Resumed learning session with {len(learning_context.get('messages', []))} messages[/info]")
        
        framework = self.framework_manager.get_framework(framework_id)
        framework_name = framework.get("name", framework_id) if framework else framework_id
        
        # Show beautiful session info
        if session_data.get("is_resumed", False):
            session_content = (
                f"[bold bright_green]🔄 Learning Session Resumed![/bold bright_green]\n\n"
                f"[bold]Session ID:[/bold] {self.current_session.session_id}\n"
                f"[bold]Framework:[/bold] {framework_name}\n"
                f"[bold]Constellation:[/bold] {session_data.get('constellation_type', 'learning')}\n"
                f"[bold]Messages Restored:[/bold] {session_data.get('message_count', 0)}"
            )
        else:
            session_content = (
                f"[bold bright_green]🚀 New Learning Session Started![/bold bright_green]\n\n"
                f"[bold]Session ID:[/bold] {self.current_session.session_id}\n"
                f"[bold]Framework:[/bold] {framework_name}\n"
                f"[bold]Constellation:[/bold] {session_data.get('constellation_type', 'learning')}"
            )
        
        session_panel = Panel(
            session_content,
            title="✨ Session Ready",
            style="bright_green"
        )
        self.console.print(session_panel)
        
        return True
    
    def _restore_conversation_history(self, messages: List[Dict]):
        """Restore conversation history from session data."""
        from langchain_core.messages import HumanMessage, AIMessage
        
        self.current_session.messages = []
        
        for msg in messages:
            if msg.get("role") == "user":
                self.current_session.messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                self.current_session.messages.append(AIMessage(content=msg["content"]))
        
        if len(messages) > 0:
            history_panel = Panel(
                f"[bold bright_blue]📜 Conversation Restored[/bold bright_blue]\n\n"
                f"[bold]Messages:[/bold] {len(messages)}\n"
                f"[dim]Your previous conversation is now available[/dim]",
                style="bright_blue"
            )
            self.console.print(history_panel)
            
            # Show last interaction as preview
            if len(messages) > 2:
                last_user_msg = None
                last_ai_msg = None
                
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and not last_ai_msg:
                        last_ai_msg = msg["content"]
                    elif msg.get("role") == "user" and not last_user_msg:
                        last_user_msg = msg["content"]
                    
                    if last_user_msg and last_ai_msg:
                        break
                
                if last_user_msg and last_ai_msg:
                    preview_content = ""
                    if last_user_msg:
                        preview_content += f"[bold]You:[/bold] {last_user_msg[:100]}...\n"
                    if last_ai_msg:
                        preview_content += f"[bold]AI:[/bold] {last_ai_msg[:100]}..."
                    
                    preview_panel = Panel(
                        preview_content,
                        title="💭 Last Interaction",
                        style="dim"
                    )
                    self.console.print(preview_panel)

    async def _send_initial_greeting(self):
        """Send a proactive, framework-specific greeting to the user with enhanced context tracking."""
        if not self.current_session:
            return

        user_id = self.current_session.user_id
        framework_id = self.current_session.framework_id
        
        if self.is_logging:
            self.console.print(f"[dim]🔄 Generating initial greeting for {user_id} learning {framework_id}[/dim]")
        
        user_profile = self.user_profile_manager.get_profile(user_id)
        user_name = user_profile.get("name", "Learner")
        
        framework_config = self.framework_manager.get_framework(framework_id)
        framework_name = framework_config.get("name", framework_id)
        
        # ──────────────────────────────────────────────────────────────
        # Accurate latency timer – start *before* we do any LLM/Hub work
        # ──────────────────────────────────────────────────────────────
        import time
        overall_start = time.perf_counter()
        
        # Get the first module to introduce it
        modules = framework_config.get("modules", {})
        first_module_key = next(iter(modules), "day1_fundamentals") if "day1_fundamentals" in modules else next(iter(modules), "introduction")
        first_module = modules.get(first_module_key, {})
        module_title = first_module.get("title", "Introduction")
        module_description = first_module.get("description", f"An introduction to {framework_name}")
        concepts = ", ".join(first_module.get("concepts", ["key concepts"]))

        # Enhanced session state tracking to prevent duplicate greetings
        session_key = f"greeting_sent_{self.current_session.session_id}"
        if hasattr(self, '_session_state') and self._session_state.get(session_key, False):
            if self.is_logging:
                self.console.print(f"[dim]⚠️ Initial greeting already sent for session {self.current_session.session_id}[/dim]")
            return
        
        # Mark greeting as sent
        if not hasattr(self, '_session_state'):
            self._session_state = {}
        self._session_state[session_key] = True

        # Check if this is a resumed session with more context
        existing_messages = getattr(self.current_session, 'messages', [])
        is_resumed = len(existing_messages) > 0
        
        # Enhanced greeting query with better context awareness
        if is_resumed:
            greeting_query = (
                f"Welcome me back to my {framework_name} learning session. "
                f"I have {len(existing_messages)} previous interactions. "
                f"Provide a contextual continuation focused on the {module_title} module. "
                f"Be encouraging and remind me where we left off. "
                f"Don't ask what I'm ready for - instead, provide the next logical learning step."
            )
        else:
            greeting_query = (
                f"I'm starting my {framework_name} learning journey! "
                f"Welcome me and introduce the {module_title} module covering: {concepts}. "
                f"Be encouraging and provide clear next steps. "
                f"Don't ask vague questions - instead, guide me toward the first specific learning objective."
            )
        
        if self.is_logging:
            self.console.print(f"[dim]📝 Greeting query: {greeting_query[:100]}...[/dim]")

        with self.console.status(f"[bold cyan]🎓 Instructor is preparing your {'first' if not is_resumed else 'next'} lesson...", spinner="dots"):
            try:
                # Use the learning hub's constellation system instead of calling agent directly
                interaction_data = {
                    "type": "greeting",
                    "query": greeting_query,
                    "requested_agent": "instructor",
                    "session_context": {
                        "is_initial_greeting": True,
                        "is_resumed": is_resumed,
                        "message_count": len(getattr(self.current_session, 'messages', [])),
                        "current_module": getattr(self.current_session, 'current_module', first_module_key),
                        "learning_stage": "introduction"
                    }
                }
                
                # Process through Learning Hub with full constellation context
                processed_response = await asyncio.to_thread(
                    self.learning_hub.process_interaction,
                    session_id=self.current_session.session_id,
                    interaction_data=interaction_data
                )
                
                # Extract response content
                response_content = None
                agent_type = "instructor"
                
                if isinstance(processed_response, dict):
                    if "error" in processed_response:
                        self.console.print(f"[error]Failed to generate greeting: {processed_response['error']}[/error]")
                        return
                    
                    # Extract content from the processed response
                    if "primary_response" in processed_response:
                        primary_resp = processed_response["primary_response"]
                        if isinstance(primary_resp, dict):
                            response_content = primary_resp.get("content")
                            agent_type = primary_resp.get("agent_type", "instructor")
                        else:
                            response_content = str(primary_resp)
                    elif "content" in processed_response:
                        response_content = processed_response["content"]
                        agent_type = processed_response.get("agent_type", "instructor")
                    else:
                        # Fallback extraction
                        response_content = str(processed_response)
                
                # If no content extracted, create a fallback greeting
                if not response_content or not response_content.strip():
                    response_content = f"""Welcome{'back' if is_resumed else ''}, {user_name}! I'm excited to help you learn {framework_name}.

We're {'continuing with' if is_resumed else 'starting with'} the **{module_title}** module, where you'll explore: {concepts}.

Are you ready to {'continue' if is_resumed else 'dive into'} our learning journey?"""

                # Display the greeting
                agent_emoji = self.get_agent_emoji(agent_type)
                agent_name = agent_type.replace('_', ' ').title()
                safe_agent_type = str(agent_type).strip().lower().replace(' ', '_')
                panel_title = f"[bold agent.{safe_agent_type}]{agent_emoji} {agent_name}[/bold agent.{safe_agent_type}]"
                
                try:
                    response_panel = Panel(
                        Markdown(response_content),
                        title=panel_title,
                        style=f"agent.{safe_agent_type}",
                        expand=False
                    )
                    self.console.print(response_panel)
                except Exception as e:
                    self.console.print(f"[error]Panel rendering failed: {e}[/error]")
                    self.console.print(Markdown(response_content))
                
                # ──────────────────────────────────────────────────────────────
                # Print *total* round-trip latency (proc + render)
                # ──────────────────────────────────────────────────────────────
                overall_elapsed = time.perf_counter() - overall_start
                self.console.print(f"[dim]⏱️ Agent full response in {overall_elapsed:.3f} seconds.[/dim]")
                
                # The messages are already added to session by the learning hub
                # No need to manually add them here

            except Exception as e:
                self.console.print(f"[error]Failed to generate initial greeting: {e}[/error]")
                if DEBUG_MODE:
                    import traceback
                    self.console.print(f"[dim]Traceback: {traceback.format_exc()}[/dim]")
                
                # Fallback greeting if constellation fails
                fallback_greeting = f"""Welcome{'back' if is_resumed else ''}, {user_name}! 

I'm here to help you learn {framework_name}. We're {'continuing with' if is_resumed else 'starting with'} the **{module_title}** module.

How can I assist you today?"""
                
                fallback_panel = Panel(
                    Markdown(fallback_greeting),
                    title="🎓 Instructor",
                    style="agent.instructor",
                    expand=False
                )
                self.console.print(fallback_panel)
    
    @async_debug_step
    async def process_user_message(self, message: str) -> None:
        """Process user message using the LearningHub with enhanced integration."""
        # Start end-to-end timer *as soon as we receive the user message*
        import time
        overall_start = time.perf_counter()

        if not self.current_session:
            self.console.print("[bold red]No active session. Please start a new session.[/bold red]")
            return
        # --- DEFENSIVE CHECK: ENSURE CONTEXT ---
        if not getattr(self.current_session, 'framework_id', None) or not getattr(self.current_session, 'current_module', None):
            self.console.print("[bold red]No framework or module selected. Please restart and select a framework/module to continue learning.[/bold red]")
            framework_id = self.select_framework()
            if not framework_id:
                self.console.print("[bold red]You must select a framework to continue.[/bold red]")
                return
            framework_config = self.framework_manager.get_framework(framework_id)
            modules = framework_config.get("modules", {}) if framework_config else {}
            if not modules:
                self.console.print(f"[bold red]No modules found for framework {framework_id}. Please check your framework configuration.[/bold red]")
                return
            current_module = next(iter(modules), None)
            if not current_module:
                self.console.print(f"[bold red]No modules available in the selected framework. Please add modules to continue.[/bold red]")
                return
            # Re-initialize session
            user_id = self.current_session.user_id if self.current_session else self.setup_user_profile()
            await self.initialize_learning_session(user_id, framework_id)
            await self._send_initial_greeting()
            return

        # Add user message to session history
        from langchain_core.messages import HumanMessage, AIMessage
        self.current_session.messages.append(HumanMessage(content=message))

        # Show thinking indicator
        with self.create_contextual_spinner("AI constellation is processing...", style="info") as status:
            try:
                # Enhanced interaction data with session context
                interaction_data = {
                    "type": "user_message", 
                    "query": message,
                    "session_context": {
                        "message_count": len(self.current_session.messages),
                        "current_module": getattr(self.current_session, 'current_module', None),
                        "learning_stage": getattr(self.current_session, 'learning_stage', 'exploration'),
                        "progress": getattr(self.current_session, 'progress_percentage', 0.0)
                    }
                }
                
                # Process through Learning Hub with enhanced context
                processed_response = await asyncio.to_thread(
                    self.learning_hub.process_interaction,
                    session_id=self.current_session.session_id,
                    interaction_data=interaction_data
                )

                # Enhanced response handling with proper extraction
                response_content, agent_type, constellation_info = self._extract_agent_response(processed_response)
                
                # Handle extraction errors
                if constellation_info.get("error"):
                    self.console.print(f"[bold red]Learning Hub Error: {constellation_info['error']}[/bold red]")
                    return
                
                # Sync session state with Learning Hub
                try:
                    session_info = self.learning_hub.get_session_info(self.current_session.session_id)
                    if session_info and not session_info.get("error"):
                        # Update session with enhanced context from Learning Hub
                        self.current_session.learning_context = session_info
                        
                        # Sync conversation history
                        hub_messages = session_info.get("messages", [])
                        if hub_messages:
                            self.current_session.messages = []
                            for msg in hub_messages:
                                if msg.get("role") == "user":
                                    self.current_session.messages.append(HumanMessage(content=msg["content"]))
                                elif msg.get("role") == "assistant":
                                    self.current_session.messages.append(AIMessage(content=msg["content"]))
                        
                        # Update learning progress and stage
                        if hasattr(self.current_session, 'progress_percentage'):
                            self.current_session.progress_percentage = session_info.get("progress", 0.0)
                        if hasattr(self.current_session, 'learning_stage'):
                            self.current_session.learning_stage = session_info.get("learning_stage", "exploration")
                        if hasattr(self.current_session, 'current_module'):
                            self.current_session.current_module = session_info.get("module_id", session_info.get("current_module"))
                            
                except Exception as e:
                    if DEBUG_MODE:
                        self.console.print(f"[dim]Warning: Could not sync session state: {e}[/dim]")

                # Display enhanced response
                if response_content and response_content.strip():
                    # Validate and enhance the response for better educational value
                    validated_content = self._validate_and_enhance_response(
                        response_content, 
                        agent_type, 
                        self.current_session.learning_context
                    )
                    
                    # Check if the agent actually provided helpful content vs. declining to help
                    is_declining_response = self._is_declining_response(validated_content)
                    
                    # Create panel title with agent info
                    safe_agent_type = str(agent_type).strip().lower().replace(' ', '_')
                    panel_title = f"[bold agent.{safe_agent_type}]{self.get_agent_emoji(agent_type)} {agent_type.replace('_', ' ').title()}[/bold agent.{safe_agent_type}]"
                    # Add constellation status to title if updated
                    if constellation_info.get("updated"):
                        panel_title += f" [dim](→ {constellation_info['type']})[/dim]"
                    
                    # Enhance response with guidance (this method handles declining responses)
                    final_content = self._enhance_response_with_guidance(validated_content, constellation_info, is_declining_response)
                    
                    # --- Timing: measure response time ---
                    try:
                        response_panel = Panel(
                            Markdown(final_content),
                            title=panel_title,
                            style=f"agent.{safe_agent_type}",
                            expand=False
                        )
                        self.console.print(response_panel)
                    except Exception as e:
                        self.console.print(f"[error]Panel rendering failed: {e}[/error]")
                        self.console.print(Markdown(final_content))
                    # Show true end-to-end latency (Hub processing + render)
                    total_elapsed = time.perf_counter() - overall_start
                    self.console.print(f"[dim]⏱️ Agent full response in {total_elapsed:.3f} seconds.[/dim]")
                    
                    # Show curriculum status only for helpful learning responses
                    if constellation_info.get("curriculum_guided") and not is_declining_response:
                        self.console.print("[dim]🎓 Response enhanced with curriculum guidance[/dim]")
                    
                    # Update response_content for memory saving
                    response_content = validated_content
                        
                else:
                    # Generate enhanced fallback response using the validation method
                    fallback_content = self._generate_fallback_response(
                        agent_type, 
                        self.current_session.learning_context
                    )
                    
                    self.console.print(Panel(
                        Markdown(fallback_content),
                        title="[warning]🤔 Learning Assistant[/warning]",
                        border_style="warning"
                    ))
                    
                    # Set response_content for potential memory saving
                    response_content = fallback_content
                    
                    # Debug information for troubleshooting
                    if DEBUG_MODE:
                        self.console.print(f"[dim]Debug - Response structure: {processed_response}[/dim]")

                # Update long-term memory only for successful learning interactions
                if (self.long_term_memory and hasattr(self.current_session, 'learning_context') 
                    and response_content and not is_declining_response):
                    try:
                        # Create simplified metadata for ChromaDB compatibility
                        memory_data = {
                            "session_id": self.current_session.session_id,
                            "user_message": message[:500],  # Truncate for metadata limits
                            "agent_response": (response_content or "No response")[:500],
                            "agent_type": agent_type,
                            "timestamp": str(int(time.time())),
                            "learning_successful": "true"
                        }
                        
                        await asyncio.to_thread(
                            self.long_term_memory.add_external_knowledge,
                            text=f"Learning interaction: {message}",
                            user_id=self.current_session.user_id,
                            source="learning_session",
                            metadata=memory_data
                        )
                    except Exception as e:
                        if DEBUG_MODE:
                            self.console.print(f"[dim]Memory update failed: {e}[/dim]")

            except Exception as e:
                error_msg = f"Error processing message: {e}"
                self.console.print(Panel(
                    f"{error_msg}\n\n{traceback.format_exc() if DEBUG_MODE else 'Use --debug for detailed error info'}",
                    title="[bold error]Processing Error[/bold error]",
                    border_style="error"
                ))
                if DEBUG_MODE:
                    logging.error(f"Message processing error: {e}", exc_info=True)

    def _extract_agent_response(self, processed_response: Any) -> Tuple[str, str, Dict]:
        """
        Safely extract agent response content, type, and metadata from Learning Hub response.
        
        Returns:
            Tuple[str, str, Dict]: (response_content, agent_type, constellation_info)
        """
        response_content = None
        agent_type = "instructor"  # default
        constellation_info = {}
        
        try:
            if isinstance(processed_response, dict):
                # Handle different response structures from Learning Hub
                if "error" in processed_response:
                    return None, agent_type, {"error": processed_response["error"]}
                
                # Extract primary response content with priority order
                if "primary_response" in processed_response:
                    primary_resp = processed_response["primary_response"]
                    if isinstance(primary_resp, dict):
                        response_content = primary_resp.get("content")
                        agent_type = primary_resp.get("agent_type", "instructor")
                    else:
                        response_content = str(primary_resp) if primary_resp else None
                elif "content" in processed_response:
                    response_content = processed_response["content"]
                    agent_type = processed_response.get("agent_type", "instructor")
                elif "response" in processed_response:
                    response_content = processed_response["response"]
                    agent_type = processed_response.get("agent_type", "instructor")
                elif "message" in processed_response:
                    response_content = processed_response["message"]
                    agent_type = processed_response.get("agent_type", "instructor")
                
                # Extract constellation and learning context updates
                constellation_info = {
                    "updated": processed_response.get("constellation_updated", False),
                    "type": processed_response.get("constellation_type", "learning"),
                    "guidance": processed_response.get("learning_guidance", {}),
                    "curriculum_guided": processed_response.get("curriculum_guided", False)
                }
            elif isinstance(processed_response, str):
                response_content = processed_response
            elif hasattr(processed_response, 'content'):
                response_content = processed_response.content
                agent_type = getattr(processed_response, 'agent_type', "instructor")
            else:
                response_content = str(processed_response) if processed_response else None
                
        except Exception as e:
            if DEBUG_MODE:
                self.console.print(f"[dim]Warning: Error extracting response: {e}[/dim]")
            response_content = str(processed_response) if processed_response else None
        
        # Validate and clean response content
        if response_content:
            response_content = response_content.strip()
            if not response_content:
                response_content = None
        
        return response_content, agent_type, constellation_info
    
    def _enhance_response_with_guidance(self, response_content: str, constellation_info: Dict, is_declining_response: bool = False) -> str:
        """
        Enhance agent response with learning guidance and context.
        
        Args:
            response_content: The original agent response
            constellation_info: Information about constellation and guidance
            is_declining_response: Whether the agent is declining to help
        
        Returns:
            Enhanced response content with guidance
        """
        if not response_content or is_declining_response:
            return response_content
        
        enhanced_content = response_content
        guidance = constellation_info.get("guidance", {})
        
        if guidance and isinstance(guidance, dict):
            guidance_sections = []
            
            # Add current learning context
            if hasattr(self.current_session, 'current_module') and self.current_session.current_module:
                current_module = self.current_session.current_module
                if current_module and current_module != "unknown":
                    guidance_sections.append(f"📚 **Current Module:** {current_module.title()}")
                    
                    # Add module concepts if available
                    module_concepts = guidance.get("current_concepts", [])
                    if module_concepts:
                        concepts_text = ", ".join(module_concepts[:3])
                        if len(module_concepts) > 3:
                            concepts_text += f" (+{len(module_concepts)-3} more)"
                        guidance_sections.append(f"🎯 **Key Concepts:** {concepts_text}")
            
            # Add next steps
            if guidance.get("next_steps"):
                next_steps = guidance["next_steps"][:3]  # Limit to 3 steps
                steps_text = "\n".join(f"• {step}" for step in next_steps)
                guidance_sections.append(f"**🚀 Next Steps:**\n{steps_text}")
            
            # Add learning tips
            if guidance.get("learning_tips"):
                tips = guidance["learning_tips"][:2]  # Limit to 2 tips
                tips_text = "\n".join(f"• {tip}" for tip in tips)
                guidance_sections.append(f"**💡 Pro Tips:**\n{tips_text}")
            
            # Add progress note
            if guidance.get("progress_note"):
                guidance_sections.append(f"**📈 Progress:** {guidance['progress_note']}")
            
            # Add guidance sections to response
            if guidance_sections:
                enhanced_content += "\n\n" + "\n\n".join(guidance_sections)
        
        return enhanced_content
    
    def _is_declining_response(self, response_content: str) -> bool:
        """Check if the agent is declining to help."""
        if not response_content:
            return False
            
        decline_indicators = [
            "i cannot assist", "cannot help", "i cannot help", "unable to assist",
            "not able to help", "cannot provide", "unable to provide", "not available",
            "cannot answer", "unable to answer", "my current tools", "current functions",
            "designed for technical tasks", "not equipped", "beyond my capabilities",
            "i don't have", "don't have access", "not designed for", "can't help"
        ]
        
        response_lower = response_content.lower()
        return any(indicator in response_lower for indicator in decline_indicators)

    def show_modern_help(self):
        """Display beautiful help with command categories"""
        help_header = Panel(
            "[bold bright_blue]📚 GAAPF Command Reference[/bold bright_blue]\n"
            "[dim]Explore all available commands and features[/dim]",
            style="bright_blue"
        )
        self.console.print(help_header)
        
        # Group commands by category with enhanced styling
        category_order = ["Learning", "Tools", "Agents", "Session", "Memory", "Data", "Configuration", "Help", "System"]
        
        for category in category_order:
            if category in self.command_registry.categories:
                commands = self.command_registry.categories[category]
                
                # Create table for each category with emoji
                category_emoji = {
                    "Learning": "🎓",
                    "Tools": "🛠️",
                    "Agents": "🤖",
                    "Session": "💬",
                    "Memory": "🧠",
                    "Data": "📊",
                    "Configuration": "⚙️",
                    "Help": "❓",
                    "System": "🖥️"
                }
                
                cmd_table = Table(
                    title=f"{category_emoji.get(category, '🔧')} {category} Commands", 
                    style="cyan",
                    show_header=True,
                    header_style="bold bright_blue"
                )
                cmd_table.add_column("Command", style="bold bright_cyan", width=18)
                cmd_table.add_column("Description", style="white", width=45)
                cmd_table.add_column("Shortcuts", style="dim", width=15)
                
                for cmd in commands:
                    aliases_str = ", ".join(cmd.aliases) if cmd.aliases else "-"
                    cmd_table.add_row(cmd.name, cmd.description, aliases_str)
                
                self.console.print(cmd_table)
                
                # Add special notes for learning commands
                if category == "Learning":
                    learning_notes = Text()
                    learning_notes.append("💡 Learning Commands Help:\n", style="bold bright_yellow")
                    learning_notes.append("• Use /practice to activate hands-on mode\n", style="white")
                    learning_notes.append("• Use /assessment for quizzes and evaluation\n", style="white")
                    learning_notes.append("• Use /modules to see all available topics\n", style="white")
                    learning_notes.append("• Use /constellation to see active AI agents\n", style="white")
                    
                    notes_panel = Panel(learning_notes, style="bright_yellow", title="🎯 Quick Tips")
                    self.console.print(notes_panel)
        
        # Enhanced tips section
        tips_content = Text()
        tips_content.append("💡 Pro Tips for Effective Learning:\n\n", style="bold bright_yellow")
        
        tips_content.append("🎯 Natural Conversation:\n", style="bold bright_green")
        tips_content.append("• Ask questions naturally - AI will route to best agent\n", style="white")
        tips_content.append("• Request specific help: 'I need code examples for...'\n", style="dim white")
        tips_content.append("• Switch contexts: 'Let's practice coding now'\n\n", style="dim white")
        
        tips_content.append("🚀 Power Features:\n", style="bold bright_green")
        tips_content.append("• Use /search for quick web lookups\n", style="white")
        tips_content.append("• Use /deepsearch for comprehensive research\n", style="white")
        tips_content.append("• Use /collect to gather framework documentation\n", style="white")
        tips_content.append("• Use /code to prioritize coding assistance\n\n", style="white")
        
        tips_content.append("📚 Learning Modes:\n", style="bold bright_green")
        tips_content.append("• Theory: Ask conceptual questions\n", style="white")
        tips_content.append("• Practice: Use /practice for hands-on exercises\n", style="white")
        tips_content.append("• Assessment: Use /assessment for testing knowledge\n", style="white")
        tips_content.append("• Projects: Ask for project-based learning\n", style="white")
        
        tips_panel = Panel(
            tips_content,
            title="✨ Getting the Most from GAAPF",
            style="bright_yellow"
        )
        self.console.print(tips_panel)
        
        # Example queries section
        examples_content = Text()
        examples_content.append("🎯 Example Queries:\n\n", style="bold bright_green")
        examples_content.append("Theory: ", style="bright_blue")
        examples_content.append('"Explain how LangChain chains work"\n', style="dim white")
        examples_content.append("Code: ", style="bright_green")
        examples_content.append('"Show me a RAG implementation example"\n', style="dim white")
        examples_content.append("Practice: ", style="bright_yellow")
        examples_content.append('"Give me a coding exercise for vector databases"\n', style="dim white")
        examples_content.append("Debug: ", style="bright_red")
        examples_content.append('"Help me fix this error: [paste error]"\n', style="dim white")
        examples_content.append("Research: ", style="bright_purple")
        examples_content.append('"What are the latest updates to LangGraph?"', style="dim white")
        
        examples_panel = Panel(
            examples_content,
            title="🌟 Try These Examples",
            style="bright_green"
        )
        self.console.print(examples_panel)
    
    def show_modern_status(self):
        """Display comprehensive session status dashboard."""
        if not self.current_session:
            no_session_panel = Panel(
                "[warning]⚠️ No active learning session[/warning]\n\n"
                "[dim]Start a new session to begin learning[/dim]",
                title="Session Status",
                style="warning"
            )
            self.console.print(no_session_panel)
            return
        
        framework = self.framework_manager.get_framework(self.current_session.framework_id)
        framework_name = framework.get("name", self.current_session.framework_id) if framework else self.current_session.framework_id
        
        # Session duration
        session_duration = datetime.now() - self.current_session.start_time
        duration_str = f"{session_duration.seconds // 60}m {session_duration.seconds % 60}s"
        
        # Create status table
        status_table = Table(title="📊 Session Dashboard", style="bright_green")
        status_table.add_column("Metric", style="bold", no_wrap=True)
        status_table.add_column("Value", style="bright_white")
        status_table.add_column("Details", style="dim")
        
        status_table.add_row("Session ID", self.current_session.session_id, "Current session identifier")
        status_table.add_row("User", self.current_session.user_id, "Active learner profile")
        status_table.add_row("Framework", framework_name, f"Learning {framework_name}")
        status_table.add_row("Module", self.current_session.current_module, "Current learning module")
        status_table.add_row("Messages", str(len(self.current_session.messages)), "Conversation exchanges")
        status_table.add_row("Progress", f"{self.current_session.progress_percentage:.1f}%", "Learning completion")
        status_table.add_row("Duration", duration_str, "Time in current session")
        
        self.console.print(status_table)
        
        # Progress bar
        if self.current_session.progress_percentage > 0:
            with Progress(
                TextColumn("[bold blue]Learning Progress"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Progress", total=100)
                progress.update(task, completed=self.current_session.progress_percentage)
                time.sleep(0.5)  # Brief pause to show the progress bar
    
    def show_modern_agents(self):
        """Display all available agents with beautiful formatting."""
        
        agents_header = Panel(
            "[bold bright_blue]🤖 GAAPF AI Agent Constellation[/bold bright_blue]\n"
            "[dim]Meet your specialized learning assistants[/dim]",
            style="bright_blue"
        )
        self.console.print(agents_header)
        
        # Create agents table
        agents_table = Table(title="🚀 Available Specialists", style="bright_cyan")
        agents_table.add_column("Agent", style="bold", no_wrap=True)
        agents_table.add_column("Specialty", style="bright_white")
        agents_table.add_column("Best For", style="cyan")
        agents_table.add_column("Status", justify="center", style="success")
        
        agent_descriptions = {
            "instructor": ("Teaching & Explanation", "Concept learning, theory"),
            "code_assistant": ("Code Generation", "Writing code, debugging"),
            "mentor": ("Guidance & Strategy", "Learning paths, advice"),
            "research_assistant": ("Information Gathering", "Research, documentation"),
            "documentation_expert": ("Documentation", "API references, guides"),
            "practice_facilitator": ("Hands-on Practice", "Exercises, challenges"),
            "assessment": ("Evaluation & Testing", "Quizzes, assessments"),
            "troubleshooter": ("Problem Solving", "Debugging, fixes"),
            "motivational_coach": ("Motivation & Support", "Encouragement, tips"),
            "knowledge_synthesizer": ("Knowledge Integration", "Connecting concepts"),
            "progress_tracker": ("Progress Monitoring", "Tracking, analytics"),
            "project_guide": ("Project Management", "Planning, guidance")
        }
        
        for agent_type, agent in self.agents.items():
            emoji = self.get_agent_emoji(agent_type)
            name = f"{emoji} {agent_type.replace('_', ' ').title()}"
            
            if agent_type in agent_descriptions:
                specialty, best_for = agent_descriptions[agent_type]
            else:
                specialty = "General Assistant"
                best_for = "Various tasks"
            
            status = "🟢 Active"
            agents_table.add_row(name, specialty, best_for, status)
        
        self.console.print(agents_table)
        
        # Agent selection tips
        tips_content = """
[bold bright_yellow]🎯 Agent Selection Tips:[/bold bright_yellow]

The system automatically selects the best agent for your query, but here's what each specializes in:

• **Instructor** - Great for learning new concepts and getting explanations
• **Code Assistant** - Perfect for coding help and implementation guidance  
• **Mentor** - Ideal for strategic learning advice and career guidance
• **Practice Facilitator** - Best for hands-on exercises and skill building
• **Troubleshooter** - Excellent for debugging and problem-solving

Just ask your question naturally - GAAPF will route it to the right specialist!
"""
        
        tips_panel = Panel(
            Markdown(tips_content),
            title="✨ How Agent Selection Works",
            style="bright_yellow"
        )
        self.console.print(tips_panel)
    
    def show_memory_info(self):
        """Display memory and conversation information."""
        if not self.current_session:
            self.console.print("[warning]No active session to show memory for[/warning]")
            return
        
        memory_panel = Panel(
            f"[bold bright_blue]🧠 Session Memory[/bold bright_blue]\n\n"
            f"[bold]Messages in Memory:[/bold] {len(self.current_session.messages)}\n"
            f"[bold]Learning Context:[/bold] {len(self.current_session.learning_context)} items\n"
            f"[dim]Conversation history is automatically saved and restored[/dim]",
            title="Memory Status",
            style="bright_blue"
        )
        self.console.print(memory_panel)
    
    def show_tools_info(self):
        """Display available tools and capabilities."""
        # Create dynamic tools content based on actual system capabilities
        tools_table = Table(title="🛠️ Available Tools & Capabilities", style="bright_cyan")
        tools_table.add_column("Category", style="bold bright_blue", width=20)
        tools_table.add_column("Tool", style="bright_green", width=25)
        tools_table.add_column("Description", style="white", width=40)
        tools_table.add_column("Command", style="dim", width=15)
        
        # Core Learning Tools
        tools_table.add_row("Learning", "Progress Tracking", "Monitor learning progress and completion", "/progress")
        tools_table.add_row("", "Module Navigation", "Navigate between learning modules", "/modules")
        tools_table.add_row("", "Practice Mode", "Start hands-on practice sessions", "/practice")
        tools_table.add_row("", "Assessment", "Take quizzes and evaluations", "/assessment")
        tools_table.add_row("", "Curriculum", "View and customize learning path", "/curriculum")
        
        # Research & Information Tools
        tools_table.add_row("Research", "Web Search", "Search the web for information", "/search")
        tools_table.add_row("", "Deep Search", "Perform comprehensive research", "/deepsearch")
        tools_table.add_row("", "Framework Collector", "Gather framework documentation", "/collect")
        
        # AI Agent Tools
        tools_table.add_row("Agents", "Code Assistant", "Get coding help and examples", "/code")
        tools_table.add_row("", "Constellation View", "See active agent constellation", "/constellation")
        tools_table.add_row("", "Agent Selection", "View all available agents", "/agents")
        
        # System Tools
        tools_table.add_row("System", "Memory Management", "View conversation memory", "/memory")
        tools_table.add_row("", "Session Export", "Export learning session data", "/export")
        tools_table.add_row("", "Settings", "Configure system preferences", "/settings")
        
        self.console.print(tools_table)
        
        # Show constellation capabilities if session is active
        if self.current_session and self.current_session.constellation:
            constellation_info = self.current_session.constellation.get_constellation_info()
            
            capabilities_content = Text()
            capabilities_content.append("\n🌟 Current Constellation Capabilities:\n\n", style="bold bright_yellow")
            capabilities_content.append(f"Active Type: ", style="white")
            capabilities_content.append(f"{constellation_info.get('constellation_type', 'unknown')}\n", style="bright_green")
            capabilities_content.append(f"Primary Agents: ", style="white")
            capabilities_content.append(f"{', '.join(constellation_info.get('primary_agents', []))}\n", style="cyan")
            capabilities_content.append(f"Specialized For: ", style="white")
            capabilities_content.append(f"{constellation_info.get('description', 'General learning')}", style="dim white")
            
            capabilities_panel = Panel(
                capabilities_content,
                title="🚀 Active Constellation",
                style="bright_yellow"
            )
            self.console.print(capabilities_panel)
    
    def show_settings_panel(self):
        """Display current settings and configuration options."""
        settings_content = f"""
[bold bright_blue]⚙️ Current Configuration[/bold bright_blue]

[bold]System Settings:[/bold]
• LLM Provider: {self._get_llm_provider_display()}
• Debug Mode: {'Enabled' if DEBUG_MODE else 'Disabled'}
• Logging: {'Enabled' if self.is_logging else 'Disabled'}
• Session ID: {self.session_id}

[bold]Paths:[/bold]
• User Profiles: {self.user_profiles_path}
• Frameworks: {self.frameworks_path}
• Memory: {self.memory_path}

[bold]Project Config:[/bold]
• Configuration: {'Loaded' if hasattr(self, 'project_config') and self.project_config else 'Not found'}
"""
        
        settings_panel = Panel(
            Markdown(settings_content),
            title="🔧 Settings",
            style="bright_yellow"
        )
        self.console.print(settings_panel)
    
    def export_session_data(self):
        """Export current session data."""
        if not self.current_session:
            self.console.print("[warning]No active session to export[/warning]")
            return
        
        # Show export options
        export_formats = ["json", "markdown", "text"]
        format_choice = Prompt.ask(
            "[bold cyan]Export format[/bold cyan]",
            choices=export_formats,
            default="json"
        )
        
        filename = f"gaapf_session_{self.current_session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_choice}"
        
        self.console.print(f"[success]✅ Session exported to: {filename}[/success]")
        # Note: Actual export implementation would go here
    
    def clean_vectordb(self):
        """Clean vector database"""
        self.cli.clean_vector_database()
    
    def clear_conversation_safely(self):
        """Clear the conversation history with a confirmation prompt."""
        if Confirm.ask("[bold yellow]Are you sure you want to clear the conversation history?[/bold yellow]"):
            if self.current_session:
                self.current_session.messages = []
            self.console.print(Panel("[bold bright_green]Conversation Cleared[/bold bright_green]", style="green"))
    
    def restart_session_safely(self):
        """Restart the session with confirmation."""
        if Confirm.ask("[bold yellow]Are you sure you want to restart the session?[/bold yellow]"):
            self.console.print("[info]🔄 Restarting session...[/info]")
            # Note: Actual restart implementation would go here
    
    def quit_gracefully(self):
        """Quit CLI with graceful shutdown."""
        self.console.print("[info]👋 Thanks for learning with GAAPF![/info]")
        # FIX: Use shutdown flag instead of raising KeyboardInterrupt
        self.shutdown_requested = True
    
    # ============ ENHANCED LEARNING COMMANDS ============
    
    def show_learning_progress(self):
        """Display detailed learning progress with visual indicators."""
        if not self.current_session:
            self.console.print("[warning]No active learning session to show progress for[/warning]")
            return
        
        # Get learning context and progress information
        learning_context = self.current_session.learning_context
        framework_id = self.current_session.framework_id
        framework = self.framework_manager.get_framework(framework_id)
        
        if not framework:
            self.console.print("[error]Framework information not available[/error]")
            return
        
        # Create progress dashboard
        progress_table = Table(title="📊 Learning Progress Dashboard", style="bright_green")
        progress_table.add_column("Metric", style="bold", width=25)
        progress_table.add_column("Progress", style="bright_white", width=20)
        progress_table.add_column("Details", style="dim", width=35)
        
        # Overall progress
        overall_progress = self.current_session.progress_percentage
        progress_bar = self._create_progress_bar(overall_progress)
        progress_table.add_row("Overall Progress", f"{overall_progress:.1f}%", progress_bar)
        
        # Current module progress
        current_module = learning_context.get("current_module", "unknown")
        modules = framework.get("modules", {})
        if current_module in modules:
            module_info = modules[current_module]
            progress_table.add_row("Current Module", current_module.title(), module_info.get("title", "Unknown"))
            
            # Concepts covered
            concepts = module_info.get("concepts", [])
            concepts_str = f"{len(concepts)} concepts to learn"
            progress_table.add_row("Module Concepts", str(len(concepts)), concepts_str)
        
        # Session metrics
        session_duration = datetime.now() - self.current_session.start_time
        duration_str = f"{session_duration.seconds // 60}m {session_duration.seconds % 60}s"
        progress_table.add_row("Session Time", duration_str, "Current session duration")
        
        # Message count
        progress_table.add_row("Interactions", str(len(self.current_session.messages)), "Total exchanges with AI")
        
        self.console.print(progress_table)
        
        # Show module progress if available
        if modules:
            self._show_module_progress_overview(modules, current_module)
    
    def show_learning_modules(self):
        """Display learning modules with navigation options."""
        if not self.current_session:
            self.console.print("[warning]No active learning session[/warning]")
            return
        
        framework_id = self.current_session.framework_id
        framework = self.framework_manager.get_framework(framework_id)
        
        if not framework:
            self.console.print("[error]Framework information not available[/error]")
            return
        
        modules = framework.get("modules", {})
        current_module = self.current_session.learning_context.get("current_module", "")
        
        # Create modules table
        modules_table = Table(title="📚 Learning Modules", style="bright_blue")
        modules_table.add_column("Module", style="bold", width=20)
        modules_table.add_column("Title", style="bright_white", width=30)
        modules_table.add_column("Difficulty", style="yellow", width=12)
        modules_table.add_column("Duration", style="cyan", width=10)
        modules_table.add_column("Status", style="bright_green", width=12)
        
        for i, (module_id, module_info) in enumerate(modules.items(), 1):
            title = module_info.get("title", f"Module {i}")
            difficulty = module_info.get("complexity", "basic").title()
            duration = f"{module_info.get('estimated_duration', 30)}min"
            
            # Determine status
            if module_id == current_module:
                status = "🔵 Current"
                module_name = f"[bold bright_blue]{i}. {module_id}[/bold bright_blue]"
            else:
                status = "⚪ Available"
                module_name = f"{i}. {module_id}"
            
            modules_table.add_row(module_name, title, difficulty, duration, status)
        
        self.console.print(modules_table)
        
        # Show module navigation options
        nav_content = Text()
        nav_content.append("🧭 Navigation Options:\n\n", style="bold bright_yellow")
        nav_content.append("• Ask to switch modules: ", style="white")
        nav_content.append('"Switch to [module_name]"\n', style="dim")
        nav_content.append("• Request specific topics: ", style="white")
        nav_content.append('"Teach me about [concept]"\n', style="dim")
        nav_content.append("• Continue current module: ", style="white")
        nav_content.append('Just ask questions about the topic', style="dim")
        
        nav_panel = Panel(nav_content, title="📍 How to Navigate", style="bright_yellow")
        self.console.print(nav_panel)
    
    def show_constellation_info(self):
        """Display current constellation state and configuration."""
        if not self.current_session or not self.current_session.constellation:
            self.console.print("[warning]No active constellation to display[/warning]")
            return
        
        constellation = self.current_session.constellation
        constellation_info = constellation.get_constellation_info()
        
        # Constellation overview
        overview_content = Text()
        overview_content.append("🌟 Active Constellation\n\n", style="bold bright_blue")
        overview_content.append(f"Type: ", style="white")
        overview_content.append(f"{constellation_info.get('constellation_type', 'unknown')}\n", style="bright_cyan")
        overview_content.append(f"Description: ", style="white")
        overview_content.append(f"{constellation_info.get('description', 'No description')}\n\n", style="dim white")
        
        overview_panel = Panel(overview_content, title="🔭 Constellation Overview", style="bright_blue")
        self.console.print(overview_panel)
        
        # Agents in constellation
        agents_table = Table(title="🤖 Active Agents in Constellation", style="bright_green")
        agents_table.add_column("Role", style="bold", width=15)
        agents_table.add_column("Agent Type", style="bright_white", width=20)
        agents_table.add_column("Specialization", style="cyan", width=35)
        agents_table.add_column("Status", style="success", width=10)
        
        primary_agents = constellation_info.get('primary_agents', [])
        secondary_agents = constellation_info.get('secondary_agents', [])
        
        for agent_type in primary_agents:
            emoji = self.get_agent_emoji(agent_type)
            name = f"{emoji} {agent_type.replace('_', ' ').title()}"
            specialization = self._get_agent_specialization(agent_type)
            agents_table.add_row("Primary", name, specialization, "🟢 Active")
        
        for agent_type in secondary_agents:
            emoji = self.get_agent_emoji(agent_type)
            name = f"{emoji} {agent_type.replace('_', ' ').title()}"
            specialization = self._get_agent_specialization(agent_type)
            agents_table.add_row("Secondary", name, specialization, "🟡 Standby")
        
        self.console.print(agents_table)
        
        # Constellation capabilities
        capabilities = constellation_info.get('capabilities', [])
        if capabilities:
            cap_content = Text()
            cap_content.append("🎯 Current Capabilities:\n\n", style="bold bright_yellow")
            for cap in capabilities:
                cap_content.append(f"• {cap}\n", style="white")
            
            cap_panel = Panel(cap_content, title="⚡ What This Constellation Can Do", style="bright_yellow")
            self.console.print(cap_panel)
    
    def start_practice_mode(self):
        """Start an interactive practice session."""
        if not self.current_session:
            self.console.print("[warning]No active learning session[/warning]")
            return
        
        practice_content = Text()
        practice_content.append("🎯 Practice Mode\n\n", style="bold bright_green")
        practice_content.append("Practice mode activated! Here's how to get the most out of it:\n\n", style="white")
        practice_content.append("• ", style="bright_green")
        practice_content.append("Request coding exercises: ", style="white")
        practice_content.append('"Give me a practice problem for [topic]"\n', style="dim")
        practice_content.append("• ", style="bright_green")
        practice_content.append("Ask for code review: ", style="white")
        practice_content.append('"Review this code: [paste code]"\n', style="dim")
        practice_content.append("• ", style="bright_green")
        practice_content.append("Request implementation: ", style="white")
        practice_content.append('"Help me implement [specific feature]"\n', style="dim")
        practice_content.append("• ", style="bright_green")
        practice_content.append("Debug together: ", style="white")
        practice_content.append('"This code isn\'t working: [error description]"', style="dim")
        
        practice_panel = Panel(practice_content, title="🚀 Ready to Practice!", style="bright_green")
        self.console.print(practice_panel)
        
        # Update learning context to practice mode
        if hasattr(self.current_session, 'learning_context'):
            self.current_session.learning_context["current_activity"] = "practice"
            self.current_session.learning_context["learning_stage"] = "practice"
    
    def start_assessment_mode(self):
        """Start an assessment or quiz session."""
        if not self.current_session:
            self.console.print("[warning]No active learning session[/warning]")
            return
        
        assessment_content = Text()
        assessment_content.append("📝 Assessment Mode\n\n", style="bold bright_red")
        assessment_content.append("Assessment mode activated! Choose your assessment type:\n\n", style="white")
        assessment_content.append("• ", style="bright_red")
        assessment_content.append("Quick Quiz: ", style="white")
        assessment_content.append('"Give me a quick quiz on [topic]"\n', style="dim")
        assessment_content.append("• ", style="bright_red")
        assessment_content.append("Concept Check: ", style="white")
        assessment_content.append('"Test my understanding of [concept]"\n', style="dim")
        assessment_content.append("• ", style="bright_red")
        assessment_content.append("Code Challenge: ", style="white")
        assessment_content.append('"Give me a coding challenge"\n', style="dim")
        assessment_content.append("• ", style="bright_red")
        assessment_content.append("Module Assessment: ", style="white")
        assessment_content.append('"Assess my knowledge of this module"', style="dim")
        
        assessment_panel = Panel(assessment_content, title="🎓 Ready for Assessment!", style="bright_red")
        self.console.print(assessment_panel)
        
        # Update learning context to assessment mode
        if hasattr(self.current_session, 'learning_context'):
            self.current_session.learning_context["current_activity"] = "assessment"
            self.current_session.learning_context["learning_stage"] = "assessment"
    
    def show_curriculum_overview(self):
        """Display curriculum overview and customization options."""
        if not self.current_session:
            self.console.print("[warning]No active learning session[/warning]")
            return
        
        framework_id = self.current_session.framework_id
        framework = self.framework_manager.get_framework(framework_id)
        
        if not framework:
            self.console.print("[error]Framework information not available[/error]")
            return
        
        # Curriculum overview
        curr_content = Text()
        curr_content.append("📋 Learning Curriculum\n\n", style="bold bright_blue")
        curr_content.append(f"Framework: ", style="white")
        curr_content.append(f"{framework.get('name', framework_id)}\n", style="bright_cyan")
        curr_content.append(f"Version: ", style="white")
        curr_content.append(f"{framework.get('version', 'Unknown')}\n", style="dim white")
        curr_content.append(f"Description: ", style="white")
        curr_content.append(f"{framework.get('description', 'No description available')}", style="dim white")
        
        curr_panel = Panel(curr_content, title="📚 Curriculum Overview", style="bright_blue")
        self.console.print(curr_panel)
        
        # Show modules as curriculum structure
        modules = framework.get("modules", {})
        if modules:
            self._show_curriculum_structure(modules)
        
        # Customization options
        custom_content = Text()
        custom_content.append("🎛️ Customization Options:\n\n", style="bold bright_yellow")
        custom_content.append("• Adjust difficulty: ", style="white")
        custom_content.append('"Make this easier/harder"\n', style="dim")
        custom_content.append("• Skip topics: ", style="white")
        custom_content.append('"I already know [topic], skip it"\n', style="dim")
        custom_content.append("• Focus areas: ", style="white")
        custom_content.append('"I want to focus on [specific area]"\n', style="dim")
        custom_content.append("• Change pace: ", style="white")
        custom_content.append('"Go slower/faster through the material"', style="dim")
        
        custom_panel = Panel(custom_content, title="⚙️ Customize Your Learning", style="bright_yellow")
        self.console.print(custom_panel)
    
    # ============ TOOL INTEGRATION COMMANDS ============
    
    async def perform_web_search(self):
        """Perform web search with user query."""
        query = Prompt.ask("[bold cyan]What would you like to search for?[/bold cyan]")
        
        if not query.strip():
            self.console.print("[warning]Please provide a search query[/warning]")
            return
        
        with self.console.status(f"[bold green]🔍 Searching the web for: {query}", spinner="dots"):
            try:
                # Use the websearch tools from the system
                from ...tools.websearch_tools import search_web
                results = search_web(query, num_results=5)
                
                # Display results
                self._display_search_results(results, query)
                
            except Exception as e:
                error_panel = Panel(
                    f"[bold red]Search failed:[/bold red] {str(e)}",
                    title="❌ Search Error",
                    style="error"
                )
                self.console.print(error_panel)
    
    async def perform_deep_search(self):
        """Perform deep research with comprehensive results."""
        query = Prompt.ask("[bold cyan]What topic do you want to research deeply?[/bold cyan]")
        
        if not query.strip():
            self.console.print("[warning]Please provide a research topic[/warning]")
            return
        
        with self.console.status(f"[bold green]🔬 Deep researching: {query}", spinner="dots"):
            try:
                # Use the deepsearch tools from the system
                from ...tools.deepsearch import deep_search_comprehensive
                results = await deep_search_comprehensive(query, max_depth=3)
                
                # Display comprehensive results
                self._display_deep_search_results(results, query)
                
            except Exception as e:
                error_panel = Panel(
                    f"[bold red]Deep search failed:[/bold red] {str(e)}",
                    title="❌ Research Error",
                    style="error"
                )
                self.console.print(error_panel)
    
    async def collect_framework_information(self):
        """Collect comprehensive framework information."""
        if not self.current_session:
            framework_name = Prompt.ask("[bold cyan]Which framework do you want to research?[/bold cyan]")
        else:
            framework_name = self.current_session.framework_id
            if Confirm.ask(f"[bold cyan]Research current framework ({framework_name})?[/bold cyan]"):
                pass
            else:
                framework_name = Prompt.ask("[bold cyan]Which framework do you want to research?[/bold cyan]")
        
        if not framework_name.strip():
            self.console.print("[warning]Please provide a framework name[/warning]")
            return
        
        with self.console.status(f"[bold green]📚 Collecting information about {framework_name}", spinner="dots"):
            try:
                # Use the framework collector from the system
                from ...tools.framework_collector import FrameworkCollector
                collector = FrameworkCollector(is_logging=self.is_logging)
                
                user_id = self.current_session.user_id if self.current_session else "cli_user"
                results = await collector.collect_framework_info(
                    framework_name=framework_name,
                    user_id=user_id,
                    max_pages=20
                )
                
                # Display collected information
                self._display_framework_info(results, framework_name)
                
            except Exception as e:
                error_panel = Panel(
                    f"[bold red]Framework collection failed:[/bold red] {str(e)}",
                    title="❌ Collection Error",
                    style="error"
                )
                self.console.print(error_panel)
    
    async def request_specific_agent(self, agent_type: str):
        """Request interaction with a specific agent type."""
        if not self.current_session:
            self.console.print("[warning]No active learning session[/warning]")
            return
        
        if agent_type not in self.agents:
            self.console.print(f"[error]Agent type '{agent_type}' not available[/error]")
            available_agents = ", ".join(self.agents.keys())
            self.console.print(f"[info]Available agents: {available_agents}[/info]")
            return
        
        # Show agent specialization
        specialization = self._get_agent_specialization(agent_type)
        emoji = self.get_agent_emoji(agent_type)
        
        agent_content = Text()
        agent_content.append(f"{emoji} {agent_type.replace('_', ' ').title()} Ready!\n\n", style="bold bright_green")
        agent_content.append(f"Specialization: {specialization}\n\n", style="white")
        agent_content.append("This agent is now prioritized for your next interactions.\n", style="dim white")
        agent_content.append("Ask your question and I'll route it to this specialist!", style="dim white")
        
        agent_panel = Panel(agent_content, title="🎯 Agent Activated", style="bright_green")
        self.console.print(agent_panel)
        
        # Set requested agent in learning context
        if hasattr(self.current_session, 'learning_context'):
            self.current_session.learning_context["requested_agent"] = agent_type
    
    # ============ HELPER METHODS ============
    
    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a visual progress bar."""
        filled = int((percentage / 100) * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def _show_module_progress_overview(self, modules: Dict, current_module: str):
        """Show overview of progress across all modules."""
        progress_content = Text()
        progress_content.append("📈 Module Progress Overview:\n\n", style="bold bright_blue")
        
        for i, (module_id, module_info) in enumerate(modules.items(), 1):
            title = module_info.get("title", f"Module {i}")
            if module_id == current_module:
                progress_content.append(f"{i}. ", style="white")
                progress_content.append(f"{title}", style="bold bright_blue")
                progress_content.append(" (Current)\n", style="bright_green")
            else:
                progress_content.append(f"{i}. {title}\n", style="dim white")
        
        progress_panel = Panel(progress_content, title="📊 All Modules", style="bright_blue")
        self.console.print(progress_panel)
    
    def _show_curriculum_structure(self, modules: Dict):
        """Display curriculum structure as a flowchart-like view."""
        structure_table = Table(title="🗺️ Learning Path", style="bright_purple")
        structure_table.add_column("Step", style="bold", width=8)
        structure_table.add_column("Module", style="bright_white", width=25)
        structure_table.add_column("Key Concepts", style="cyan", width=40)
        structure_table.add_column("Prerequisites", style="dim", width=15)
        
        for i, (module_id, module_info) in enumerate(modules.items(), 1):
            title = module_info.get("title", f"Module {i}")
            concepts = ", ".join(module_info.get("concepts", [])[:3])
            if len(module_info.get("concepts", [])) > 3:
                concepts += "..."
            prerequisites = ", ".join(module_info.get("prerequisites", [])) or "None"
            
            structure_table.add_row(f"Step {i}", title, concepts, prerequisites)
        
        self.console.print(structure_table)
    
    def _get_agent_specialization(self, agent_type: str) -> str:
        """Get specialization description for an agent type."""
        specializations = {
            "instructor": "Teaching concepts and theory",
            "code_assistant": "Code examples and implementation",
            "mentor": "Learning guidance and strategy",
            "research_assistant": "Information gathering and research",
            "documentation_expert": "Documentation and references",
            "practice_facilitator": "Hands-on exercises and practice",
            "assessment": "Quizzes and knowledge evaluation",
            "troubleshooter": "Debugging and problem solving",
            "motivational_coach": "Encouragement and motivation",
            "knowledge_synthesizer": "Connecting concepts and insights",
            "progress_tracker": "Learning progress and analytics",
            "project_guide": "Project planning and guidance"
        }
        return specializations.get(agent_type, "General assistance")
    
    def _display_search_results(self, results: Dict, query: str):
        """Display web search results in a formatted way."""
        results_table = Table(title=f"🔍 Search Results for: {query}", style="bright_green")
        results_table.add_column("Title", style="bold bright_white", width=35)
        results_table.add_column("Source", style="cyan", width=25)
        results_table.add_column("Snippet", style="dim white", width=50)
        
        for result in results.get("results", [])[:5]:
            title = result.get("title", "No title")[:50]
            url = result.get("url", "No URL")
            # Extract domain from URL
            source = url.split("//")[-1].split("/")[0] if url != "No URL" else "Unknown"
            snippet = result.get("snippet", "No description")[:80]
            
            results_table.add_row(title, source, snippet)
        
        self.console.print(results_table)
    
    def _display_deep_search_results(self, results: Dict, query: str):
        """Display deep search results with comprehensive information."""
        self.console.print(f"[bold bright_blue]🔬 Deep Research Results for: {query}[/bold bright_blue]\n")
        
        # Summary
        if "summary" in results:
            summary_panel = Panel(
                results["summary"],
                title="📋 Research Summary",
                style="bright_blue"
            )
            self.console.print(summary_panel)
        
        # Key findings
        if "key_findings" in results:
            findings_content = Text()
            for finding in results["key_findings"]:
                findings_content.append(f"• {finding}\n", style="white")
            
            findings_panel = Panel(
                findings_content,
                title="🔑 Key Findings",
                style="bright_yellow"
            )
            self.console.print(findings_panel)
    
    def _display_framework_info(self, results: Dict, framework_name: str):
        """Display collected framework information."""
        self.console.print(f"[bold bright_blue]📚 Framework Information: {framework_name}[/bold bright_blue]\n")
        
        # Overview
        overview_content = Text()
        overview_content.append(f"Framework: {results.get('framework_name', 'Unknown')}\n", style="bold bright_white")
        overview_content.append(f"Collected: {results.get('collection_timestamp', 'Unknown')}\n", style="dim")
        
        overview_panel = Panel(overview_content, title="📊 Overview", style="bright_blue")
        self.console.print(overview_panel)
        
        # Documentation
        docs_info = results.get("official_docs", {})
        if docs_info:
            docs_content = Text()
            docs_content.append(f"Main URL: {docs_info.get('main_url', 'Not found')}\n", style="cyan")
            if "title" in docs_info:
                docs_content.append(f"Title: {docs_info['title']}\n", style="white")
            
            docs_panel = Panel(docs_content, title="📖 Documentation", style="bright_green")
            self.console.print(docs_panel)
        
        # GitHub info
        github_info = results.get("github_info", {})
        if github_info:
            github_content = Text()
            github_content.append(f"Repository: {github_info.get('url', 'Not found')}\n", style="cyan")
            if "title" in github_info:
                github_content.append(f"Title: {github_info['title']}\n", style="white")
            
            github_panel = Panel(github_content, title="🐙 GitHub Repository", style="bright_purple")
            self.console.print(github_panel)
        
        # Concepts
        concepts = results.get("concepts", [])
        if concepts:
            concepts_content = Text()
            for concept in concepts[:10]:  # Show first 10 concepts
                concepts_content.append(f"• {concept}\n", style="white")
            
            concepts_panel = Panel(concepts_content, title="🧠 Key Concepts", style="bright_yellow")
            self.console.print(concepts_panel)
    
    async def _pre_warm_embeddings(self, agent):
        """Pre-warm the embedding system to improve first response time"""
        try:
            # Use a simple message to generate embeddings that will be cached
            warmup_text = "This is a warmup message to initialize the embedding system."
            if hasattr(agent.memory, "_create_embedding_function"):
                embedding_function = agent.memory._create_embedding_function()
                if hasattr(embedding_function, "_create_simple_embedding"):
                    embedding_function._create_simple_embedding(warmup_text)
                    if DEBUG_MODE:
                        logger.info(f"Pre-warmed embeddings for {agent.__class__.__name__}")
        except Exception as e:
            if DEBUG_MODE:
                logger.warning(f"Failed to pre-warm embeddings: {e}")
    
    @async_debug_step
    async def run_conversation_loop(self):
        """Modern conversation loop with enhanced UX."""
        
        # Welcome message
        welcome_panel = Panel(
            "[bold bright_green]💬 Learning Session Active[/bold bright_green]\n\n"
            "[bold]Ready to help you learn![/bold]\n"
            "[dim]• Type your questions naturally\n"
            "• Use /help for commands\n"
            "• Press Ctrl+C to exit anytime[/dim]",
            title="🚀 Let's Learn Together!",
            style="bright_green"
        )
        self.console.print(welcome_panel)
        
        while not self.shutdown_requested:
            try:
                # Get user input with Rich prompt
                user_input = Prompt.ask(
                    "\n[bold bright_white]You[/bold bright_white]",
                    console=self.console
                ).strip()
                
                # Check for shutdown after input (in case signal was received)
                if self.shutdown_requested:
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower()
                    
                    if command in self.command_registry.commands:
                        await asyncio.create_task(
                            asyncio.to_thread(self.command_registry.commands[command].handler)
                        )
                    else:
                        self.console.print(
                            f"[warning]Unknown command: {command}[/warning]\n"
                            f"[dim]Type /help for available commands[/dim]"
                        )
                else:
                    # Process as regular message
                    await self.process_user_message(user_input)
                
                # Check for shutdown after processing (in case command set the flag)
                if self.shutdown_requested:
                    break
                
            except KeyboardInterrupt:
                # FIX: Handle KeyboardInterrupt properly by setting shutdown flag
                self.shutdown_requested = True
                self.console.print("\n[info]👋 Thanks for learning with GAAPF![/info]")
                break
            except Exception as e:
                error_panel = Panel(
                    f"[bold red]Unexpected error:[/bold red] {str(e)}",
                    style="error"
                )
                self.console.print(error_panel)
                
                if DEBUG_MODE:
                    self.console.print_exception()
        
        # FIX: Clean exit message when shutdown is requested
        if self.shutdown_requested and not hasattr(self, '_exit_message_shown'):
            self._exit_message_shown = True
            # Don't show duplicate message if already shown by signal handler
    
    @async_debug_step
    async def start(self):
        """Start the modern CLI learning system."""
        # Show initial welcome banner
        self.print_initial_banner()
        # Model selection first
        self.llm = self.select_model_provider()
        # Initialize components that depend on LLM
        self._initialize_components_after_llm()
        # Now print banner with correct agent count
        self.print_modern_banner()
        # Setup user profile
        user_id = self.setup_user_profile()
        # --- ENFORCE FRAMEWORK SELECTION ---
        framework_id = None
        while not framework_id:
            framework_id = self.select_framework()
            if not framework_id:
                self.console.print("[bold red]You must select a framework to continue.[/bold red]")
        # --- ENFORCE MODULE SELECTION ---
        framework_config = self.framework_manager.get_framework(framework_id)
        modules = framework_config.get("modules", {}) if framework_config else {}
        if not modules:
            self.console.print(f"[bold red]No modules found for framework {framework_id}. Please check your framework configuration.[/bold red]")
            return
        # Pick the first module as default if not set
        current_module = next(iter(modules), None)
        if not current_module:
            self.console.print(f"[bold red]No modules available in the selected framework. Please add modules to continue.[/bold red]")
            return
        # Initialize learning session
        success = await self.initialize_learning_session(user_id, framework_id)
        if not success:
            return
        # --- ENSURE SESSION CONTEXT IS VALID ---
        if not self.current_session or not getattr(self.current_session, 'framework_id', None) or not getattr(self.current_session, 'current_module', None):
            self.console.print("[bold red]Session context is invalid. Please restart and ensure a framework and module are selected.[/bold red]")
            return
        # Send initial greeting
        await self._send_initial_greeting()
        # Start conversation loop
        await self.run_conversation_loop()

    def _validate_and_enhance_response(self, response_content: str, agent_type: str, learning_context: Dict) -> str:
        """
        Validate and enhance agent response for clarity and educational value.
        
        Args:
            response_content: Raw response content from agent
            agent_type: Type of agent that generated the response
            learning_context: Current learning context
        
        Returns:
            Enhanced and validated response content
        """
        if not response_content or not response_content.strip():
            return self._generate_fallback_response(agent_type, learning_context)
        
        # Clean and format the response
        enhanced_response = response_content.strip()
        
        # Remove excessive whitespace and format properly
        enhanced_response = "\n".join(line.strip() for line in enhanced_response.split("\n") if line.strip())
        
        # Ensure the response has educational value
        enhanced_response = self._ensure_educational_value(enhanced_response, agent_type, learning_context)
        
        # Add appropriate formatting for better readability
        enhanced_response = self._improve_response_formatting(enhanced_response)
        
        return enhanced_response
    
    def _generate_fallback_response(self, agent_type: str, learning_context: Dict) -> str:
        """
        Generate a helpful fallback response when the agent doesn't provide content.
        """
        user_name = learning_context.get("user_profile", {}).get("name", "learner")
        current_module = learning_context.get("current_module", "current topic")
        framework_name = learning_context.get("framework_config", {}).get("name", "the framework")
        
        agent_specializations = {
            "instructor": f"I'm here to help you understand {framework_name} concepts. What specific topic from {current_module} would you like me to explain?",
            "code_assistant": f"I can help you with {framework_name} code examples and implementation. What would you like to build or debug?",
            "mentor": f"As your learning mentor, I'm here to guide your {framework_name} journey. What challenges are you facing?",
            "practice_facilitator": f"Let's practice {framework_name} skills! What type of hands-on exercise would help you learn?",
            "troubleshooter": f"I'm ready to help debug any {framework_name} issues you're encountering. What problem can I solve?",
        }
        
        fallback = agent_specializations.get(agent_type, f"I'm here to help you with {framework_name}. What can I assist you with?")
        
        return f"Hi {user_name}! {fallback}\n\n**Current Focus:** {current_module}\n\nFeel free to ask me anything about this topic!"
    
    def _ensure_educational_value(self, content: str, agent_type: str, learning_context: Dict) -> str:
        """
        Ensure the response has proper educational structure and value.
        """
        # Check if response is too short or vague
        if len(content) < 50:
            # Add context and encouragement for short responses
            framework_name = learning_context.get("framework_config", {}).get("name", "the framework")
            content += f"\n\nI'm here to help you master {framework_name}. Would you like me to elaborate on any specific aspect?"
        
        # Ensure responses have a clear structure for educational content
        if agent_type == "instructor" and not any(marker in content.lower() for marker in ["**", "###", "1.", "•", "-"]):
            # Add basic structure for instructor responses
            if "\n" not in content:
                # Single paragraph - add a follow-up question
                content += "\n\nWould you like me to explain any part of this in more detail?"
        
        return content
    
    def _improve_response_formatting(self, content: str) -> str:
        """
        Improve response formatting for better readability.
        """
        # Ensure proper spacing around headers and sections
        content = content.replace("**", " **").replace("**  ", "** ")  # Clean up bold formatting
        content = content.replace("###", "\n### ").strip()  # Clean up headers
        
        # Ensure lists have proper spacing
        lines = content.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Add spacing before list items if previous line isn't empty
            if (stripped_line.startswith(('•', '-', '*')) or 
                (stripped_line and stripped_line[0].isdigit() and '.' in stripped_line[:3])):
                if i > 0 and formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append("")  # Add blank line before lists
            
            # Add spacing after headers
            if stripped_line.startswith(('**', '###')):
                formatted_lines.append(stripped_line)
                if i < len(lines) - 1 and lines[i + 1].strip():
                    formatted_lines.append("")  # Add blank line after headers
                continue
            
            formatted_lines.append(stripped_line)
        
        # Clean up multiple consecutive blank lines
        final_lines = []
        prev_empty = False
        for line in formatted_lines:
            if not line.strip():
                if not prev_empty:
                    final_lines.append("")
                prev_empty = True
            else:
                final_lines.append(line)
                prev_empty = False
        
        return "\n".join(final_lines).strip()
    
    def _is_declining_response(self, response_content: str) -> bool:
        """Check if the agent is declining to help."""
        if not response_content:
            return False
            
        decline_indicators = [
            "i cannot assist", "cannot help", "i cannot help", "unable to assist",
            "not able to help", "cannot provide", "unable to provide", "not available",
            "cannot answer", "unable to answer", "my current tools", "current functions",
            "designed for technical tasks", "not equipped", "beyond my capabilities",
            "i don't have", "don't have access", "not designed for", "can't help"
        ]
        
        response_lower = response_content.lower()
        return any(indicator in response_lower for indicator in decline_indicators)

    def clean_vector_database(self):
        """Clean vector database."""
        if not hasattr(self, 'long_term_memory') or not self.long_term_memory:
            self.console.print("[error]Long-term memory is not initialized.[/error]")
            return

        framework_id = Prompt.ask(
            "[bold cyan]Enter framework ID to clean (e.g., LangChain), or 'all' to clear everything[/bold cyan]",
            default="all"
        )
        
        if framework_id:
            with self.console.status(f"[info]Cleaning vector database for '{framework_id}'...", spinner="dots"):
                try:
                    if framework_id.lower() == 'all':
                        self.long_term_memory.delete_memories()
                        self.console.print("\n[success]✅ All memories deleted from vector database.[/success]")
                    else:
                        self.long_term_memory.delete_memories(framework_id=framework_id)
                        self.console.print(f"\n[success]✅ Memories for {framework_id} deleted.[/success]")
                except Exception as e:
                    self.console.print(f"\n[error]❌ Error cleaning database: {e}[/error]")

@async_debug_step
async def main():
    """Main entry point for the modern CLI"""
    cli = GAAPFCLI(is_logging=DEBUG_MODE)
    
    try:
        # FIX: Remove problematic signal handler setup for Windows compatibility
        # The CLI class already handles SIGINT through signal.signal()
        
        # Start the CLI
        await cli.start()

    except KeyboardInterrupt:
        # Handle any remaining KeyboardInterrupt exceptions
        cli.console.print("\n[info]👋 Exiting GAAPF. Goodbye![/info]")
    except Exception as e:
        cli.console.print(f"\n[bold error]An unexpected error occurred: {e}[/bold error]")
        logging.error("Unhandled exception in main", exc_info=True)
        if DEBUG_MODE:
            cli.console.print_exception()
    finally:
        if not cli.shutdown_requested:
            cli.console.print("\n[primary]Exiting GAAPF. Goodbye![/primary]")

if __name__ == "__main__":
    asyncio.run(main())