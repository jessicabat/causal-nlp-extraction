"""
Professional logging system for Causal-Copilot
Academic-oriented, clean, and informative console output
"""

import sys
import time
import logging
from enum import Enum
from typing import Optional, Any, Dict, List
from datetime import datetime
import json


class LogLevel(Enum):
    """Log levels for different types of messages"""
    TRACE = 0
    DEBUG = 1
    INFO = 2
    SUCCESS = 3
    SAVE = 4
    WARNING = 5
    ERROR = 6
    CRITICAL = 7


class Colors:
    """ANSI color codes for terminal output"""
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m' 
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


class Icons:
    """Unicode icons for different message types"""
    INFO = "ℹ"
    SUCCESS = "✓"
    SAVE = "💾"
    WARNING = "⚠"
    ERROR = "✗"
    CRITICAL = "🚨"
    DEBUG = "🔍"
    TRACE = "📍"
    ALGORITHM = "🧠"
    DATA = "📊"
    GRAPH = "📈"
    PROCESS = "⚙"
    RESEARCH = "🔬"
    EVALUATION = "📋"
    REPORT = "📄"
    TIME = "⏱"
    CHECKPOINT = "🏁"


class CausalLogger:
    """
    Professional logger for Causal-Copilot with academic styling
    """
    
    def __init__(self, name: str = "CausalCopilot", level: LogLevel = LogLevel.INFO, 
                 show_timestamp: bool = False, show_module: bool = False):
        self.name = name
        self.level = level
        self.show_timestamp = show_timestamp
        self.show_module = show_module
        self._start_time = time.time()
        
        # 屏蔽外部库的日志信息
        self._configure_external_loggers()
    
    def _configure_external_loggers(self):
        """Configure external libraries to reduce noise"""
        # 设置常见外部库的日志级别为WARNING或ERROR
        external_loggers = [
            'castle',
            'castle.backend',
            'castle.algorithms', 
            'httpx',
            'urllib3',
            'requests',
            'matplotlib',
            'sklearn',
            'numpy',
            'pandas',
            'torch',
            'tensorflow',
            'transformers',
            'openai',
            'tigramite'
        ]
        
        for logger_name in external_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
    def _format_timestamp(self) -> str:
        """Format current timestamp"""
        if not self.show_timestamp:
            return ""
        return f"{Colors.DIM}[{datetime.now().strftime('%H:%M:%S')}]{Colors.RESET} "
    
    def _format_level(self, level: LogLevel) -> str:
        """Format log level with appropriate color and icon"""
        level_configs = {
            LogLevel.TRACE: (Icons.TRACE, Colors.DIM, "TRACE"),
            LogLevel.DEBUG: (Icons.DEBUG, Colors.BRIGHT_BLACK, "DEBUG"),
            LogLevel.INFO: (Icons.INFO, Colors.BRIGHT_BLUE, "INFO"),
            LogLevel.SUCCESS: (Icons.SUCCESS, Colors.BRIGHT_GREEN, "SUCCESS"),
            LogLevel.SAVE: (Icons.SAVE, Colors.BRIGHT_CYAN, "SAVE"),
            LogLevel.WARNING: (Icons.WARNING, Colors.BRIGHT_YELLOW, "WARN"),
            LogLevel.ERROR: (Icons.ERROR, Colors.BRIGHT_RED, "ERROR"),
            LogLevel.CRITICAL: (Icons.CRITICAL, Colors.RED + Colors.BOLD, "CRITICAL")
        }
        
        icon, color, text = level_configs[level]
        return f"{color}{icon} {text}{Colors.RESET}"
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on current level"""
        return level.value >= self.level.value
    
    def _log(self, level: LogLevel, message: str, module: Optional[str] = None, 
             icon: Optional[str] = None, **kwargs):
        """Core logging method"""
        if not self._should_log(level):
            return
            
        # Build the log message
        parts = []
        
        # Timestamp
        parts.append(self._format_timestamp())
        
        # Level indicator
        parts.append(self._format_level(level))
        
        # Module name
        if module and self.show_module:
            parts.append(f"{Colors.DIM}[{module}]{Colors.RESET}")
        
        # Custom icon
        if icon:
            parts.append(f"{icon}")
        
        # Main message
        parts.append(message)
        
        # Join and print
        log_line = " ".join(filter(None, parts))
        print(log_line, **kwargs)
    
    # Core logging methods
    def trace(self, message: str, module: Optional[str] = None, **kwargs):
        """Log trace message (most verbose)"""
        self._log(LogLevel.TRACE, message, module, **kwargs)
    
    def debug(self, message: str, module: Optional[str] = None, **kwargs):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, module, **kwargs)
    
    def info(self, message: str, module: Optional[str] = None, **kwargs):
        """Log info message"""
        self._log(LogLevel.INFO, message, module, **kwargs)
    
    def success(self, message: str, module: Optional[str] = None, **kwargs):
        """Log success message"""
        self._log(LogLevel.SUCCESS, message, module, **kwargs)
    
    def save(self, message: str, file_path: Optional[str] = None, **kwargs):
        """Log file save operation"""
        if file_path:
            # Extract just the filename for cleaner display
            import os
            filename = os.path.basename(file_path)
            full_message = f"{message}: {filename}"
        else:
            full_message = message
        self._log(LogLevel.SAVE, full_message, **kwargs)
    
    def warning(self, message: str, module: Optional[str] = None, **kwargs):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, module, **kwargs)
    
    def error(self, message: str, module: Optional[str] = None, **kwargs):
        """Log error message"""
        self._log(LogLevel.ERROR, message, module, **kwargs)
    
    def critical(self, message: str, module: Optional[str] = None, **kwargs):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, module, **kwargs)
    
    # Domain-specific logging methods
    def algorithm(self, message: str, algorithm_name: Optional[str] = None, **kwargs):
        """Log algorithm-related message"""
        icon = f"{Icons.ALGORITHM}"
        if algorithm_name:
            message = f"{Colors.BOLD}{algorithm_name}{Colors.RESET}: {message}"
        self._log(LogLevel.INFO, message, "Algorithm", icon, **kwargs)
    
    def data(self, message: str, **kwargs):
        """Log data-related message"""
        self._log(LogLevel.INFO, message, "Data", Icons.DATA, **kwargs)
    
    def graph(self, message: str, **kwargs):
        """Log graph-related message"""
        self._log(LogLevel.INFO, message, "Graph", Icons.GRAPH, **kwargs)
    
    def evaluation(self, message: str, **kwargs):
        """Log evaluation-related message"""
        self._log(LogLevel.INFO, message, "Evaluation", Icons.EVALUATION, **kwargs)
    
    def research(self, message: str, **kwargs):
        """Log research-related message"""
        self._log(LogLevel.INFO, message, "Research", Icons.RESEARCH, **kwargs)
    
    def report(self, message: str, **kwargs):
        """Log report generation message"""
        self._log(LogLevel.INFO, message, "Report", Icons.REPORT, **kwargs)
    
    def process(self, message: str, **kwargs):
        """Log process-related message"""
        self._log(LogLevel.INFO, message, "Process", Icons.PROCESS, **kwargs)
    
    # Special formatting methods
    def header(self, title: str, width: int = 60, char: str = "="):
        """Print a clean header section"""
        if not self._should_log(LogLevel.INFO):
            return
            
        title_line = f"{char * 2} {title.upper()} {char * (width - len(title) - 6)}"
        print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}{title_line}{Colors.RESET}")
    
    def section(self, title: str, width: int = 60, char: str = "-"):
        """Print a section divider"""
        if not self._should_log(LogLevel.INFO):
            return
            
        section_line = f"{char * 3} {title} {char * (width - len(title) - 6)}"
        print(f"\n{Colors.BRIGHT_CYAN}{section_line}{Colors.RESET}")
    
    def subsection(self, title: str):
        """Print a subsection title"""
        if not self._should_log(LogLevel.INFO):
            return
        print(f"\n{Colors.CYAN}● {title}{Colors.RESET}")
    
    def progress(self, current: int, total: int, description: str = "", 
                bar_length: int = 30):
        """Display a progress bar"""
        if not self._should_log(LogLevel.INFO):
            return
            
        percentage = (current / total) * 100
        filled_length = int(bar_length * current // total)
        
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        progress_line = (f"\r{self._format_timestamp()}"
                        f"{Colors.BRIGHT_BLUE}⏳ PROGRESS{Colors.RESET} "
                        f"[{Colors.BRIGHT_GREEN}{bar}{Colors.RESET}] "
                        f"{percentage:.1f}% ({current}/{total})")
        
        if description:
            progress_line += f" {description}"
            
        print(progress_line, end="", flush=True)
        
        if current == total:
            print()  # New line when complete
    
    def metrics_table(self, metrics: Dict[str, Any], title: str = "Metrics"):
        """Display metrics in a formatted table"""
        if not self._should_log(LogLevel.INFO):
            return
            
        self.subsection(title)
        
        # Calculate column width
        max_key_len = max(len(str(k)) for k in metrics.keys()) if metrics else 0
        
        for key, value in metrics.items():
            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, (list, dict)):
                formatted_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            else:
                formatted_value = str(value)
            
            key_formatted = f"{key:<{max_key_len}}"
            print(f"  {Colors.DIM}{key_formatted}{Colors.RESET} : {Colors.BRIGHT_WHITE}{formatted_value}{Colors.RESET}")
    
    def results_summary(self, results: Dict[str, Any]):
        """Display results summary"""
        self.section("Results Summary", char="=")
        
        for category, data in results.items():
            if isinstance(data, dict):
                self.metrics_table(data, category)
            else:
                self.info(f"{category}: {data}")
    
    def elapsed_time(self, message: str = "Total elapsed time"):
        """Show elapsed time since logger creation"""
        elapsed = time.time() - self._start_time
        time_str = self._format_time(elapsed)
        self.info(f"{Icons.TIME} {message}: {Colors.BRIGHT_WHITE}{time_str}{Colors.RESET}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable time"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"
    
    def checkpoint(self, message: str):
        """Mark an important checkpoint"""
        self._log(LogLevel.SUCCESS, 
                 f"{Icons.CHECKPOINT} {Colors.BOLD}{message}{Colors.RESET}",
                 "Checkpoint")
    
    def status(self, message: str, value: Any = None, **kwargs):
        """Show clean status information"""
        if value is not None:
            if isinstance(value, (int, float)) and abs(value) < 1000:
                formatted_value = f"{Colors.BRIGHT_WHITE}{value}{Colors.RESET}"
            elif isinstance(value, str):
                # For Copilot responses, show full content with proper formatting
                if message == "Copilot" or "copilot" in message.lower():
                    # Show full content with line breaks preserved
                    formatted_value = f"{Colors.BRIGHT_WHITE}{value}{Colors.RESET}"
                elif len(value) < 50:
                    formatted_value = f"{Colors.BRIGHT_WHITE}{value}{Colors.RESET}"
                else:
                    # For other long strings, show first 100 chars with ellipsis
                    truncated = value[:100] + "..." if len(value) > 100 else value
                    formatted_value = f"{Colors.BRIGHT_WHITE}{truncated}{Colors.RESET}"
            else:
                # For complex values, show type info only
                formatted_value = f"{Colors.DIM}{type(value).__name__}({len(str(value))} chars){Colors.RESET}"
            message = f"{message}: {formatted_value}"
        self._log(LogLevel.INFO, message, **kwargs)
    
    def data_info(self, message: str, details: Dict[str, Any] = None, **kwargs):
        """Show detailed data information"""
        self._log(LogLevel.INFO, f"{Icons.DATA} {message}", "Data", **kwargs)
        if details:
            for key, value in details.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                else:
                    formatted_value = str(value)
                print(f"  {Colors.DIM}{key}:{Colors.RESET} {Colors.BRIGHT_WHITE}{formatted_value}{Colors.RESET}")
    
    def step(self, step_num: int, total_steps: int, description: str, **kwargs):
        """Show step progress"""
        progress = f"[{step_num}/{total_steps}]"
        self._log(LogLevel.INFO, f"{Colors.BRIGHT_CYAN}{progress}{Colors.RESET} {description}", **kwargs)
    
    def detail(self, message: str, **kwargs):
        """Show detailed processing information"""
        self._log(LogLevel.INFO, f"{Colors.DIM}→{Colors.RESET} {message}", **kwargs)


# Configure logging before creating logger instance
import os
import warnings

# 设置环境变量屏蔽一些库的输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow
warnings.filterwarnings('ignore', category=UserWarning)

# 在导入时就配置外部库日志
def _configure_global_loggers():
    """Configure external libraries to reduce noise globally"""
    # 设置常见外部库的日志级别为WARNING或ERROR
    external_loggers = [
        'castle',
        'castle.backend',
        'castle.algorithms', 
        'httpx',
        'urllib3',
        'requests',
        'matplotlib',
        'sklearn',
        'numpy',
        'pandas',
        'torch',
        'tensorflow',
        'transformers',
        'openai',
        'tigramite'
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

# 初始化时就配置
_configure_global_loggers()

# Global logger instance
logger = CausalLogger()

# Convenience functions for common use cases
def set_log_level(level: LogLevel):
    """Set the global log level"""
    logger.level = level

def set_quiet_mode():
    """Set to quiet mode (only warnings and errors)"""
    set_log_level(LogLevel.WARNING)

def set_verbose_mode():
    """Set to verbose mode (all messages)"""
    set_log_level(LogLevel.TRACE)

def set_normal_mode():
    """Set to normal mode (info and above)"""
    set_log_level(LogLevel.INFO)


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    logger = CausalLogger("TestLogger")
    
    logger.header("Causal-Copilot Analysis Session")
    
    logger.info("System initialized successfully")
    logger.data("Loading dataset: sachs.csv (11 variables, 7466 samples)")
    logger.algorithm("Selected algorithm", "PC Algorithm")
    logger.success("Data preprocessing completed")
    logger.warning("Some variables contain missing values")
    
    logger.section("Algorithm Execution")
    
    # Simulate progress
    for i in range(0, 101, 20):
        time.sleep(0.1)
        logger.progress(i, 100, "Running PC Algorithm")
    
    logger.graph("Causal graph discovered (15 edges, 0 cycles)")
    
    # Show metrics
    metrics = {
        "Structural Hamming Distance": 12.5,
        "Precision": 0.8234,
        "Recall": 0.7651,
        "F1-Score": 0.7931
    }
    logger.metrics_table(metrics, "Performance Metrics")
    
    logger.checkpoint("Analysis completed successfully")
    logger.elapsed_time()
    
    logger.error("This is an error message")
    logger.critical("This is a critical message")