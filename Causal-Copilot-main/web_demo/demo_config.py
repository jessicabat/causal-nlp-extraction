import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

@dataclass
class DemoConfig:
    """Configuration settings loaded from environment variables (.env).
    Values have sensible defaults when the corresponding variable is absent.
    """

    # Basic Flags
    demo_mode: bool = True  # Always run demo in demo mode unless overridden in code

    # Input / Output
    data_file: Optional[str] = os.getenv("DATA_FILE")  # Set when a file is uploaded or via env
    output_report_dir: str = os.getenv("OUTPUT_REPORT_DIR", "output_report")
    output_graph_dir: str = os.getenv("OUTPUT_GRAPH_DIR", "output_graph")

    # LLM Settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # 'openai' or 'ollama'
    model_name: str = os.getenv("LLM_MODEL", "llama3.2")     # e.g. 'llama2', 'llama3.2', 'mistral'
    apikey: str = os.getenv("OPENAI_API_KEY", "")             # Only used if llm_provider == 'openai'

    # Analysis Settings
    simulation_mode: str = os.getenv("SIMULATION_MODE", "offline")
    data_mode: str = os.getenv("DATA_MODE", "real")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    initial_query: Optional[str] = os.getenv("INITIAL_QUERY")  # Can be provided at runtime
    parallel: bool = os.getenv("PARALLEL", "True").lower() == "true"

    # Statistical Analysis Settings
    alpha: float = float(os.getenv("ALPHA", "0.1"))
    ratio: float = float(os.getenv("RATIO", "0.5"))
    num_test: int = int(os.getenv("NUM_TEST", "100"))

    def __post_init__(self):
        # Ensure output directories exist
        os.makedirs(self.output_report_dir, exist_ok=True)
        os.makedirs(self.output_graph_dir, exist_ok=True)


def get_demo_config() -> "DemoConfig":
    """Return a DemoConfig instance populated from environment variables."""
    return DemoConfig()
