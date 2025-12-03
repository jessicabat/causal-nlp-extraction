import os
import sys
import yaml
import json
import logging

# Add src directory to path for imports
sys.path.append("./src")
from models import *
from pipeline import *

logger = logging.getLogger("OneKE_FinancialRunner")
logger.setLevel(logging.INFO)

# File + console handlers
file_handler = logging.FileHandler("financial_pipeline.log", mode="w")
console_handler = logging.StreamHandler()

# Format logs
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

#MAIN LOGIC
CONFIG_DIR = "./FinancialConfigExamples"

def run_yaml_config(config_path):
    """Run one YAML configuration through the OneKE pipeline"""
    try:
        logger.info(f"Loading config: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load {config_path}: {e}")
        return

    # Model configuration
    try:
        model_info = config["model"]
        model_cls = globals().get(model_info["category"], None)
        if model_cls is None:
            raise ValueError(f"Unknown model category: {model_info['category']}")

        model = model_cls(
            model_name_or_path=model_info["model_name_or_path"],
            api_key=model_info.get("api_key"),
            base_url=model_info.get("base_url"),
        )
        pipeline = Pipeline(model)
        logger.info(f"Initialized model: {model_info['category']} ({model_info['model_name_or_path']})")
    except Exception as e:
        logger.exception(f"Model initialization failed for {config_path}: {e}")
        return

    # Extraction setup
    extraction = config["extraction"]
    task = extraction.get("task")
    instruction = extraction.get("instruction", "")
    use_file = extraction.get("use_file", False)
    file_path = extraction.get("file_path", "")
    mode = extraction.get("mode", "quick")
    update_case = extraction.get("update_case", False)
    show_trajectory = extraction.get("show_trajectory", False)

    # Run pipeline
    try:
        logger.info(f"Running extraction task='{task}' | mode='{mode}' | file='{file_path}'")
        result, trajectory, frontend_schema, frontend_res = pipeline.get_extraction_result(
            task=task,
            instruction=instruction,
            use_file=use_file,
            file_path=file_path,
            mode=mode,
            update_case=update_case,
            show_trajectory=show_trajectory,
        )

        # Save output
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(config_path).replace(".yaml", "_output.json"))
        with open(output_path, "w") as out_file:
            json.dump(result, out_file, indent=2)

        logger.info(f"Extraction complete. Results saved to: {output_path}")

    except Exception as e:
        logger.exception(f"Extraction failed for {config_path}: {e}")

def main():
    logger.info("Starting batch execution of financial YAML configs...")
    configs = [os.path.join(CONFIG_DIR, f) for f 
               in os.listdir(CONFIG_DIR) if f.endswith(".yaml")]

    if not configs:
        logger.warning("No YAML configuration files found in FinancialConfigExamples.")
        return

    for cfg in configs:
        run_yaml_config(cfg)

    logger.info("All YAML extractions completed successfully.")

if __name__ == "__main__":
    main()
