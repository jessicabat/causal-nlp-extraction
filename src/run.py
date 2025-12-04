import sys
import os

runner_dir = "causal-nlp-extraction/OneKE/src"
sys.path.append(runner_dir)

import src.run as run

if len(sys.argv) < 2:
    print("Usage: python run.py --config <config_path>")
    sys.exit(1)

config_path = sys.argv[1]

sys.argv = ["run.py", "--config", config_path]
run.main()
