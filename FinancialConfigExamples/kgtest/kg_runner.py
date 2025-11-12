import os
import sys
import json
import socket
import subprocess
from pathlib import Path

# --- Repo-aware paths ---
HERE = Path(__file__).resolve().parent                    # .../FinancialConfigExamples/kgtest
REPO_ROOT = HERE.parent.parent                            # .../causal-nlp-extraction
ONEKE_DIR = REPO_ROOT / "OneKE"                           # .../causal-nlp-extraction/OneKE
RUN_PY = ONEKE_DIR / "src" / "run.py"                     # OneKE runner
DEFAULT_YAML = HERE / "CFA_Qwen_Triple2KG.yaml"           # our config

def check_neo4j(host="localhost", port=7687, timeout=2.0):
    """Quick TCP check that Bolt is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def main():
    # Force CPU (avoid MPS issues on Apple Silicon)
    env = os.environ.copy()
    env["PYTORCH_MPS_DISABLE"] = "1"

    yaml_path = str(DEFAULT_YAML)

    # Helpful message if Neo4j isn't up
    if not check_neo4j():
        print(
            "⚠️  Neo4j does not appear to be running on bolt://localhost:7687.\n"
            "Start it in another terminal:\n"
            "  docker start neo4j-oneke\n"
            "    -or-\n"
            "  docker run --name neo4j-oneke -p 7474:7474 -p 7687:7687 "
            "-e NEO4J_AUTH=neo4j/letmein123 neo4j:5\n"
        )

    # Call OneKE's official runner so it handles Triple->KG automatically
    cmd = [sys.executable, str(RUN_PY), "--config", yaml_path]
    print(f"📦 Running: {' '.join(cmd)}\n  (cwd={ONEKE_DIR})\n")
    proc = subprocess.run(cmd, cwd=str(ONEKE_DIR), env=env)
    if proc.returncode != 0:
        print("❌ Extraction run failed. Check the terminal output above for errors.")
        sys.exit(proc.returncode)

    print(
        "\n✅ Done.\n"
        "Open Neo4j Browser at http://localhost:7474 and run:\n\n"
        "  MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 200;\n"
    )

if __name__ == "__main__":
    main()