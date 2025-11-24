import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import sys
import json
import inspect
from pathlib import Path

import yaml

# ---------- Paths & imports ----------

ROOT = Path(__file__).resolve().parent        # e.g. /home/.../causal-nlp-extraction
ONEKE_SRC = ROOT / "OneKE" / "src"

if str(ONEKE_SRC) not in sys.path:
    sys.path.insert(0, str(ONEKE_SRC))

from pipeline import Pipeline
from models import *      # Qwen, DeepSeek, etc.
from neo4j import GraphDatabase


def build_model(model_info: dict):
    """
    Instantiate the correct model class (Qwen, DeepSeek, etc.)
    using only kwargs its __init__ actually supports.
    """
    category = model_info["category"]   # e.g. "Qwen"
    model_cls = globals()[category]     # class object from models.py

    sig = inspect.signature(model_cls.__init__)
    allowed = set(p.name for p in sig.parameters.values())

    candidate_kwargs = {
        "model_name_or_path": model_info.get("model_name_or_path"),
        "api_key": model_info.get("api_key"),
        "base_url": model_info.get("base_url"),
        "vllm_serve": model_info.get("vllm_serve"),
        "device": model_info.get("device"),
    }
    safe_kwargs = {
        k: v for k, v in candidate_kwargs.items()
        if k in allowed and v not in ("", None)
    }

    # fallback: some classes only take model_name_or_path positionally
    if not safe_kwargs and model_info.get("model_name_or_path"):
        return model_cls(model_info["model_name_or_path"])

    print("🧠 Model init kwargs:", safe_kwargs)
    return model_cls(**safe_kwargs)


# ---------- Neo4j helpers ----------

def norm(s):
    return s.strip() if isinstance(s, str) else None


def rel_name(s):
    """Neo4j relationship type: only A–Z, 0–9, underscore."""
    if not isinstance(s, str):
        return "RELATED_TO"
    import re
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", s.strip())
    return cleaned or "RELATED_TO"


def push_triple(tx, h, ht, r, rt, t, tt):
    head_label = ht or "Entity"
    tail_label = tt or "Entity"
    tx.run(
        f"""
        MERGE (h:{head_label} {{name:$h}})
        MERGE (t:{tail_label} {{name:$t}})
        MERGE (h)-[rel:{rel_name(r)} {{relation_type:$rt}}]->(t)
        """,
        h=h, t=t, rt=rt or "",
    )


def is_triple_dict(d: dict) -> bool:
    """Heuristic: does this dict look like a single Triple?"""
    if not isinstance(d, dict):
        return False
    return all(k in d for k in ("head", "relation", "tail"))


# ---------- Main runner ----------

def main(config_override: str = None):
    # --- Config path resolution ---
    if config_override:
        cfg_path = Path(config_override)
    else:
        cfg_path = ROOT / "FinancialConfigExamples" / "kgtest" / "FinancialNews_Triple2KG.yaml"

    print(f"📄 Loading config: {cfg_path}")

    if not cfg_path.exists():
        print("❌ Config file not found!")
        return

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_info = cfg["model"]
    extraction = cfg["extraction"]
    construct = cfg.get("construct", {})
    output_cfg = cfg.get("output", {})

    # --- Build the model ---
    print("🧠 Building model…")
    print("   category:", model_info.get("category"))
    print("   model_name_or_path:", model_info.get("model_name_or_path"))
    model = build_model(model_info)
    pipe = Pipeline(model)

    # --- Resolve file path (PDF or TXT both work) ---
    use_file = extraction.get("use_file", False)
    file_path = extraction.get("file_path")
    if use_file and file_path:
        doc_path = (ROOT / file_path).resolve()
        print("📚 Using file:", doc_path)
        if not doc_path.exists():
            print("❌ File does not exist at that path!")
            return
    else:
        print("❌ extraction.use_file is False or no file_path set.")
        return

    # --- Run extraction ---
    print("🚀 Starting extraction…")

    truth = extraction.get("truth") or ""   # must be a string

    result, trajectory, frontend_schema, frontend_res = pipe.get_extract_result(
        task=extraction.get("task"),
        instruction=extraction.get("instruction") or "",
        text=extraction.get("text"),              # usually None when using files
        use_file=True,
        file_path=str(doc_path),
        output_schema=extraction.get("output_schema", ""),
        constraint=extraction.get("constraint"),
        mode=extraction.get("mode", "quick"),
        update_case=extraction.get("update_case", False),
        show_trajectory=extraction.get("show_trajectory", False),
        truth=truth,
    )

    print("✅ Model call finished, normalizing triples…")

    # --- Normalize triples robustly from result / frontend_res ---
    triples = []

    # 1) If result is a dict with a real list under "triple_list"
    if isinstance(result, dict):
        tl = result.get("triple_list")
        if isinstance(tl, list):
            triples = tl
        elif is_triple_dict(result):
            # Single triple dict
            triples = [result]

    # 2) If we didn't get anything yet, check frontend_res
    if not triples and isinstance(frontend_res, dict):
        tl = frontend_res.get("triple_list")
        if isinstance(tl, list):
            triples = tl
        else:
            # Sometimes nested under "extraction_result"
            ex_res = frontend_res.get("extraction_result")
            if isinstance(ex_res, dict):
                tl2 = ex_res.get("triple_list")
                if isinstance(tl2, list):
                    triples = tl2
                elif is_triple_dict(ex_res):
                    triples = [ex_res]
            elif is_triple_dict(frontend_res):
                triples = [frontend_res]

    # 3) If result itself is already a list of triples
    if not triples and isinstance(result, list):
        # Keep only triple-shaped dicts
        triples = [d for d in result if is_triple_dict(d)]

    print(f"✅ Extraction complete. Triples found: {len(triples)}")

    if triples:
        print("🔎 First 5 triples (if available):")
        for tr in triples[:5]:
            print("   -", tr)
    else:
        print("⚠️ No actual triple list found. It looks like we may have gotten only a schema or empty output.")
        # Continue anyway so JSON save still happens for debugging

    # --- Save JSON if configured ---
    out_path = output_cfg.get("save_path")
    if out_path:
        out_full = (ROOT / out_path).resolve()
        out_full.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "triples": triples,
            "raw_result": result,
            "frontend_res": frontend_res,
        }

        try:
            with open(out_full, "w") as f:
                json.dump(payload, f, indent=2)
            print("💾 Saved triples JSON to:", out_full)
        except TypeError as e:
            print("⚠️ json.dump failed, falling back to string serialization:", e)
            with open(out_full, "w") as f:
                f.write(json.dumps(str(payload), indent=2))
            print("💾 Saved (stringified) payload to:", out_full)
    else:
        print("ℹ️ No output.save_path configured; skipping JSON save.")

    # --- Push to Neo4j if construct is set and we actually have triples ---
    if construct and construct.get("database", "").lower() == "neo4j":
        if not triples:
            print("⚠️ Neo4j configured but no triples found; skipping KG push.")
        else:
            uri = construct.get("url")
            user = construct.get("username")
            # Prefer env var for password; fallback to YAML if present
            pwd = construct.get("password") or os.environ.get("NEO4J_PASSWORD_NEWS", "")

            print(f"🌐 Connecting to Neo4j at {uri} as {user}…")
            driver = GraphDatabase.driver(uri, auth=(user, pwd))

            pushed = 0
            with driver.session() as session:
                for tr in triples:
                    if not isinstance(tr, dict):
                        continue
                    h = norm(tr.get("head"))
                    t = norm(tr.get("tail"))
                    r = norm(tr.get("relation"))
                    ht = norm(tr.get("head_type")) or None
                    tt = norm(tr.get("tail_type")) or None
                    rt = norm(tr.get("relation_type")) or None

                    if not h or not t or not r:
                        continue

                    session.execute_write(push_triple, h, ht, r, rt, t, tt)
                    pushed += 1

            driver.close()
            print(f"✅ Neo4j push complete. Pushed {pushed} triples.")
    else:
        print("ℹ️ No Neo4j construct section found; skipping KG push.")

    # --- Show model device (optional) ---
    try:
        import torch
        dev = next(model.model.parameters()).device
        print("🖥️ Model device:", dev)
    except Exception as e:
        print("ℹ️ Could not inspect model device:", e)


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg_arg)
