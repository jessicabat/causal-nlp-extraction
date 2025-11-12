# # FinancialConfigExamples/kgtest/kg_runner_direct.py
# import os, sys, json, inspect, argparse, traceback

# # ---- Force CPU BEFORE importing torch/transformers/OneKE ----
# os.environ["PYTORCH_MPS_DISABLE"] = "1"
# os.environ["ACCELERATE_USE_MPS_DEVICE"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Repo root = three levels up from this file
# ROOT = os.path.abspath(os.path.join(__file__, "../../.."))
# ONEKE_SRC = os.path.join(ROOT, "OneKE", "src")
# if ONEKE_SRC not in sys.path:
#     sys.path.insert(0, ONEKE_SRC)

# # Make MPS look unavailable & force CPU for transformers load
# import torch
# try:
#     torch.backends.mps.is_available = lambda: False
# except Exception:
#     pass

# import transformers
# _real_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
# def _cpu_from_pretrained(*args, **kwargs):
#     # Always force CPU
#     kwargs["device_map"] = "cpu"
#     # Required by HF when device_map is passed
#     kwargs["low_cpu_mem_usage"] = True
#     # Safe dtype on CPU
#     kwargs.setdefault("torch_dtype", torch.float32)
#     return _real_from_pretrained(*args, **kwargs)

# transformers.AutoModelForCausalLM.from_pretrained = _cpu_from_pretrained  # type: ignore

# # Now import OneKE
# from pipeline import Pipeline
# from models import *  # brings Qwen/ChatGPT/etc.
# import yaml

# def _build_model(model_info: dict):
#     """Instantiate the model class with only the kwargs it supports."""
#     category = model_info["category"]             # e.g. "Qwen"
#     model_cls = globals()[category]               # class object
#     sig = inspect.signature(model_cls.__init__)
#     allowed = set(p.name for p in sig.parameters.values())

#     candidate_kwargs = {
#         "model_name_or_path": model_info.get("model_name_or_path"),
#         "api_key": model_info.get("api_key"),
#         "base_url": model_info.get("base_url"),
#         "vllm_serve": model_info.get("vllm_serve"),
#     }
#     safe_kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed and v is not None}

#     if not safe_kwargs and "model_name_or_path" in model_info:
#         return model_cls(model_info["model_name_or_path"])
#     return model_cls(**safe_kwargs)

# def run(config_path: str):
#     print(f"📄 Using config: {config_path}")
#     with open(config_path, "r") as f:
#         cfg = yaml.safe_load(f)

#     # Build model (local, free HF model; no API keys used)
#     model = _build_model(cfg["model"])
#     pipe = Pipeline(model)

#     extraction = cfg["extraction"]

#     # ---- Coerce optional fields to safe defaults ----
#     instruction   = extraction.get("instruction") or ""
#     text          = extraction.get("text") or ""   # we mainly use use_file
#     truth         = extraction.get("truth") or ""  # avoid TypeError in extract_json_dict
#     output_schema = extraction.get("output_schema") or ""
#     constraint    = extraction.get("constraint") or []

#     use_file = bool(extraction.get("use_file", False))
#     file_path = extraction.get("file_path")

#     # Resolve file_path relative to the repo root if not absolute
#     resolved_path = None
#     if use_file and file_path:
#         if not os.path.isabs(file_path):
#             candidate = os.path.join(ROOT, file_path)
#             resolved_path = candidate if os.path.exists(candidate) else os.path.abspath(file_path)
#         else:
#             resolved_path = file_path

#     if use_file:
#         print(f"📚 PDF path resolved to: {resolved_path}")

#     print("🚀 Starting extraction on CPU…")
#     result, trajectory, frontend_schema, frontend_res = pipe.get_extract_result(
#         task=extraction.get("task"),
#         instruction=instruction,
#         text=text,
#         use_file=use_file,
#         file_path=resolved_path,
#         output_schema=output_schema,
#         constraint=constraint,
#         mode=extraction.get("mode", "quick"),
#         update_case=extraction.get("update_case", False),
#         show_trajectory=extraction.get("show_trajectory", False),
#         truth=truth,
#     )

#     # Optional: save triples JSON
#     out_cfg = cfg.get("output", {})
#     if out_cfg.get("save_result"):
#         out_path = out_cfg.get("save_path", "kg_output.json")
#         os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
#         with open(out_path, "w") as f:
#             json.dump(result, f, indent=2)
#         print(f"✅ Saved triples JSON to: {out_path}")

#     # Print a tiny summary
#     try:
#         n = len(result.get("triple_list", [])) if isinstance(result, dict) else None
#         print(f"✔️  Extraction done. Triples found: {n if n is not None else 'unknown'}")
#     except Exception:
#         pass

#     # Sanity: show device
#     try:
#         print("🖥️  Model device:", next(model.model.parameters()).device)
#     except Exception:
#         pass

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", default=os.path.join(
#         ROOT, "FinancialConfigExamples", "kgtest", "CFA_Qwen_Triple2KG.yaml"
#     ))
#     args = parser.parse_args()
#     try:
#         run(args.config)
#     except Exception:
#         traceback.print_exc()
#         sys.exit(1)
# FinancialConfigExamples/kgtest/kg_runner_direct.py
import os, sys, json, inspect, argparse, traceback, re

# ---- Force CPU BEFORE importing torch/transformers/OneKE ----
os.environ["PYTORCH_MPS_DISABLE"] = "1"
os.environ["ACCELERATE_USE_MPS_DEVICE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Repo root = three levels up from this file
ROOT = os.path.abspath(os.path.join(__file__, "../../.."))
ONEKE_SRC = os.path.join(ROOT, "OneKE", "src")
if ONEKE_SRC not in sys.path:
    sys.path.insert(0, ONEKE_SRC)

# Make MPS look unavailable & force CPU for transformers load
import torch
try:
    torch.backends.mps.is_available = lambda: False
except Exception:
    pass

import transformers
_real_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
def _cpu_from_pretrained(*args, **kwargs):
    kwargs["device_map"] = "cpu"
    kwargs["low_cpu_mem_usage"] = True       # required if device_map is passed
    kwargs.setdefault("torch_dtype", torch.float32)
    return _real_from_pretrained(*args, **kwargs)
transformers.AutoModelForCausalLM.from_pretrained = _cpu_from_pretrained  # type: ignore

# Now import OneKE
from pipeline import Pipeline
from models import *  # brings Qwen/ChatGPT/etc.
import yaml

# --- Neo4j helpers (10–20 lines) ---
from neo4j import GraphDatabase
GENERIC = re.compile(r"^(a|an)\s+(dataset|metric|model|task|paper|organization|concept|experiment|market)$", re.I)

def _is_generic(s: str) -> bool:
    if not isinstance(s, str): return True
    t = s.strip()
    return (t == "") or bool(GENERIC.match(t))

def _rel_name(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return "RELATED_TO"
    return re.sub(r"[^A-Za-z0-9_]", "_", s.strip())

def _push_triple(tx, h, ht, r, rt, t, tt):
    head_label = ht if ht else "Entity"
    tail_label = tt if tt else "Entity"
    tx.run(
        f"""
        MERGE (h:{head_label} {{name:$h}})
        MERGE (t:{tail_label} {{name:$t}})
        MERGE (h)-[rel:{_rel_name(r)} {{relation_type:$rt}}]->(t)
        """,
        h=h, t=t, rt=(rt or "")
    )

def _build_model(model_info: dict):
    """Instantiate the model class with only the kwargs it supports."""
    category = model_info["category"]             # e.g. "Qwen"
    model_cls = globals()[category]               # class object
    sig = inspect.signature(model_cls.__init__)
    allowed = set(p.name for p in sig.parameters.values())

    candidate_kwargs = {
        "model_name_or_path": model_info.get("model_name_or_path"),
        "api_key": model_info.get("api_key"),
        "base_url": model_info.get("base_url"),
        "vllm_serve": model_info.get("vllm_serve"),
    }
    safe_kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed and v is not None}

    if not safe_kwargs and "model_name_or_path" in model_info:
        return model_cls(model_info["model_name_or_path"])
    return model_cls(**safe_kwargs)

def run(config_path: str):
    print(f"📄 Using config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Build model (local, free HF model; no API keys used)
    model = _build_model(cfg["model"])
    pipe = Pipeline(model)

    extraction = cfg["extraction"]

    # ---- Coerce optional fields to safe defaults ----
    instruction   = extraction.get("instruction") or ""
    text          = extraction.get("text") or ""   # we mainly use use_file
    truth         = extraction.get("truth") or ""  # avoid TypeError in extract_json_dict
    output_schema = extraction.get("output_schema") or ""
    constraint    = extraction.get("constraint") or []

    use_file = bool(extraction.get("use_file", False))
    file_path = extraction.get("file_path")

    # Resolve file_path relative to the repo root if not absolute
    resolved_path = None
    if use_file and file_path:
        if not os.path.isabs(file_path):
            candidate = os.path.join(ROOT, file_path)
            resolved_path = candidate if os.path.exists(candidate) else os.path.abspath(file_path)
        else:
            resolved_path = file_path

    if use_file:
        print(f"📚 PDF path resolved to: {resolved_path}")

    print("🚀 Starting extraction on CPU…")
    result, trajectory, frontend_schema, frontend_res = pipe.get_extract_result(
        task=extraction.get("task"),
        instruction=instruction,
        text=text,
        use_file=use_file,
        file_path=resolved_path,
        output_schema=output_schema,
        constraint=constraint,
        mode=extraction.get("mode", "quick"),
        update_case=extraction.get("update_case", False),
        show_trajectory=extraction.get("show_trajectory", False),
        truth=truth,
    )

    # Optional: save triples JSON
    out_cfg = cfg.get("output", {})
    out_path = None
    if out_cfg.get("save_result"):
        out_path = out_cfg.get("save_path", "kg_output.json")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Saved triples JSON to: {out_path}")

    # Print a tiny summary
    triples = result.get("triple_list", []) if isinstance(result, dict) else []
    print(f"✔️  Extraction done. Triples found: {len(triples)}")

    # --- Auto-push to Neo4j (uses YAML construct section for creds) ---
    neo = cfg.get("construct", {})
    if neo and neo.get("database", "").lower() == "neo4j":
        uri  = neo.get("url", "neo4j://localhost:7687")
        user = neo.get("username", "neo4j")
        pwd  = neo.get("password", "letmein123")

        # Light cleanup: drop generic stuff & broken rows
        cleaned = []
        for tr in triples:
            h  = tr.get("head");   t  = tr.get("tail");   r  = tr.get("relation")
            ht = tr.get("head_type"); tt = tr.get("tail_type"); rt = tr.get("relation_type")
            if not h or not t or not r: 
                continue
            if _is_generic(h) or _is_generic(t): 
                continue
            if isinstance(h, str) and isinstance(t, str) and h.strip().lower() == t.strip().lower():
                continue
            cleaned.append((h, ht, r, rt, t, tt))

        print(f"🧹 Pushing {len(cleaned)}/{len(triples)} cleaned triples to Neo4j…")
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        with driver.session() as session:
            for (h, ht, r, rt, t, tt) in cleaned:
                session.execute_write(_push_triple, h, ht, r, rt, t, tt)
        driver.close()
        print("✅ Neo4j push complete.")

    # Sanity: show device
    try:
        print("🖥️  Model device:", next(model.model.parameters()).device)
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(
        ROOT, "FinancialConfigExamples", "kgtest", "CFA_Qwen_Triple2KG.yaml"
    ))
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
