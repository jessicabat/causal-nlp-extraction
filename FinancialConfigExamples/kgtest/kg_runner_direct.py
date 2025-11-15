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
    # Always force CPU
    kwargs["device_map"] = "cpu"
    kwargs["low_cpu_mem_usage"] = True       # required if device_map is passed
    kwargs.setdefault("torch_dtype", torch.float32)
    return _real_from_pretrained(*args, **kwargs)

transformers.AutoModelForCausalLM.from_pretrained = _cpu_from_pretrained  # type: ignore

# Now import OneKE
from pipeline import Pipeline
from models import *  # brings Qwen/ChatGPT/etc.
import yaml

# --- Neo4j helpers ---
from neo4j import GraphDatabase

# Generic placeholders we want to drop
GENERIC = re.compile(
    r"^(a|an|the)\s+(dataset|metric|model|task|paper|organization|concept|"
    r"experiment|market|segment|company|business)$",
    re.I,
)

def _is_generic(s: str) -> bool:
    if not isinstance(s, str):
        return True
    t = s.strip()
    return (t == "") or bool(GENERIC.match(t))

def _rel_name(s: str) -> str:
    """Safe Neo4j relationship type (uppercase + underscores)."""
    if not isinstance(s, str) or not s.strip():
        return "RELATED_TO"
    base = s.strip()
    base = re.sub(r"[^A-Za-z0-9]", "_", base)
    return base.upper()

def _label_name(label: str) -> str:
    """Safe Neo4j label name from head_type/tail_type."""
    if not label:
        return "Entity"
    lbl = re.sub(r"[^A-Za-z0-9]", "_", label.strip())
    if not lbl:
        return "Entity"
    # Capitalize first letter to stay neat
    return lbl[0].upper() + lbl[1:]

def _push_triple(tx, h, ht, r, rt, t, tt):
    head_label = _label_name(ht)
    tail_label = _label_name(tt)
    rel_type   = _rel_name(r)
    tx.run(
        f"""
        MERGE (h:{head_label} {{name:$h}})
        MERGE (t:{tail_label} {{name:$t}})
        MERGE (h)-[rel:{rel_type} {{relation_type:$rt}}]->(t)
        """,
        h=h, t=t, rt=(rt or ""),
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
    safe_kwargs = {k: v for k, v in candidate_kwargs.items()
                   if k in allowed and v is not None}

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
    text          = extraction.get("text") or ""   # we will override if we pre-read PDF
    truth         = extraction.get("truth") or ""
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

    # 🔹 NEW: optional local truncation of the PDF before calling OneKE
    max_pages = extraction.get("max_pages")
    max_chars = extraction.get("max_chars")
    if use_file and resolved_path and (max_pages or max_chars):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(resolved_path)
            pages_to_read = len(reader.pages)
            if isinstance(max_pages, int):
                pages_to_read = min(pages_to_read, max_pages)

            buf = []
            for i in range(pages_to_read):
                page = reader.pages[i]
                buf.append(page.extract_text() or "")
            text = "\n".join(buf)

            if isinstance(max_chars, int) and max_chars > 0:
                text = text[:max_chars]

            # Now we tell OneKE: "don't read the file, use this text instead"
            use_file = False
            resolved_path = None
            print(f"✂️  Truncated transcript to {pages_to_read} page(s), {len(text)} chars.")
        except Exception as e:
            print(f"⚠️  Failed to pre-read/truncate PDF: {e}")
            # fall back to original file behavior

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
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(
            ROOT,
            "FinancialConfigExamples",
            "kgtest",
            "CFA_Qwen_Triple2KG.yaml",
        ),
    )
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception:
        traceback.print_exc()
        sys.exit(1)