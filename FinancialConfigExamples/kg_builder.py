import json
import os
import pandas as pd

# ========= CONFIG ==========
JSON_PATH = "FinancialConfigExamples/triples/EarningsCall_triples.json"
OUTPUT_DIR = "FinancialConfigExamples/KnowledgeGraph/EarningsCall_triples"
# ===========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

triples = data.get("triple_list", [])

nodes = {}
edges = []

for t in triples:
    head = t["head"]
    head_type = t["head_type"]
    tail = t["tail"]
    tail_type = t["tail_type"]
    relation = t["relation"]
    relation_type = t["relation_type"]

    # Register nodes
    nodes[head] = head_type
    nodes[tail] = tail_type

    # Register relation
    edges.append({
        "source": head,
        "target": tail,
        "relation": relation,
        "relation_type": relation_type
    })

# Convert nodes → CSV
nodes_df = pd.DataFrame([
    {"id": name, "type": ntype} for name, ntype in nodes.items()
])

# Convert edges → CSV
edges_df = pd.DataFrame(edges)

# Save both files
nodes_df.to_csv(f"{OUTPUT_DIR}/nodes.csv", index=False)
edges_df.to_csv(f"{OUTPUT_DIR}/edges.csv", index=False)

print("Knowledge Graph CSVs created:")
print(f"- {OUTPUT_DIR}/nodes.csv")
print(f"- {OUTPUT_DIR}/edges.csv")
