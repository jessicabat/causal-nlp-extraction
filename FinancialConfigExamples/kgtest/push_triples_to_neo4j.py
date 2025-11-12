import json
from neo4j import GraphDatabase

BOLT_URI = "neo4j://localhost:7687"
USER = "neo4j"
PASSWORD = "letmein123"
JSON_PATH = "CFA_triples.json"

def norm(s):
    return s.strip() if isinstance(s, str) else s

def rel_name(s):
    # Neo4j rel types must be bare A–Z chars and underscores (no spaces)
    if not isinstance(s, str): return "RELATED_TO"
    return s.strip().replace(" ", "_").replace("-", "_")

def push_triple(tx, h, ht, r, rt, t, tt):
    # label by type if present, otherwise generic Entity
    head_label = ht if ht else "Entity"
    tail_label = tt if tt else "Entity"
    # MERGE nodes by (name, type) combo
    tx.run(
        f"""
        MERGE (h:{head_label} {{name:$h}})
        MERGE (t:{tail_label} {{name:$t}})
        MERGE (h)-[rel:{rel_name(r)} {{relation_type:$rt}}]->(t)
        """,
        h=h, t=t, rt=rt
    )

def main():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    triples = data.get("triple_list", [])

    driver = GraphDatabase.driver(BOLT_URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        count = 0
        for tr in triples:
            h  = norm(tr.get("head"))
            ht = norm(tr.get("head_type"))
            r  = norm(tr.get("relation"))
            rt = norm(tr.get("relation_type"))
            t  = norm(tr.get("tail"))
            tt = norm(tr.get("tail_type"))

            if not h or not t or not r:
                continue  # skip incomplete rows

            session.execute_write(push_triple, h, ht, r, rt, t, tt)
            count += 1

        print(f"Pushed {count} triples into Neo4j.")
    driver.close()

if __name__ == "__main__":
    main()
