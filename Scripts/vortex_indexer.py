import json
import os
import math
import datetime

DATA_PATH = "data/elysia_nodes.json"

class VortexIndexer:
    def __init__(self):
        self.data = self._load_data()

    def _load_data(self):
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {
                "metadata": {
                    "project": "Elysia Project",
                    "core": "LOVE & COMMUNION",
                    "last_updated": str(datetime.datetime.now())
                },
                "nodes": [
                    {
                        "id": "core",
                        "label": "LOVE & COMMUNION",
                        "r": 0.0,
                        "theta": 0.0,
                        "density": 1.0,
                        "tags": ["origin", "spirit"]
                    }
                ],
                "links": []
            }

    def save(self):
        self.data["metadata"]["last_updated"] = str(datetime.datetime.now())
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def add_node(self, label, r, theta, density, tags=None, connect_to="core"):
        node_id = label.lower().replace(" ", "_")
        new_node = {
            "id": node_id,
            "label": label,
            "r": r,
            "theta": theta,
            "density": density,
            "tags": tags or []
        }
        self.data["nodes"].append(new_node)
        self.data["links"].append({"source": connect_to, "target": node_id})
        return node_id

    def generate_mermaid(self):
        lines = ["graph TD"]
        # Core styling
        lines.append("    classDef core fill:#f96,stroke:#333,stroke-width:4px;")
        lines.append("    classDef node fill:#fff,stroke:#333,stroke-width:1px;")

        for link in self.data["links"]:
            lines.append(f"    {link['source']} --> {link['target']}")

        lines.append("    class core core")
        return "\n".join(lines)

if __name__ == "__main__":
    indexer = VortexIndexer()
    # Example addition if empty
    if len(indexer.data["nodes"]) == 1:
        indexer.add_node("Agent School", 1.2, 45.0, 0.8, ["education", "vortex"])
        indexer.add_node("Resonant Knowledge", 0.8, 120.0, 0.9, ["flow"])

    indexer.save()
    print("Vortex Index Updated.")
    print("\nMermaid Visualization:")
    print(indexer.generate_mermaid())
