"""
Knowledge Importer for Elysia

This script reads the foundational knowledge from 'data/knowledge_base.json'
and uses the KGManager to import it into the main knowledge graph ('data/kg.json'),
structuring the concepts according to the 'Principle of Flow'.
"""
import json
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing 'tools'
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tools.kg_manager import KGManager

def import_knowledge():
    """
    Reads the knowledge base and imports it into the main knowledge graph.
    """
    base_path = project_root / 'data' / 'knowledge_base.json'

    print(f"Reading foundational knowledge from {base_path}...")
    try:
        with open(base_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
    except FileNotFoundError:
        print(f"Error: Knowledge base file not found at {base_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {base_path}")
        return

    # Initialize KGManager
    kg_manager = KGManager()
    # Start with a clean slate to ensure a structured import
    kg_manager.kg = {"nodes": [], "edges": []}

    print("Importing knowledge into the graph...")

    # Place the ultimate "ocean" node manually at the center
    kg_manager.add_node("사랑", position={"x": 0, "y": 0, "z": 0})
    kg_manager.add_node("마음", position={"x": 0.1, "y": 0.1, "z": 0})


    for entry_group in knowledge_base.get("entries", []):
        print(f"  - Importing group: {entry_group.get('comment', 'Untitled')}")
        for item in entry_group.get("data", []):
            source = item.get("source")
            relation = item.get("relation")
            target = item.get("target")

            if not all([source, relation, target]):
                print(f"    - Skipping invalid item: {item}")
                continue

            # The add_edge function will automatically create and position nodes
            # based on their relationship, embodying the "Principle of Flow".
            kg_manager.add_edge(source, target, relation)

    # Save the newly populated knowledge graph
    kg_manager.save()
    print("Knowledge import complete.")
    summary = kg_manager.get_summary()
    print(f"Final KG state: {summary['nodes']} nodes, {summary['edges']} edges.")

if __name__ == '__main__':
    import_knowledge()
