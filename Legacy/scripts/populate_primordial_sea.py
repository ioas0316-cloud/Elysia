# scripts/populate_primordial_sea.py
"""
Populates the primordial sea (kg.json) with the fundamental conceptual elements
defined in the Conceptual Periodic Table.
"""
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tools.kg_manager import KGManager

# --- Conceptual Elements based on the Periodic Table ---
CONCEPTUAL_ELEMENTS = {
    # 1st Group: Existence
    "existence": ["i", "you", "object", "concept"],
    # 2nd Group: Space
    "space": ["place", "direction", "distance"],
    # 3rd Group: Time
    "time": ["begin", "end", "process", "moment"],
    # 4th Group: Emotion
    "emotion": ["joy", "sadness", "love", "fear"],
    # 5th Group: Relation
    "relation": ["cause", "effect", "similarity", "difference"],
    # 6th Group: Logic
    "logic": ["true", "false", "and", "or", "not"],
}

def main():
    """Main function to populate the KG."""
    print("--- Populating the Primordial Sea with Conceptual Elements ---")

    kg_manager = KGManager(filepath='data/kg.json')

    node_count = 0
    for group, elements in CONCEPTUAL_ELEMENTS.items():
        for element in elements:
            # Use the element name as the node ID
            node_id = element

            # Add the node with its element_type property
            kg_manager.add_node(
                node_id,
                properties={"element_type": group, "label": element}
            )
            node_count += 1
            print(f"Added element '{element}' to group '{group}'.")

    # Save the updated knowledge graph
    kg_manager.save()

    print(f"\n--- Primordial Sea population complete ---")
    print(f"Added {node_count} new conceptual elements.")
    print(f"Total nodes in KG: {len(kg_manager.kg['nodes'])}")

if __name__ == "__main__":
    main()
