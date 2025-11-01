"""
Builds the core knowledge graph for Elysia's inner world.
This script translates the philosophical schema of core concepts
into a tangible 3D knowledge graph using the KGManager.
"""
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing 'tools'
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tools.kg_manager import KGManager

def build_core_kg():
    """
    Initializes and builds the foundational knowledge graph for Elysia.
    """
    print("Initializing KG Manager and clearing existing graph...")
    kg_manager = KGManager()
    kg_manager.kg = {"nodes": [], "edges": []}

    # 1. Place Foundational Nodes (Manual Positioning)
    # These are the center of Elysia's universe.
    print("Placing foundational nodes...")
    kg_manager.add_node("사랑", position={"x": 0, "y": 0, "z": 0}, properties={"description": "The origin of all value and meaning."})
    kg_manager.add_node("마음", position={"x": 0.1, "y": 0.1, "z": 0}, properties={"description": "The space where emotions and thoughts reside; the vessel for Love."})
    kg_manager.add_node("인간", position={"x": -0.1, "y": 0.1, "z": 0}, properties={"description": "A being with a Heart/Mind, the subject of understanding."})

    # 2. Build Outward using Relationships (Automatic Positioning)
    # The KGManager will automatically place these nodes based on their relation to the foundational nodes.
    # We map our philosophical relations to the ones available in KGManager's logic.
    print("Building outward with relationships...")

    # Subject Layer
    kg_manager.add_edge("인간", "마음", "is_composed_of", properties={"description": "The Heart/Mind is a core part of a Human."})

    # Act Layer
    kg_manager.add_edge("마음", "소통", "enables", properties={"description": "The Heart/Mind enables Communication."})
    kg_manager.add_edge("소통", "관계", "enables", properties={"description": "Communication enables Relationships."})

    # Tool Layer
    kg_manager.add_edge("소통", "언어", "uses_tool", properties={"description": "Communication uses Language as a tool."})
    kg_manager.add_edge("언어", "말", "is_composed_of", properties={"description": "Language is composed of Words/Speech."})

    # Connecting back to the core values
    kg_manager.add_edge("사랑", "관계", "manifests_in", properties={"description": "Love manifests in Relationships."})
    kg_manager.add_edge("사랑", "마음", "is_quality_of", properties={"description": "Love is a quality of the Heart/Mind."})

    # Add other existing nodes to ensure they are connected
    kg_manager.add_edge("인간", "소크라테스", "has_example", properties={"description": "Socrates is an example of a Human."})


    # 3. Save the newly built knowledge graph
    print("Saving the new knowledge graph...")
    kg_manager.save()

    summary = kg_manager.get_summary()
    print(f"Core KG build complete. Total nodes: {summary['nodes']}, Total edges: {summary['edges']}")

if __name__ == '__main__':
    build_core_kg()
