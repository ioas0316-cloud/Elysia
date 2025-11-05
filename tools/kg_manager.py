"""
Knowledge Graph Manager for a 3D Conceptual Space

Manages a knowledge graph where each node has a 3D position,
creating a spatial representation of concepts and their relationships.
This version supports a property graph model where edges can have attributes.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

# --- Default Path Configuration ---
# The primary knowledge graph for Elysia's core brain.
# All modules should default to this unless a specific KG is required.
DATA_DIR = Path("data")
DEFAULT_KG_PATH = DATA_DIR / 'kg.json'

class KGManager:
    def __init__(self, filepath: Optional[Path] = None):
        """
        Initializes the KGManager.
        It will use the provided filepath or fall back to the project-wide default KG path.
        """
        self.filepath = filepath if filepath else DEFAULT_KG_PATH
        self.filepath.parent.mkdir(exist_ok=True)
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self.kg = json.load(f)
            except (json.JSONDecodeError, ): # Handle case where file is empty or corrupt
                self.kg = {"nodes": [], "edges": []}
        else:
            self.kg = {"nodes": [], "edges": []}

    def save_kg(self):
        """Saves the current state of the knowledge graph to its file."""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.kg, f, ensure_ascii=False, indent=2)

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Finds a node by its ID."""
        for node in self.kg.get('nodes', []):
            if node['id'] == node_id:
                return node
        return None

    def add_node(self, node_id: str, position: Optional[Dict] = None, properties: Optional[Dict] = None) -> Dict:
        """Adds a new node. If it exists, returns the existing node."""
        existing_node = self.get_node(node_id)
        if existing_node:
            if properties:
                existing_node.update(properties)
            return existing_node

        new_node = {
            "id": node_id,
            "position": position if position else {"x": random.uniform(-1, 1), "y": random.uniform(-1, 1), "z": random.uniform(-1, 1)},
            "activation_energy": 0.0
        }
        if properties:
            new_node.update(properties)

        self.kg['nodes'].append(new_node)
        return new_node

    def add_or_update_edge(self, source_id: str, target_id: str, relation: str, properties: Optional[Dict[str, Any]] = None):
        """
        Adds or updates a directional edge with optional properties.
        """
        # Check if an edge with the same source, target, and relation exists
        for edge in self.kg.get('edges', []):
            if edge['source'] == source_id and edge['target'] == target_id and edge['relation'] == relation:
                if properties:
                    edge.update(properties)
                return

        # If no such edge exists, create a new one
        new_edge = {"source": source_id, "target": target_id, "relation": relation}
        if properties:
            new_edge.update(properties)
        self.kg['edges'].append(new_edge)


    def find_causes(self, target_id: str) -> List[Dict[str, Any]]:
        """Finds all causes for a given target node."""
        causes = []
        for edge in self.kg.get('edges', []):
            if edge.get('target') == target_id and edge.get('relation') == 'causes':
                causes.append(edge)
        return causes

    def find_effects(self, source_id: str) -> List[Dict[str, Any]]:
        """Finds all effects of a given source node."""
        effects = []
        for edge in self.kg.get('edges', []):
            if edge.get('source') == source_id and edge.get('relation') == 'causes':
                effects.append(edge)
        return effects

    def get_summary(self):
        return {"nodes": len(self.kg['nodes']), "edges": len(self.kg['edges'])}

if __name__ == '__main__':
    kg_manager = KGManager() # This will now use kg.json by default
    kg_manager.kg = {"nodes": [], "edges": []}

    # Example of a causal chain
    kg_manager.add_or_update_edge("rain", "wet_ground", "causes", properties={"strength": 0.9})
    kg_manager.add_or_update_edge("wet_ground", "slippery_surface", "causes", properties={"strength": 0.7})

    kg_manager.save_kg()

    print('KG summary:', kg_manager.get_summary())
    print("KG rebuilt with causal query functions.")

    # Test the new query functions
    causes_of_wet_ground = kg_manager.find_causes("wet_ground")
    print("\nCauses of 'wet_ground':")
    for cause in causes_of_wet_ground:
        print(f"- Source: {cause['source']}, Strength: {cause.get('strength', 'N/A')}")

    effects_of_wet_ground = kg_manager.find_effects("wet_ground")
    print("\nEffects of 'wet_ground':")
    for effect in effects_of_wet_ground:
        print(f"- Target: {effect['target']}, Strength: {effect.get('strength', 'N/A')}")
