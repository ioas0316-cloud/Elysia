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

DATA_DIR = Path("data")
KG_PATH = DATA_DIR / 'kg.json'

class KGManager:
    def __init__(self, locked=True):
        self.locked = locked
        DATA_DIR.mkdir(exist_ok=True)
        if KG_PATH.exists():
            with open(KG_PATH, 'r', encoding='utf-8') as f:
                self.kg = json.load(f)
        else:
            self.kg = {"nodes": [], "edges": []}

    def lock(self):
        self.locked = True
        print("KGManager is locked. Write operations are disabled.")

    def unlock(self):
        self.locked = False
        print("KGManager is unlocked. Write operations are enabled.")

    def save(self):
        """Saves to the default KG path."""
        self.save_to(KG_PATH)

    def save_to(self, filepath: str):
        """Saves the current KG to a specific file."""
        if self.locked:
            print(f"Cannot save to {filepath}: KGManager is locked.")
            return

        path_obj = Path(filepath)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(self.kg, f, ensure_ascii=False, indent=2)

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Finds a node by its ID."""
        for node in self.kg['nodes']:
            if node['id'] == node_id:
                return node
        return None

    def get_edge(self, source_id: str, target_id: str, relation: str) -> Optional[Dict]:
        """Finds a specific edge."""
        for edge in self.kg['edges']:
            if edge['source'] == source_id and edge['target'] == target_id and edge['relation'] == relation:
                return edge
        return None

    def add_node(self, node_id: str, position: Optional[Dict] = None, properties: Optional[Dict] = None) -> Optional[Dict]:
        """Adds a new node. If locked, does nothing."""
        if self.locked:
            print("Cannot add node: KGManager is locked.")
            return None
        return self._add_node_unlocked(node_id, position, properties)

    def _add_node_unlocked(self, node_id: str, position: Optional[Dict] = None, properties: Optional[Dict] = None) -> Dict:
        """Internal function to add a node without the lock check."""
        existing_node = self.get_node(node_id)
        if existing_node:
            if properties:
                existing_node.update(properties)
            return existing_node

        new_node = {
            "id": node_id,
            "position": position if position else {"x": 0, "y": 0, "z": 0},
            "activation_energy": 0.0
        }
        if properties:
            new_node.update(properties)

        self.kg['nodes'].append(new_node)
        return new_node

    def add_edge(self, source_id: str, target_id: str, relation: str, properties: Optional[Dict[str, Any]] = None):
        """Adds a directional edge. If locked, does nothing."""
        if self.locked:
            print("Cannot add edge: KGManager is locked.")
            return

        if any(e['source'] == source_id and e['target'] == target_id and e['relation'] == relation for e in self.kg['edges']):
            return

        source_node = self._add_node_unlocked(source_id) # Use unlocked version internally
        target_node = self.get_node(target_id)

        if not target_node or target_node['position'] == {"x": 0, "y": 0, "z": 0}:
            target_node = self._add_node_unlocked(target_id)
            pos = source_node['position'].copy()

            if relation == "is_composed_of":
                pos['z'] += 1
            elif relation == "is_a":
                pos['y'] += 1
            elif relation == "causes":
                pos['x'] += 1.5
            else:
                pos['x'] += 1

            pos['x'] += random.uniform(-0.2, 0.2)
            pos['y'] += random.uniform(-0.2, 0.2)
            pos['z'] += random.uniform(-0.2, 0.2)

            target_node['position'] = pos

        new_edge = {"source": source_id, "target": target_id, "relation": relation}
        if properties:
            new_edge.update(properties)

        self.kg['edges'].append(new_edge)

    def update_edge_property(self, source_id: str, target_id: str, relation: str, prop_key: str, prop_value: Any):
        """Updates a property of a specific edge. If locked, does nothing."""
        if self.locked:
            print("Cannot update edge: KGManager is locked.")
            return

        edge = self.get_edge(source_id, target_id, relation)
        if edge:
            edge[prop_key] = prop_value
            print(f"Updated edge ({source_id}-{target_id}) property '{prop_key}' to '{prop_value}'.")
        else:
            print("Edge not found for update.")

    def remove_edge(self, source_id: str, target_id: str, relation: str):
        """Removes a specific edge from the graph. If locked, does nothing."""
        if self.locked:
            print("Cannot remove edge: KGManager is locked.")
            return

        initial_edge_count = len(self.kg['edges'])
        self.kg['edges'] = [
            edge for edge in self.kg['edges']
            if not (edge['source'] == source_id and edge['target'] == target_id and edge['relation'] == relation)
        ]

        if len(self.kg['edges']) < initial_edge_count:
            print(f"Successfully removed edge: {source_id} -[{relation}]-> {target_id}")
        else:
            print(f"Edge not found for removal: {source_id} -[{relation}]-> {target_id}")

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
    kg_manager = KGManager()
    kg_manager.kg = {"nodes": [], "edges": []}

    # Example of a causal chain
    kg_manager.add_edge("rain", "wet_ground", "causes", properties={"strength": 0.9})
    kg_manager.add_edge("wet_ground", "slippery_surface", "causes", properties={"strength": 0.7})

    kg_manager.save()

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
