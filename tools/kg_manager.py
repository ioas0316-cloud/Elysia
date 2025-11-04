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
KG_PATH = DATA_DIR / 'kg_with_embeddings.json'

class KGManager:
    def __init__(self):
        DATA_DIR.mkdir(exist_ok=True)
        if KG_PATH.exists():
            with open(KG_PATH, 'r', encoding='utf-8') as f:
                self.kg = json.load(f)
        else:
            self.kg = {"nodes": [], "edges": []}

    def save(self):
        with open(KG_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.kg, f, ensure_ascii=False, indent=2)

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Finds a node by its ID."""
        for node in self.kg['nodes']:
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
            "position": position if position else {"x": 0, "y": 0, "z": 0},
            "activation_energy": 0.0
        }
        if properties:
            new_node.update(properties)

        self.kg['nodes'].append(new_node)
        return new_node

    def add_or_update_edge(self, source_id: str, target_id: str, relation: str, properties: Optional[Dict[str, Any]] = None):
        """
        Adds a new directional edge or updates an existing one with new properties.
        Custom properties are stored in a nested 'properties' dictionary to avoid key collisions.
        """
        # Check for an existing edge
        for edge in self.kg['edges']:
            if edge['source'] == source_id and edge['target'] == target_id and edge['relation'] == relation:
                # If found, update its properties
                if properties:
                    edge.setdefault('properties', {}).update(properties)
                return

        # If no edge exists, create a new one
        source_node = self.add_node(source_id)
        target_node = self.get_node(target_id)

        # Position the target node if it's new or at the origin
        if not target_node or target_node['position'] == {"x": 0, "y": 0, "z": 0}:
            target_node = self.add_node(target_id)
            pos = source_node['position'].copy()

            # Expanded positioning logic for new causal types
            if relation == "is_composed_of":
                pos['z'] += 1
            elif relation == "is_a":
                pos['y'] += 1
            elif relation in ["causes", "enables", "requires"]:
                pos['x'] += 1.5
            elif relation == "prevents":
                pos['x'] -= 1.5 # Move in opposite direction for opposition
            elif relation == "contextualizes":
                pos['y'] -= 1 # Place contextually below
            else:
                pos['x'] += 1 # Default for other relations

            pos['x'] += random.uniform(-0.2, 0.2)
            pos['y'] += random.uniform(-0.2, 0.2)
            pos['z'] += random.uniform(-0.2, 0.2)

            target_node['position'] = pos

        # Create the new edge with a nested properties dictionary
        new_edge = {
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "properties": properties if properties else {}
        }
        self.kg['edges'].append(new_edge)

    def find_related_edges(self, node_id: str, relation: Optional[str] = None, direction: str = 'any') -> List[Dict[str, Any]]:
        """
        Finds all edges connected to a node, with optional filters for relation type and direction.
        - direction: 'out' (source), 'in' (target), or 'any'
        """
        results = []
        for edge in self.kg.get('edges', []):
            is_related = False
            # Check direction
            if direction == 'out' and edge.get('source') == node_id:
                is_related = True
            elif direction == 'in' and edge.get('target') == node_id:
                is_related = True
            elif direction == 'any' and (edge.get('source') == node_id or edge.get('target') == node_id):
                is_related = True

            if is_related:
                # Check relation type if specified
                if relation is None or edge.get('relation') == relation:
                    results.append(edge)
        return results

    def find_causes(self, target_id: str) -> List[Dict[str, Any]]:
        """Finds all causes for a given target node (semantic wrapper)."""
        return self.find_related_edges(target_id, relation='causes', direction='in')

    def find_effects(self, source_id: str) -> List[Dict[str, Any]]:
        """Finds all effects of a given source node (semantic wrapper)."""
        return self.find_related_edges(source_id, relation='causes', direction='out')

    def get_summary(self):
        return {"nodes": len(self.kg['nodes']), "edges": len(self.kg['edges'])}

if __name__ == '__main__':
    # --- In-memory test to prevent overwriting the actual data file ---
    kg_manager = KGManager()
    # Use an in-memory KG for this test run, do not load from file
    kg_manager.kg = {"nodes": [], "edges": []}

    print("--- Testing KGManager Enhancements (In-Memory) ---")

    # 1. Add a basic causal chain with properties
    print("\n1. Adding a basic causal chain...")
    kg_manager.add_or_update_edge("sun_is_out", "park_is_sunny", "causes",
                                  properties={"strength": 0.95, "certainty": 0.98, "source": "observation"})
    kg_manager.add_or_update_edge("park_is_sunny", "people_go_to_park", "enables",
                                  properties={"strength": 0.6, "certainty": 0.8, "modality": "sometimes"})

    # 2. Add more complex relationships based on the new schema
    print("2. Adding complex relationships...")
    kg_manager.add_or_update_edge("writing_a_book", "creativity", "requires",
                                  properties={"strength": 0.8, "source": "user_defined"})
    kg_manager.add_or_update_edge("heavy_rain", "picnic", "prevents",
                                  properties={"strength": 0.99, "certainty": 0.95})

    # 3. Test updating an existing edge
    print("3. Updating an existing edge...")
    kg_manager.add_or_update_edge("sun_is_out", "park_is_sunny", "causes",
                                  properties={"strength": 0.97, "notes": "updated value"})

    # NOTE: The save() method is intentionally NOT called to protect the data file.
    print("\n4. Data operations performed in-memory. File was NOT saved.")

    # 5. Test query functions with the new nested property structure
    print("\n5. Testing query functions...")

    # Test find_causes (wrapper for the new generic function)
    causes_of_park_sunny = kg_manager.find_causes("park_is_sunny")
    print(f"\nCauses of 'park_is_sunny' (found {len(causes_of_park_sunny)}):")
    for edge in causes_of_park_sunny:
        props = edge.get('properties', {})
        print(f"- Source: {edge['source']}, Strength: {props.get('strength', 'N/A')}, Notes: {props.get('notes', 'N/A')}")

    # Test find_related_edges for 'enables'
    enablers_of_people_go_to_park = kg_manager.find_related_edges("people_go_to_park", relation="enables", direction="in")
    print(f"\nEnablers of 'people_go_to_park' (found {len(enablers_of_people_go_to_park)}):")
    for edge in enablers_of_people_go_to_park:
        props = edge.get('properties', {})
        print(f"- Source: {edge['source']}, Certainty: {props.get('certainty', 'N/A')}")

    # Test find_related_edges for 'requires'
    requirements_for_writing = kg_manager.find_related_edges("writing_a_book", relation="requires", direction="out")
    print(f"\n'writing_a_book' requires (found {len(requirements_for_writing)}):")
    for edge in requirements_for_writing:
        props = edge.get('properties', {})
        print(f"- Target: {edge['target']}, Strength: {props.get('strength', 'N/A')}")

    print("\n--- Test Complete ---")
    print('Final KG summary:', kg_manager.get_summary())
