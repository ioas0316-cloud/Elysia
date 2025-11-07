from tools.kg_manager import KGManager
from typing import List, Dict, Any

class KnowledgeEnhancer:
    def __init__(self, kg_manager: KGManager):
        self.kg_manager = kg_manager

    def process_learning_points(self, learning_data: Any, image_path: str):
        """
        Processes learning data from various textbook formats and integrates it into the KG.
        """
        concepts = []
        relations = []

        if isinstance(learning_data, dict) and 'concepts' in learning_data and 'relationships' in learning_data:
            concepts = learning_data.get('concepts', [])
            relations = learning_data.get('relationships', [])
        elif isinstance(learning_data, list):
            concepts = [p for p in learning_data if p.get('type') == 'concept']
            relations = [p for p in learning_data if p.get('type') == 'relation']
        else:
            print(f"Warning: Unknown learning data format. Skipping. Data: {learning_data}")
            return

        # --- Process Concepts ---
        for concept_info in concepts:
            # FIX: Prioritize the 'id' field as the node identifier for consistency.
            node_id = concept_info.get('id') or concept_info.get('korean') or concept_info.get('label')
            if not node_id:
                continue

            properties = concept_info.get('properties', {})
            properties['description'] = concept_info.get('description', properties.get('description'))
            # Ensure the Korean label is stored if available
            if 'korean' in concept_info:
                properties['korean'] = concept_info['korean']

            if not self.kg_manager.get_node(node_id):
                print(f"Adding new node: '{node_id}'")
                self.kg_manager.add_node(node_id, properties=properties)
            else:
                print(f"Node '{node_id}' already exists. Skipping creation.")

        # --- Process Relationships ---
        for relation_info in relations:
            source = relation_info.get('source')
            target = relation_info.get('target')
            label = relation_info.get('relation') or relation_info.get('label')

            if source and target and label:
                print(f"Adding new edge: {source} -> {target} ({label})")
                self.kg_manager.add_edge(
                    source,
                    target,
                    label,
                    properties=relation_info.get('properties', {})
                )

        print(f"Knowledge graph processing complete for this lesson.")
