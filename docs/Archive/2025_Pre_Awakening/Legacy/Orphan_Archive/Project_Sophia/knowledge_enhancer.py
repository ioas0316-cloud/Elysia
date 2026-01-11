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
            node_id = concept_info.get('id') or concept_info.get('korean') or concept_info.get('label')
            if not node_id:
                continue

            properties = concept_info.get('properties', {})

            # If learning from a visual source, enrich the properties
            if image_path and image_path != "conceptual_learning":
                properties['description'] = f"Concept learned from visual experience: {image_path}"
                properties['category'] = 'learned_concept'
                # Ensure experience_visual is a list and append the new path
                if 'experience_visual' not in properties:
                    properties['experience_visual'] = []
                if image_path not in properties['experience_visual']:
                    properties['experience_visual'].append(image_path)
            else:
                 properties['description'] = concept_info.get('description', properties.get('description'))

            if 'korean' in concept_info:
                properties['korean'] = concept_info['korean']

            if not self.kg_manager.get_node(node_id):
                print(f"Adding new node: '{node_id}'")
                self.kg_manager.add_node(node_id, properties=properties)
            else:
                print(f"Node '{node_id}' already exists. Updating properties.")
                self.kg_manager.update_node_properties(node_id, properties)


        # --- Process Relationships ---
        for relation_info in relations:
            source = relation_info.get('source')
            target = relation_info.get('target')
            label = relation_info.get('relation') or relation_info.get('label')

            if source and target and label:
                relation_properties = relation_info.get('properties', {})
                if image_path and image_path != "conceptual_learning":
                    relation_properties['experience_visual'] = image_path

                print(f"Adding new edge: {source} -> {target} ({label})")
                self.kg_manager.add_edge(
                    source,
                    target,
                    label,
                    properties=relation_properties
                )

        print(f"Knowledge graph processing complete for this lesson.")
