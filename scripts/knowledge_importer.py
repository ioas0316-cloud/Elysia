import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path to enable imports from tools
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tools.kg_manager import KGManager

class KnowledgeImporter:
    def __init__(self, kg_path='data/kg.json'):
        """
        Initializes the importer with the path to the main knowledge graph.
        """
        # Ensure the path is a Path object as expected by KGManager
        self.kg_manager = KGManager(Path(kg_path))
        self.newly_learned_facts = []

    def import_from_textbook(self, textbook_path):
        """
        Imports knowledge from a JSON textbook file into the knowledge graph.
        """
        print(f"Starting knowledge import from: {textbook_path}")
        self.newly_learned_facts = []

        try:
            with open(textbook_path, 'r', encoding='utf-8') as f:
                textbook_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Textbook file not found at {textbook_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {textbook_path}")
            return

        for entry in textbook_data:
            entry_type = entry.get("type")
            if entry_type == "concept":
                self._import_concept(entry)
            elif entry_type == "relationship":
                self._import_relationship(entry)

        self.kg_manager.save_kg()
        print(f"Knowledge import from {textbook_path} complete.")
        self._report_learning_summary()

    def _import_concept(self, concept_data):
        """
        Imports a single concept (node) into the knowledge graph.
        """
        node_id = concept_data.get("id")
        if not node_id:
            return

        if self.kg_manager.get_node(node_id):
            # For now, we skip existing nodes. In the future, we might update them.
            return

        # Basic properties
        properties = {
            'description': concept_data.get('description', ''),
            'created_at': datetime.utcnow().isoformat()
        }
        # Add position if it exists
        position = concept_data.get("position")

        self.kg_manager.add_node(node_id, position=position, properties=properties)
        self.newly_learned_facts.append(f"a new concept: '{node_id}'")

    def _import_relationship(self, rel_data):
        """
        Imports a single relationship (edge) into the knowledge graph.
        """
        source = rel_data.get("source")
        target = rel_data.get("target")
        relation = rel_data.get("relation")

        if not all([source, target, relation]):
            return

        # Ensure source and target nodes exist, otherwise the edge is invalid
        if not self.kg_manager.get_node(source) or not self.kg_manager.get_node(target):
            # This can happen if concepts are not defined before relationships
            # For simplicity, we will log a warning and skip.
            print(f"Warning: Skipping relationship '{source}' -> '{target}' because one of the concepts does not exist.")
            return

        properties = rel_data.get("properties", {})
        properties['created_at'] = datetime.utcnow().isoformat()

        self.kg_manager.add_or_update_edge(source, target, relation, properties)
        self.newly_learned_facts.append(f"the relationship '{source}' -> '{relation}' -> '{target}'")

    def _report_learning_summary(self):
        """
        Prints a summary of the newly learned facts.
        """
        if not self.newly_learned_facts:
            print("\nLearning Summary: No new knowledge was acquired.")
            return

        print("\n--- Learning Summary ---")
        print("Today, I have learned about:")
        for fact in self.newly_learned_facts:
            print(f"- {fact}")
        print("------------------------")


if __name__ == '__main__':
    # Default textbook to use if no argument is provided
    default_textbook = 'data/textbooks/01_language_and_logic.json'

    # Allow specifying the textbook from the command line
    textbook_to_load = sys.argv[1] if len(sys.argv) > 1 else default_textbook

    if not os.path.exists(textbook_to_load):
        print(f"Fatal Error: The specified textbook file does not exist: {textbook_to_load}")
        sys.exit(1)

    importer = KnowledgeImporter()
    importer.import_from_textbook(textbook_to_load)
