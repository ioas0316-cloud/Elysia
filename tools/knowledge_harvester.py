"""
Knowledge Harvester for Elysia.

This tool is responsible for 'harvesting' knowledge from structured JSON files
located in the 'docs' directory and integrating them into Elysia's core
knowledge graph (kg.json). It acts as the primary mechanism for expanding
Elysia's understanding of the world from curated sources.
"""
import os
import sys
import json
from pathlib import Path

# Add the project root to the Python path to allow imports from 'tools'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tools.kg_manager import KGManager

# The directory where knowledge textbooks are stored.
DOCS_DIR = Path(project_root) / "docs"

class KnowledgeHarvester:
    def __init__(self, kg_manager: KGManager):
        self.kg_manager = kg_manager

    def harvest_from_file(self, file_path: Path):
        """Processes a single JSON file and adds its knowledge to the KG."""
        print(f"Harvesting from: {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            nodes_added = 0
            edges_added = 0

            # MODIFICATION 2: Handle 'concepts' format correctly
            for concept_data in data.get('concepts', []):
                node_id = concept_data.get('id')
                if not node_id:
                    continue

                # FINAL FIX: Collect all other keys into a single 'properties' dictionary.
                properties = {k: v for k, v in concept_data.items() if k not in ['id', 'relationships']}
                self.kg_manager.add_node(node_id, properties=properties)
                nodes_added += 1

                for edge_data in concept_data.get('relationships', []):
                    source = node_id
                    target = edge_data.get('target_id')
                    relation = edge_data.get('type')
                    if not all([source, target, relation]):
                        continue

                    edge_properties = {k: v for k, v in edge_data.items() if k not in ['target_id', 'type']}
                    self.kg_manager.add_edge(source, target, relation, properties=edge_properties)
                    edges_added += 1

            # Backwards compatibility for the old 'nodes'/'edges' format
            for node_data in data.get('nodes', []):
                node_id = node_data.get('id')
                if not node_id:
                    continue
                properties = {k: v for k, v in node_data.items() if k != 'id'}
                self.kg_manager.add_node(node_id, properties=properties)
                nodes_added += 1

            for edge_data in data.get('edges', []):
                source = edge_data.get('source')
                target = edge_data.get('target')
                relation = edge_data.get('relation')
                if not all([source, target, relation]):
                    continue
                properties = {k: v for k, v in edge_data.items() if k not in ['source', 'target', 'relation']}
                self.kg_manager.add_edge(source, target, relation, properties=properties)
                edges_added += 1

            if nodes_added > 0 or edges_added > 0:
                print(f"Successfully harvested {nodes_added} nodes and {edges_added} edges.")

        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from {file_path.name}. Reason: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path.name}: {e}")

    def harvest_all(self):
        """Scans the DOCS_DIR and harvests knowledge from all JSON files."""
        print(f"Starting harvest from directory: {DOCS_DIR.resolve()}")

        json_files = list(DOCS_DIR.glob('*.json'))

        if not json_files:
            print("No .json files found in the docs directory. Nothing to harvest.")
            return

        for file_path in json_files:
            self.harvest_from_file(file_path)

        # Save the updated knowledge graph once all files are processed.
        self.kg_manager.save()
        print("\nHarvest complete. Knowledge graph has been updated.")


if __name__ == '__main__':
    # Initialize the manager for the main knowledge graph
    main_kg_manager = KGManager()

    # Run the harvest process
    harvester = KnowledgeHarvester(main_kg_manager)
    harvester.harvest_all()

    # Print a summary of the newly enriched knowledge graph
    summary = main_kg_manager.get_summary()
    print(f"\nFinal KG Summary: {summary['nodes']} nodes, {summary['edges']} edges.")
