import json
import os
import sys
from tqdm import tqdm

# Add the project root to the Python path to allow importing from Project_Sophia
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Core.Foundation.gemini_api import get_text_embedding

def enrich_kg_with_embeddings():
    """
    Loads the knowledge graph, generates embeddings for each node,
    and saves the enriched graph to a new file.
    """
    input_kg_path = os.path.join(project_root, 'data', 'kg.json')
    output_kg_path = os.path.join(project_root, 'data', 'kg_with_embeddings.json')

    print(f"Loading knowledge graph from {input_kg_path}...")
    try:
        with open(input_kg_path, 'r', encoding='utf-8') as f:
            kg = json.load(f)
    except FileNotFoundError:
        print(f"Error: Knowledge graph file not found at {input_kg_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_kg_path}")
        return

    if 'nodes' not in kg:
        print("Error: 'nodes' key not found in the knowledge graph.")
        return

    print("Enriching knowledge graph with embeddings...")
    # Using tqdm to show progress
    for node in tqdm(kg['nodes'], desc="Generating Embeddings"):
        node_id = node.get('id')
        if node_id and 'embedding' not in node: # Process only if embedding doesn't exist
            try:
                # The embedding is generated from the node's ID, which represents the concept
                embedding = get_text_embedding(node_id)
                if embedding:
                    node['embedding'] = embedding
                else:
                    print(f"Warning: Could not generate embedding for node '{node_id}'.")
            except Exception as e:
                print(f"An error occurred while generating embedding for node '{node_id}': {e}")

    print(f"Saving enriched knowledge graph to {output_kg_path}...")
    try:
        with open(output_kg_path, 'w', encoding='utf-8') as f:
            json.dump(kg, f, indent=2, ensure_ascii=False)
        print("Successfully enriched the knowledge graph with embeddings.")
    except Exception as e:
        print(f"Error saving the enriched knowledge graph: {e}")

if __name__ == "__main__":
    enrich_kg_with_embeddings()
