import glob
import json
import networkx as nx
import os

def build_knowledge_graph():
    """
    Scans the 'docs' directory for JSON files, parses them,
    and builds a NetworkX DiGraph representing Elysia's knowledge.
    """
    print("--- Starting Knowledge Graph Construction ---")
    G = nx.DiGraph()
    
    # Determine the project root directory.
    # The script is expected to be in <project_root>/scripts/prototypes.
    try:
        script_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    except NameError:
        # Fallback for when __file__ is not available (e.g., in a REPL or notebook)
        # In this case, we assume the current working directory is the project root.
        project_root = os.getcwd()

docs_path = os.path.join(project_root, 'docs', '**', '*.json')

    print(f"Searching for JSON blueprints in: {os.path.join(project_root, 'docs')}")
    
    json_files = glob.glob(docs_path, recursive=True)
    if not json_files:
        print("Warning: No JSON files found in the 'docs' directory.")
        return G

    # First pass: Add all concepts and instances as nodes
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Add concepts
                for concept in data.get('concepts', []):
                    node_id = concept.get('id')
                    if node_id and not G.has_node(node_id):
                        G.add_node(node_id, **concept)
                
                # Add instances
                for instance in data.get('instances', []):
                    node_id = instance.get('instance_id')
                    if node_id and not G.has_node(node_id):
                        G.add_node(node_id, **instance)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}")
        except Exception as e:
            print(f"Error processing nodes in {file_path}: {e}")

    # Second pass: Add all relationships between existing nodes
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Link relationships from concepts
                for concept in data.get('concepts', []):
                    source_id = concept.get('id')
                    for rel in concept.get('relationships', []):
                        target_id = rel.get('target_id')
                        if source_id and target_id and G.has_node(source_id) and G.has_node(target_id):
                            G.add_edge(source_id, target_id, type=rel.get('type', 'unknown'), weight=1.0, **rel)

                # Link relationships from scenes
                for rel in data.get('relationships', []):
                    # Handle different role names for source/target
                    source_id = rel.get('cause') or rel.get('subject') or rel.get('object_a') or rel.get('agent')
                    target_id = rel.get('effect') or rel.get('reference') or rel.get('object_b') or rel.get('container') or rel.get('enabled_action')
                    if source_id and target_id and G.has_node(source_id) and G.has_node(target_id):
                        G.add_edge(source_id, target_id, type=rel.get('type', 'unknown'), weight=1.0, **rel)

        except Exception as e:
            print(f"Error processing relationships in {file_path}: {e}")


    print("\n--- Knowledge Graph Summary ---")
    print(f"Total nodes (concepts/instances): {G.number_of_nodes()}")
    print(f"Total edges (relationships): {G.number_of_edges()}")
    print("-----------------------------")
    print("To inspect the graph, you can run this script in an interactive session and explore the returned 'G' object.")

    return G

if __name__ == '__main__':
    knowledge_graph = build_knowledge_graph()
