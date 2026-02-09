
import json
import sys

def analyze_kg_recursive(nodes_dict, depth=0):
    total_nodes = 0
    all_edges = []
    
    for node_id, node_data in nodes_dict.items():
        total_nodes += 1
        
        # Check for inner_cosmos
        inner = node_data.get('inner_cosmos', {})
        if inner:
            inner_nodes = inner.get('nodes', {})
            inner_edges = inner.get('edges', [])
            
            all_edges.extend(inner_edges)
            
            if inner_nodes:
                # Recurse
                sub_nodes, sub_edges = analyze_kg_recursive(inner_nodes, depth + 1)
                total_nodes += sub_nodes
                all_edges.extend(sub_edges)
            
    return total_nodes, all_edges

def main():
    file_path = 'data/kg_with_embeddings.json'
    print(f"Loading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            kg = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
    
    top_nodes = kg.get('nodes', {})
    top_edges = kg.get('edges', [])
    
    print(f"Top-level nodes: {len(top_nodes)}")
    print(f"Top-level edges: {len(top_edges)}")
    
    print("Traversing recursively...")
    total_nodes, inner_edges = analyze_kg_recursive(top_nodes)
    all_edges = top_edges + inner_edges
    
    print(f"Total Nodes (Recursive): {total_nodes}")
    print(f"Total Edges (Recursive): {len(all_edges)}")
    
    if not all_edges:
        print("No edges found in the graph.")
        return

    # Count connectivity per node
    mass = {}
    for edge in all_edges:
        source = edge.get('source')
        target = edge.get('target')
        mass[source] = mass.get(source, 0) + 1
        mass[target] = mass.get(target, 0) + 1
    
    sorted_mass = sorted(mass.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Massive Concepts (Recursive Connectivity):")
    for concept, count in sorted_mass[:20]:
        print(f"- {concept}: {count}")

if __name__ == "__main__":
    main()
