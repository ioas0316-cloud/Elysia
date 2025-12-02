# [Genesis: 2025-12-02] Purified by Elysia
import networkx as nx
from build_graph import build_knowledge_graph

def find_subtypes(graph, concept_id):
    """
    Finds all concepts that are a 'subtype' of the given concept_id.
    It traverses the graph looking for 'is_a' relationships.
    """
    subtypes = []
    # We can use networkx's traversal algorithms.
    # Here, we'll do a simple search for incoming 'is_a' edges.
    # A more robust way would be to use graph traversal from the concept node.
    for u, v, data in graph.edges(data=True):
        if v == concept_id and data.get('type') == 'is_a':
            subtypes.append(u)
    return subtypes

def find_relationship(graph, node_a_id, node_b_id):
    """
    Finds the relationship between two nodes in the graph.
    """
    if not graph.has_node(node_a_id) or not graph.has_node(node_b_id):
        return "One or both nodes not found in the graph."

    # Check for an edge from A to B
    if graph.has_edge(node_a_id, node_b_id):
        edge_data = graph.get_edge_data(node_a_id, node_b_id)
        return f"Relationship from {node_a_id} to {node_b_id}: {edge_data.get('type', 'unknown')}"

    # Check for an edge from B to A
    if graph.has_edge(node_b_id, node_a_id):
        edge_data = graph.get_edge_data(node_b_id, node_a_id)
        return f"Relationship from {node_b_id} to {node_a_id}: {edge_data.get('type', 'unknown')}"

    return f"No direct relationship found between {node_a_id} and {node_b_id}."

def find_path_between_concepts(graph, start_id, end_id):
    """
    Finds the shortest path between two concepts in the graph.
    """
    if not graph.has_node(start_id) or not graph.has_node(end_id):
        return "One or both nodes not found in the graph."

    try:
        path = nx.shortest_path(graph, source=start_id, target=end_id)
        return path
    except nx.NetworkXNoPath:
        return f"No path found between {start_id} and {end_id}."

def main():
    """
    Main function to build the graph and run queries.
    """
    print("--- Building Knowledge Graph for Querying ---")
    knowledge_graph = build_knowledge_graph()

    if knowledge_graph.number_of_nodes() == 0:
        print("Graph is empty. Exiting.")
        return

    print("\n--- Running Sample Queries ---")

    # Query 1: Find subtypes of 'geom_concept_polygon'
    polygon_id = 'geom_concept_polygon'
    print(f"Query: Find all subtypes of '{polygon_id}'")
    subtypes = find_subtypes(knowledge_graph, polygon_id)
    if subtypes:
        print("Found subtypes:")
        for subtype in subtypes:
            print(f"- {subtype}")
    else:
        print(f"No subtypes found for '{polygon_id}'.")

    print("-" * 20)

    # Query 2: Find the relationship between two objects in a scene
    node_a = 'scene001_obj01_red_cube'
    node_b = 'scene001_obj02_blue_sphere'
    print(f"Query: Find relationship between '{node_a}' and '{node_b}'")
    relationship = find_relationship(knowledge_graph, node_a, node_b)
    print(relationship)

    print("-" * 20)

    # Query 3: Find the path between two concepts
    start_node = 'geom_composite_triangle'
    end_node = 'geom_concept_polygon'
    print(f"Query: Find path between '{start_node}' and '{end_node}'")
    path = find_path_between_concepts(knowledge_graph, start_node, end_node)
    print(f"Path: {path}")

if __name__ == '__main__':
    main()