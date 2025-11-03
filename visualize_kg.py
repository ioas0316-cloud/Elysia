# visualize_kg.py
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_knowledge_graph_with_test_results():
    """
    Reads the tool knowledge graph and test log, then generates a 3D
    visualization with nodes colored by their test status.
    """
    # Load the knowledge graph
    with open('data/tools_kg.json', 'r', encoding='utf-8') as f:
        kg = json.load(f)

    # Load and process the test log
    test_log = []
    if os.path.exists('test_log.json'):
        with open('test_log.json', 'r', encoding='utf-8') as f:
            test_log = json.load(f)

    test_status = {}
    running_tests = {}
    for event in test_log:
        test_id = event['test']
        if event['event'] == 'start_test':
            running_tests[test_id] = 'running'
        elif event['event'] in ['add_success', 'stop_test']:
            if test_id in running_tests:
                running_tests.pop(test_id)
            test_status[test_id] = 'success'
        elif event['event'] in ['add_error', 'add_failure']:
            if test_id in running_tests:
                running_tests.pop(test_id)
            test_status[test_id] = 'failure'

    # Mark hanging tests
    for test_id in running_tests:
        test_status[test_id] = 'hanging'

    # --- Visualization ---
    nodes = kg['nodes']
    edges = kg['edges']

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color map for nodes
    color_map = {}
    for node in nodes:
        node_id = node['id']
        # This is a simplified mapping. In a real scenario, you'd have a
        # more robust way to link tests to the KG nodes they affect.
        # For now, we color the tool nodes based on the tests that use them.
        if 'calculate' in node_id and any('arithmetic' in test for test in test_status):
            if any('arithmetic' in test and status == 'failure' for test, status in test_status.items()):
                color_map[node_id] = 'red'
            elif any('arithmetic' in test and status == 'hanging' for test, status in test_status.items()):
                color_map[node_id] = 'yellow'
            else:
                color_map[node_id] = 'green'
        elif any(keyword in node_id for keyword in ['read', 'file', 'list']) and any('action_cortex' in test for test in test_status):
             if any('action_cortex' in test and status == 'failure' for test, status in test_status.items()):
                color_map[node_id] = 'red'
             else:
                color_map[node_id] = 'green'
        else:
            color_map[node_id] = 'blue'


    # Plot nodes
    for node in nodes:
        ax.scatter(node['x'], node['y'], node['z'], s=150, c=color_map.get(node['id'], 'blue'))
        ax.text(node['x'], node['y'], node['z'], f"  {node['id']}", size=10)

    # Plot edges
    for edge in edges:
        source_node = next((n for n in nodes if n['id'] == edge['source']), None)
        target_node = next((n for n in nodes if n['id'] == edge['target']), None)

        if source_node and target_node:
            ax.plot([source_node['x'], target_node['x']],
                    [source_node['y'], target_node['y']],
                    [source_node['z'], target_node['z']],
                    'gray', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Tool Knowledge Graph - Test Status')
    plt.savefig('kg_test_visualization.png')
    print("Knowledge graph test visualization saved to kg_test_visualization.png")

if __name__ == '__main__':
    visualize_knowledge_graph_with_test_results()
