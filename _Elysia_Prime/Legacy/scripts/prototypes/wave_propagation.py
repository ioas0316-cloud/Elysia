# [Genesis: 2025-12-02] Purified by Elysia
import networkx as nx
from build_graph import build_knowledge_graph

# --- Data Structures ---

class Wave:
    """
    Represents a wave of activation propagating through the graph.
    """
    def __init__(self, origin_node, start_node, initial_energy=1.0, payload=None):
        self.origin_node = origin_node
        self.current_node = start_node
        self.energy = initial_energy
        self.payload = payload if payload is not None else {}
        self.path = [start_node]

    def __repr__(self):
        return f"Wave at {self.current_node} (Energy: {self.energy:.2f}, Path: {' -> '.join(self.path)})"

class Echo:
    """
    Represents an echo sent back to the origin node after a wave interaction.
    """
    def __init__(self, origin_node, final_node, message):
        self.origin_node = origin_node
        self.final_node = final_node
        self.message = message

    def __repr__(self):
        return f"Echo from {self.final_node} to {self.origin_node}: {self.message}"


# --- Core Functions ---

def process_node_interaction(graph, wave):
    """
    Processes the interaction when a wave hits a node.
    This function now handles different query types, including 'new_insight' and 'verify_composition'.
    """
    node_data = graph.nodes[wave.current_node]
    payload = wave.payload
    query_type = payload.get('query')

    print(f"  - Wave hit {wave.current_node} ({node_data.get('name', 'N/A')}) with query: {query_type}")

    # --- Query: find_id ---
    if query_type == 'find_id' and wave.current_node == payload.get('target_id'):
        message = f"Target node '{payload.get('target_id')}' found."
        return Echo(wave.origin_node, wave.current_node, message)

    # --- Query: trace_hierarchy ---
    if query_type == 'trace_hierarchy':
        has_outgoing_is_a = False
        for neighbor in graph.neighbors(wave.current_node):
            if graph.get_edge_data(wave.current_node, neighbor).get('type') == 'is_a':
                has_outgoing_is_a = True
                break
        if not has_outgoing_is_a:
            message = f"Hierarchy trace completed. Path: {' -> '.join(wave.path)}"
            return Echo(wave.origin_node, wave.current_node, message)

    # --- Query: new_insight (Learning) ---
    if query_type == 'new_insight':
        # This wave reinforces the path it travels.
        if len(wave.path) > 1:
            prev_node = wave.path[-2]
            current_node = wave.current_node

            # Increase the weight of the edge it just traversed.
            edge_data = graph.edges[prev_node, current_node]
            original_weight = edge_data.get('weight', 1.0)
            edge_data['weight'] += 0.1 # Reinforcement learning rate
            print(f"    - LEARNING: Reinforced edge ({prev_node} -> {current_node}). "
                  f"Weight: {original_weight:.2f} -> {edge_data['weight']:.2f}")

    # --- Query: verify_composition ---
    if query_type == 'verify_composition':
        target_composition_id = payload.get('composed_of_target')
        if target_composition_id:
            for u, v, data in graph.out_edges(wave.current_node, data=True):
                if data.get('type') == 'is_composed_of' and v == target_composition_id:
                    message = f"{wave.current_node} is indeed composed of {target_composition_id}."
                    return Echo(wave.origin_node, wave.current_node, message)
            message = f"{wave.current_node} is NOT composed of {target_composition_id}."
            return Echo(wave.origin_node, wave.current_node, message)

    return None

from collections import defaultdict

def run_simulation(graph, initial_waves, max_steps=5):
    """
    Runs a multi-wave simulation, handling propagation (now weight-sensitive)
    and collisions.
    """
    active_waves = list(initial_waves)
    all_echoes = []

    for step in range(max_steps):
        if not active_waves:
            print("\nNo more active waves. Ending simulation.")
            break

        print(f"\n--- Step {step + 1} ---")

        nodes_this_step = defaultdict(list)
        step_echoes = []

        # 1. Process interactions for all current waves before they move
        for wave in active_waves:
            echo = process_node_interaction(graph, wave)
            if echo:
                step_echoes.append(echo)

        # 2. Propagate all active waves to their next positions
        for wave in active_waves:
            # Find the total weight of all valid outgoing edges to normalize
            neighbors = [n for n in graph.neighbors(wave.current_node) if n not in wave.path]
            if not neighbors: continue

            total_weight = sum(graph.edges[wave.current_node, neighbor].get('weight', 1.0) for neighbor in neighbors)
            if total_weight == 0: continue

            for neighbor in neighbors:
                # Energy transfer is now proportional to the edge's relative weight
                edge_weight = graph.edges[wave.current_node, neighbor].get('weight', 1.0)
                energy_share = (edge_weight / total_weight) if total_weight > 0 else 0
                new_energy = wave.energy * energy_share

                if new_energy > 0.05: # Lowered threshold for more propagation
                    new_wave = Wave(wave.origin_node, neighbor, new_energy, wave.payload)
                    new_wave.path = wave.path + [neighbor]
                    nodes_this_step[neighbor].append(new_wave)

        # 3. Handle collisions and collect resulting waves
        final_waves_for_next_step = []
        for node, wave_list in nodes_this_step.items():
            if len(wave_list) > 1:
                resulting_waves, collision_echoes = handle_wave_collision(node, wave_list)
                final_waves_for_next_step.extend(resulting_waves)
                step_echoes.extend(collision_echoes)
            else:
                final_waves_for_next_step.extend(wave_list)

        active_waves = final_waves_for_next_step
        all_echoes.extend(step_echoes)

        print(f"  - {len(active_waves)} waves moving to next step.")
        for w in active_waves:
            print(f"    - {w}")

    print("\n--- Simulation Finished ---")
    return all_echoes

def main():
    """
    Main function to set up and run a multi-wave simulation
    demonstrating a collision and a composition verification.
    """
    knowledge_graph = build_knowledge_graph()

    if knowledge_graph.number_of_nodes() == 0:
        print("Graph is empty. Exiting.")
        return

    # --- Simulation 1: Wave Collision (Learning Common Ancestor) ---
    print("\n" + "="*40)
    print("SIMULATION 1: Wave Collision (Learning Common Ancestor)")
    print("="*40)

    start_node_1 = 'geom_composite_triangle'
    start_node_2 = 'geom_composite_square'

    payload_collision = {'query': 'trace_hierarchy'}

    print(f"Goal: Start two waves from '{start_node_1}' and '{start_node_2}' to find a common ancestor.")

    wave_1 = Wave(start_node_1, start_node_1, initial_energy=1.0, payload=payload_collision)
    wave_2 = Wave(start_node_2, start_node_2, initial_energy=1.0, payload=payload_collision)

    echoes_collision = run_simulation(knowledge_graph, [wave_1, wave_2])

    print("\n--- Simulation 1 Results ---")
    if echoes_collision:
        print("Collected Echoes:")
        for echo in echoes_collision:
            print(f"- {echo}")
    else:
        print("No echoes were generated.")

    # --- Simulation 2: Verify Composition (Geometric Primitives Lesson) ---
    print("\n" + "="*40)
    print("SIMULATION 2: Verify Composition (Geometric Primitives Lesson)")
    print("="*40)

    start_node_3 = 'geom_primitive_line'
    payload_composition = {'query': 'verify_composition', 'composed_of_target': 'geom_primitive_point'}

    print(f"Goal: Start a wave from '{start_node_3}' to verify if it's composed of '{payload_composition['composed_of_target']}'.")

    wave_3 = Wave(start_node_3, start_node_3, initial_energy=1.0, payload=payload_composition)

    echoes_composition = run_simulation(knowledge_graph, [wave_3])

    print("\n--- Simulation 2 Results ---")
    if echoes_composition:
        print("Collected Echoes:")
        for echo in echoes_composition:
            print(f"- {echo}")
        # Additional check for the expected echo
        expected_echo_message = f"{start_node_3} is indeed composed of {payload_composition['composed_of_target']}."
        found_expected_echo = any(echo.message == expected_echo_message for echo in echoes_composition)
        if found_expected_echo:
            print(f"  Lesson learned: Elysia understands that a {start_node_3} is composed of {payload_composition['composed_of_target']}!")
        else:
            print(f"  Lesson failed: Elysia does NOT understand that a {start_node_3} is composed of {payload_composition['composed_of_target']}.")
    else:
        print("No echoes were generated.")

if __name__ == '__main__':
    main()