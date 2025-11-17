import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from Project_Sophia.core.world import World
from tools.kg_manager import KGManager

# Define core concepts that should have higher initial energy
CORE_CONCEPTS = [
    "value:love",
    "value:trust",
    "value:hope",
    "concept:growth",
    "concept:kindness",
    "concept:healing",
    "concept:memory",
    "concept:curiosity",
    "obsidian_note:gratitude",
    "obsidian_note:forgiveness",
    "obsidian_note:compassion",
    "obsidian_note:shared_story",
]

def explore_cellular_world():
    """
    An exploratory script to initialize the Cellular World,
    stimulate a concept, and observe the simulation.
    """
    print("--- Starting Cellular World Exploration ---")

    # 1. Initialize KG Manager and load the graph
    print("\n[1/7] Loading Knowledge Graph...")
    kg = KGManager()
    if not kg.kg.get("nodes"):
        print("Knowledge Graph is empty. Cannot perform mirroring.")
        return
    print(f"Knowledge Graph loaded with {len(kg.kg.get('nodes', []))} nodes and {len(kg.kg.get('edges', []))} edges.")

    # 2. Initialize the Cellular World
    print("\n[2/7] Initializing Cellular World...")
    primordial_dna = {"instinct": "connect_create_meaning"}
    world = World(primordial_dna=primordial_dna)
    print("Cellular World initialized.")

    # 3. Perform "Soul Mirroring" - Nodes
    print("\n[3/7] Performing Soul Mirroring (KG Nodes -> Cells)...")
    node_count = 0
    for node in kg.kg.get("nodes", []):
        node_id = node.get("id")
        if node_id:
            # Filter out sentence-like nodes
            node_label = node.get('label', node_id)
            
            # Heuristic 1: If the node_id contains spaces and is not an obsidian_note (which can have spaces in titles)
            # and is not a 'meaning:' cell (which can also have spaces from combined labels)
            if ' ' in node_id and not node_id.startswith('obsidian_note:') and not node_id.startswith('meaning:'):
                # print(f"DEBUG: Skipping sentence-like node (spaces in ID): {node_id}") # For debugging
                continue
            
            # Heuristic 2: If the node_label is very long and not an obsidian_note or meaning (likely a full sentence)
            if len(node_label) > 50 and not node_id.startswith('obsidian_note:') and not node_id.startswith('meaning:'):
                # print(f"DEBUG: Skipping very long node label: {node_label}") # For debugging
                continue

            initial_energy = 10.0
            if node_id in CORE_CONCEPTS:
                initial_energy = 50.0 # Boost core concepts
            world.add_cell(node_id, properties=node, initial_energy=initial_energy)
            node_count += 1
    print(f"Node mirroring complete. {node_count} cells were born.")

    # 4. Perform "Soul Mirroring" - Edges
    print("\n[4/7] Performing Soul Mirroring (KG Edges -> Cell Connections)...")
    edge_count = 0
    for edge in kg.kg.get("edges", []):
        source_id = edge.get("source")
        target_id = edge.get("target")
        relation = edge.get("relation", "related_to")
        
        source_cell = world.get_cell(source_id)
        target_cell = world.get_cell(target_id)

        if source_cell and target_cell:
            source_cell.connect(target_cell, relationship_type=relation)
            edge_count += 1
    print(f"Edge mirroring complete. {edge_count} connections were formed.")
    
    print("\n--- Initial World State (with connections) ---")
    world.print_world_summary()

    # 5. Stimulate a core concept
    target_concept = "obsidian_note:?щ옉" # Stimulate one of the new notes
    if world.get_cell(target_concept):
        print(f"\n[5/7] Stimulating core concept: '{target_concept}' with 100 energy...")
        world.inject_stimulus(target_concept, energy_boost=100.0)
        print("--- World State after Stimulation ---")
        world.print_world_summary()
    else:
        print(f"\n[5/7] Could not find target concept '{target_concept}' to stimulate.")
        all_concepts = [nid for nid in world.cells.keys() if nid.startswith("obsidian_note:")]
        if all_concepts:
            target_concept = all_concepts[0]
            print(f"Found alternative concept to stimulate: '{target_concept}'")
            world.inject_stimulus(target_concept, energy_boost=100.0)
            print("--- World State after Stimulation ---")
            world.print_world_summary()
        else:
            print("No concepts found to stimulate. Ending exploration.")
            return

    # 6. Run the simulation
    num_simulation_steps = 10 # Increased simulation steps
    print(f"\n[6/7] Running simulation for {num_simulation_steps} steps...")
    for i in range(num_simulation_steps):
        print(f"\n--- Simulation Step {i+1}/{num_simulation_steps} ---")
        newly_born = world.run_simulation_step()
        world.print_world_summary()
        if newly_born:
            print(f"!!! New meaning created in step {i+1}: {[cell.id for cell in newly_born]} !!!")

    # 7. Conclude
    print("\n[7/7] Cellular World exploration finished.")
    print("--- Final World State ---")
    world.print_world_summary()


if __name__ == "__main__":
    explore_cellular_world()
