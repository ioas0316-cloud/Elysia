    
    tree = WorldTree()
    
    # 1. Plant the Seed
    tree.plant_seed("Existence")
    
    # 2. First Growth Cycle (The Big Bang of Concepts)
    print("\n---   Cycle 1: The First Branches ---")
    time_node = tree.grow("Time", parent_concept="Existence")
    space_node = tree.grow("Space", parent_concept="Existence")
    
    # 3. Second Growth Cycle (Differentiation)
    print("\n---   Cycle 2: Differentiation ---")
    tree.grow("Past", parent_concept="Time")
    tree.grow("Future", parent_concept="Time")
    tree.grow("Matter", parent_concept="Space")
    tree.grow("Energy", parent_concept="Space")
    
    # 4. Visualize
    print("\n---   Yggdrasil Structure ---")
    print(tree.visualize())
    
    print("\n---   Genesis Complete ---")

if __name__ == "__main__":
    genesis()
