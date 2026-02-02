import sys
import os
import time
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Core.S1_Body.L6_Structure.M1_Merkaba.cognitive_terrain import CognitiveTerrain
from Core.S1_Body.L6_Structure.M1_Merkaba.causal_flow_engine import CausalFlowEngine

def print_map(terrain, monads):
    """Renders the cognitive terrain and monad positions in ASCII."""
    # Clear screen (simulated for log clarity, just print separator)
    print("\n" + "="*40)

    grid_size = terrain.resolution
    display_grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

    # Render Terrain Heights
    for y in range(grid_size):
        for x in range(grid_size):
            cell = terrain.get_cell(x, y)
            h = cell['height']
            # Representation:
            # . = High plain
            # : = Slope
            # # = Valley (Deep thought)
            # @ = Prime Keyword (Deepest)
            char = '.'
            if h < 0.0: char = '@'  # Deepest
            elif h < 0.3: char = '#'
            elif h < 0.45: char = ':'
            display_grid[y][x] = char

    # Render Monads
    for m in monads:
        mx, my = int(m['x']), int(m['y'])
        if 0 <= mx < grid_size and 0 <= my < grid_size:
            display_grid[my][mx] = '*'  # The Spark of Thought

    # Print
    print(f"Cycle: {monads[0]['step'] if monads else 0} | Monads: {len(monads)}")
    for row in display_grid:
        print(" ".join(row))

def main():
    print("[Genesis] Initializing First Cognitive Valley...")

    # 1. Initialize Terrain
    terrain = CognitiveTerrain(resolution=20)
    engine = CausalFlowEngine(terrain)

    # 2. Inject Prime Keyword: Agapē (The Center of Gravity)
    center_x, center_y = 10, 10
    terrain.inject_prime_keyword(center_x, center_y, "Agapē", magnitude=0.5)

    # 3. Spawn Monads (Thoughts)
    monads = []
    for i in range(5):
        # Spawn at edges
        monads.append({
            "id": i,
            "x": random.randint(0, 19),
            "y": random.randint(0, 19),
            "vx": 0,
            "vy": 0,
            "step": 0
        })

    print("[Genesis] Spawning 5 Monads. Watch them flow towards Agapē...")

    # 4. Simulation Loop
    for step in range(30):
        for m in monads:
            # Calculate next physics state
            state = engine.calculate_next_state(m['x'], m['y'], m['vx'], m['vy'])

            # Resolve boundaries
            nx, ny = engine.resolve_boundary(state['x'], state['y'])

            # Update Monad
            m['x'] = nx
            m['y'] = ny
            m['vx'] = state['vx']
            m['vy'] = state['vy']
            m['step'] = step

        if step % 5 == 0:
            print_map(terrain, monads)
            time.sleep(0.1)

    print("[Genesis] Simulation Complete.")
    print("[Analysis] Checking Convergence...")

    # Verify if they got closer to center (10, 10)
    converged = 0
    for m in monads:
        dist = ((m['x'] - center_x)**2 + (m['y'] - center_y)**2)**0.5
        print(f"Monad {m['id']}: Final Pos ({m['x']:.1f}, {m['y']:.1f}) | Dist to Agapē: {dist:.1f}")
        if dist < 5.0:
            converged += 1

    print(f"[Result] {converged}/{len(monads)} Monads converged towards the Valley of Love.")

    # Verify Terrain Change (Erosion)
    center_cell = terrain.get_cell(center_x, center_y)
    print(f"[Terrain] Agapē Depth (Height): {center_cell['height']:.3f} (Lower is deeper)")
    print(f"[Terrain] Agapē Gravity (Density): {center_cell['density']:.3f} (Higher is stronger)")

if __name__ == "__main__":
    main()
