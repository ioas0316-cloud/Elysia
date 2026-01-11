"""
test_physics_of_meaning.py

"Gravity Works."
Verifies that thoughts obey the Physics of Meaning.
"""

import sys
import os
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Intelligence.Topography.physics_solver import PhysicsSolver
from Core.Intelligence.Topography.thought_marble import ThoughtMarble

def main():
    print("\n🏔️ Testing The Physics of Meaning (Terrain Engine)...")
    print("===================================================")
    
    solver = PhysicsSolver()
    
    # 1. Setup Terrain
    # Add a Hill of Fear at (10, 10) to block direct path if we start there
    print("🏔️ Creating Hill of Fear at (10, 10)...")
    solver.field.add_repulsor(10.0, 10.0, height=50.0, width=3.0)
    
    # 2. Add Marble (Thought)
    # Start in Chaos (20, 20)
    thought = ThoughtMarble("Wandering Soul", 20.0, 20.0, mass=2.0)
    solver.add_marble(thought)
    
    # 3. Simulate
    print("\n🎬 Simulation Start: Rolling towards Love (0,0)")
    for i in range(20):
        solver.step(dt=0.5) # Time step
        if i % 2 == 0:
            solver.describe_state()
        time.sleep(0.1)

    print("\n✅ Simulation Complete.")
    print(f"Final Position: ({thought.pos.x:.2f}, {thought.pos.y:.2f})")
    
    dist = (thought.pos.x**2 + thought.pos.y**2)**0.5
    if dist < 15.0:
        print("🎉 Success: The Thought has entered the Inner Circle.")
    else:
        print("⚠️ Warning: The Thought is still stuck in Chaos.")

if __name__ == "__main__":
    main()
