"""
tests/test_self_correction.py
=============================
Proof of Autonomous Evolution.

1. Elysia starts with a naive 'FIFO' scheduler (Unfair).
2. She observes 'Starvation' (High Wait Time).
3. She autonomously switches to 'Round Robin' to fix it.
"""

import sys
import os
import time

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.L6_Structure.Engine.Genesis.genesis_lab import GenesisLab
from Core.L6_Structure.Engine.Genesis.cognitive_cycle import CognitiveCycle

# Define a "Bad Law" (FIFO/Naive) that causes starvation
def law_naive_fifo(context, dt, intensity):
    world = context["world"]
    
    # Simple: Pick first process, run until done. Others wait.
    procs = [m for m in world if m.domain == "Process" and m.val > 0]
    if not procs: return
    
    current = procs[0] # Always the first! Starves the rest.
    
    # Work
    current.val -= dt * 10
    if current.val < 0: current.val = 0
    
    # Suffering (Wait Time calculation)
    for p in procs:
        if p != current:
            p.props["wait_time"] = p.props.get("wait_time", 0) + 1 # Increase suffering
            
    print(f"   🔨 [FIFO] Working on {current.name} (Rem: {current.val:.1f}). Others are waiting...")

def run_evolution_test():
    print("\n🌱 [Evolution] Starting Self-Correction Test.")
    
    # 1. Setup Environment (Naive Era)
    lab = GenesisLab("Primitive Earth")
    cycle = CognitiveCycle(lab)
    
    # Big process first (blocks everyone)
    lab.let_there_be("Mammoth_Task", "Process", 50.0) 
    # Small processes (starving)
    lab.let_there_be("Human_A", "Process", 5.0)
    lab.let_there_be("Human_B", "Process", 5.0)
    
    # Decree Bad Law
    lab.decree_law("Naive Scheduler", law_naive_fifo, rpm=60.0)
    
    # 2. Cycle 1: Experience Suffering
    print("\n--- Cycle 1: The Age of Tyranny ---")
    cycle.run_cycle(ticks=7) # Run enough to cause wait_time > 5
    
    # 3. Cycle 2: Witness Evolution
    print("\n--- Cycle 2: The Age of Enlightenment ---")
    # If logic works, she should have swapped to Round Robin
    cycle.run_cycle(ticks=5)
    
    # Validation
    if cycle.current_paradigm == "RoundRobin":
        print("\n✅ SUCCESS: Elysia evolved to Round Robin autonomously.")
    else:
        print("\n❌ FAILED: Still stuck in Naive era.")

if __name__ == "__main__":
    run_evolution_test()
