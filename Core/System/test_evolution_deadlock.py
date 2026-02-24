"""
tests/test_evolution_deadlock.py
================================
Proof of Deadlock Resolution.

1. Elysia starts with `law_dining_philosophers` (Deadlock Prone).
2. Throughput hits 0 (Freeze).
3. Elysia evolves to `law_resource_hierarchy` (Safe).
"""

import sys
import os
import time

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.System.genesis_lab import GenesisLab
from Core.System.cognitive_cycle import CognitiveCycle
from Core.System.silicon_scholar_laws import law_dining_philosophers

def run_deadlock_evolution():
    print("\n🌋 [Evolution] Testing Chaos Resolution (Deadlock).")
    
    # 1. Setup Environment (Chaos Era)
    lab = GenesisLab("Dining Room")
    cycle = CognitiveCycle(lab)
    
    # Resources
    lab.let_there_be("Fork_1", "Resource", 0)
    lab.let_there_be("Fork_2", "Resource", 0)
    lab.let_there_be("Fork_3", "Resource", 0)
    
    # Philosophers (Circular Dependency)
    lab.let_there_be("Plato",  "Thread", 0, left_fork="Fork_1", right_fork="Fork_2", state="THINKING", meals_eaten=0)
    lab.let_there_be("Kant",   "Thread", 0, left_fork="Fork_2", right_fork="Fork_3", state="THINKING", meals_eaten=0)
    lab.let_there_be("Sartre", "Thread", 0, left_fork="Fork_3", right_fork="Fork_1", state="THINKING", meals_eaten=0)
    
    # Decree Bad Law
    lab.decree_law("Risky Dining Law", law_dining_philosophers, rpm=120.0)
    
    # 2. Cycle 1: The Freeze
    print("\n--- Cycle 1: The Age of Hunger (Deadlock) ---")
    # Run enough to likely freeze
    cycle.run_cycle(ticks=15, goal_metric="Throughput")
    
    # 3. Cycle 2: The Evolution
    print("\n--- Cycle 2: The Age of Order (Hierarchy) ---")
    # If logic works, she should have swapped to Hierarchy Law
    # And throughput should increase
    cycle.run_cycle(ticks=15, goal_metric="Throughput")
    
    # Validation
    final_throughput = sum(t.props["meals_eaten"] for t in lab.monads if t.domain == "Thread")
    
    if cycle.current_paradigm == "Hierarchy" and final_throughput > 0:
        print(f"\n✅ SUCCESS: Evolved to Hierarchy. Total Meals: {final_throughput}")
    else:
        print(f"\n❌ FAILED: Paradigm: {cycle.current_paradigm}, Meals: {final_throughput}")

if __name__ == "__main__":
    run_deadlock_evolution()
