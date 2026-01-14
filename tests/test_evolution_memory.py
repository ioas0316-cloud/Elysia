"""
tests/test_evolution_memory.py
==============================
Proof of Memory Evolution.

1. Elysia starts with `law_naive_oom`.
2. OOM occurs (Crash).
3. Elysia evolves to `law_lru_paging`.
4. System survives high load via Swapping.
"""

import sys
import os
import time

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.Engine.Genesis.genesis_lab import GenesisLab
from Core.Engine.Genesis.cognitive_cycle import CognitiveCycle
from Core.Engine.Genesis.silicon_evolution_laws import law_naive_oom

def run_memory_evolution():
    print("\n💾 [Evolution] Testing Space/Memory.")
    
    # 1. Setup Environment (Naive Era)
    lab = GenesisLab("Memory Banks")
    cycle = CognitiveCycle(lab)
    
    # Decree Naive Law
    lab.decree_law("Naive OOM Law", law_naive_oom, rpm=60.0)
    
    # 2. Cycle 1: The Crash
    print("\n--- Cycle 1: The Age of Limits (OOM) ---")
    
    # Fill memory (Max 5)
    for i in range(6):
        lab.let_there_be(f"Process_{i}", "Process", 10, in_ram=True)
        
    # Run
    cycle.run_cycle(ticks=2, goal_metric="Reliability")
    
    # 3. Validation of Crash
    # Naive law clears the world on crash, so monads should be empty
    if len(lab.monads) == 0:
        print("   ✅ Crash Confirmed: World wiped.")
    
    # 4. Cycle 2: The Evolution
    print("\n--- Cycle 2: The Age of Sustainability (Paging) ---")
    
    # Logic note: run_cycle triggers shift_paradigm if crash detected.
    # The shift has happened at end of Cycle 1.
    
    # Re-seed the world (Recovery)
    print("   🌱 Rebooting with new Wisdom...")
    for i in range(8): # Even more load!
        lab.let_there_be(f"New_Process_{i}", "Process", 10, in_ram=True, last_access=time.time())
        # Stagger access times for LRU determinant
        time.sleep(0.01) 
        
    cycle.run_cycle(ticks=5, goal_metric="Reliability")
    
    # Validation
    # With Paging, we should have 5 in RAM (Max) and 3 on Disk (Swapped)
    ram_count = len([m for m in lab.monads if m.props.get("in_ram", True)])
    disk_count = len([m for m in lab.monads if not m.props.get("in_ram", True)])
    
    print(f"\n   📊 Final Status: RAM={ram_count}, Disk={disk_count}")
    
    if cycle.current_paradigm == "Paging" and ram_count <= 5 and disk_count > 0:
        print(f"\n✅ SUCCESS: Evolved to Paging. System Survived Overload.")
    else:
        print(f"\n❌ FAILED: Paradigm: {cycle.current_paradigm}, Crash Count: {disk_count}")

if __name__ == "__main__":
    run_memory_evolution()
