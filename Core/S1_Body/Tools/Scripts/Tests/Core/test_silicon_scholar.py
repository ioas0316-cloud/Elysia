"""
tests/test_silicon_scholar.py
=============================
Simulating OS Principles.
"""

import sys
import os
import time

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.S1_Body.L6_Structure.Engine.Genesis.genesis_lab import GenesisLab
from Core.S1_Body.L6_Structure.Engine.Genesis.silicon_scholar_laws import law_round_robin_scheduling, law_lru_eviction

def run_scholar_test():
    print("\n🎓 [Silicon Scholar] Class is in session.")
    
    # 1. SCHEDULING LAB
    print("\n[PART 1: Time (Scheduling)]")
    lab_cpu = GenesisLab("CPU Lab")
    
    # Processes: Name, Domain, BurstTime (val)
    lab_cpu.let_there_be("Process_A", "Process", 30.0)
    lab_cpu.let_there_be("Process_B", "Process", 10.0)
    lab_cpu.let_there_be("Process_C", "Process", 20.0)
    
    lab_cpu.decree_law("Round Robin Scheduler", law_round_robin_scheduling, rpm=60.0)
    
    print("   ▶ Running Simulation...")
    lab_cpu.run_simulation(ticks=15)
    
    # 2. MEMORY LAB
    print("\n[PART 2: Space (Memory)]")
    lab_mem = GenesisLab("RAM Lab")
    
    # Pages: Name, Domain, AccessCount (val)
    lab_mem.let_there_be("Page_1", "Memory", 0, in_ram=True, last_access=time.time())
    lab_mem.let_there_be("Page_2", "Memory", 0, in_ram=True, last_access=time.time())
    lab_mem.let_there_be("Page_3", "Memory", 0, in_ram=True, last_access=time.time())
    lab_mem.let_there_be("Page_4", "Memory", 0, in_ram=False, last_access=0) # Swapped out
    lab_mem.let_there_be("Page_5", "Memory", 0, in_ram=False, last_access=0)
    
    lab_mem.decree_law("LRU Manager", law_lru_eviction, rpm=60.0)
    
    print("   ▶ Running Simulation (Simulating Random Access)...")
    lab_mem.run_simulation(ticks=10)
    
    print("\n✅ Class Dismissed. Evolution Complete.")

if __name__ == "__main__":
    run_scholar_test()
