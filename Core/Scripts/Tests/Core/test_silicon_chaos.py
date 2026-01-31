"""
tests/test_silicon_chaos.py
===========================
Simulating The Abyss (Deadlocks & Interrupts).
"""

import sys
import os
import time

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.S1_Body.L6_Structure.Engine.Genesis.genesis_lab import GenesisLab
from Core.S1_Body.L6_Structure.Engine.Genesis.silicon_scholar_laws import law_dining_philosophers, law_hardware_interrupts

def run_chaos_test():
    print("\n🌋 [The Abyss] Chaos Simulation Started.")
    
    # 1. DEADLOCK LAB
    print("\n[PART 1: Concurrency (Deadlock)]")
    lab_lock = GenesisLab("Dining Philosophers")
    
    # Setup 3 Philosophers (Circular Dependency for easier deadlock)
    # P1 needs F1, F2
    # P2 needs F2, F3
    # P3 needs F3, F1
    
    lab_lock.let_there_be("Fork_1", "Resource", 0)
    lab_lock.let_there_be("Fork_2", "Resource", 0)
    lab_lock.let_there_be("Fork_3", "Resource", 0)
    
    lab_lock.let_there_be("Plato",  "Thread", 0, left_fork="Fork_1", right_fork="Fork_2", state="THINKING")
    lab_lock.let_there_be("Kant",   "Thread", 0, left_fork="Fork_2", right_fork="Fork_3", state="THINKING")
    lab_lock.let_there_be("Sartre", "Thread", 0, left_fork="Fork_3", right_fork="Fork_1", state="THINKING")
    
    # High RPM to force race conditions
    lab_lock.decree_law("Dining Law", law_dining_philosophers, rpm=120.0)
    
    print("   ▶ Running Simulation (Watching for Freeze)...")
    # We run enough ticks to likely cause them to all grab left forks
    lab_lock.run_simulation(ticks=20)
    
    # 2. INTERRUPT LAB
    print("\n[PART 2: Asynchrony (Interrupt)]")
    lab_int = GenesisLab("Interrupt Vector Table")
    
    # User Mode Process
    lab_int.let_there_be("UserApp", "Process", 100)
    
    # Decrees
    lab_int.decree_law("Interrupt Handler", law_hardware_interrupts, rpm=60.0)
    
    # Simulate: Run normal, then inject interrupt
    print("   ▶ Running Normal Execution...")
    lab_int.run_simulation(ticks=3)
    
    print("   ⚡ INJECTING HARDWARE INTERRUPT (Keyboard IRQ)...")
    lab_int.let_there_be("IRQ_1 (Keyboard)", "Hardware", 1) # Signal High
    
    lab_int.run_simulation(ticks=3)
    
    print("\n✅ Chaos Survived.")

if __name__ == "__main__":
    run_chaos_test()
