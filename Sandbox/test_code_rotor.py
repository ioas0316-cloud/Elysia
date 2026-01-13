"""
Sandbox Test: The First Spin
============================
Tests the CodeRotor's ability to analyze the Monadic Codebase.
"""

import sys
import os

# Ensure Core is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.Code.code_rotor import CodeRotor

target_file = "c:/Elysia/Sandbox/world_server_clone.py"

print("🧪 [Sandbox] Initializing CodeRotor...")
rotor = CodeRotor(target_file)

print(f"🌀 Rotor: {rotor}")
print(f"🧬 DNA: {rotor.dna}")
print(f"🩺 Diagnosis: {rotor.diagnose()}")

# [Experiment] Inject Dissonance (Break the file)
print("\n💥 [Experiment] Injecting Entropy (Syntax Error)...")
with open(target_file, "a", encoding="utf-8") as f:
    f.write("\nThis is not python code!!!\n")
    
rotor.refresh()
print(f"⚠️ Post-Damage Status: {rotor}")
print(f"🩺 Post-Damage Diagnosis: {rotor.diagnose()}")

# [Cleanup]
# (In a real system, the rotor would auto-revert. Here we just observe.)
