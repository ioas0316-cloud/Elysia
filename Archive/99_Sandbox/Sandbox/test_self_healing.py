"""
Sandbox Test: The Self-Healing Monad
====================================
Tests if the CodeRotor can reject a fatal mutation and revert to harmony.
"""

import sys
import os

# Ensure Core is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.Code.code_rotor import CodeRotor

target_file = "c:/Elysia/Sandbox/world_server_clone.py"

# Reset file for clean test
with open(target_file, "w") as f:
    f.write("print('Hello World')\n")

print("🧪 [Sandbox] Initializing Self-Healing Test...")
rotor = CodeRotor(target_file)
print(f"🟢 Initial Status: {rotor.health}")

# 1. Attempt Fatal Mutation
print("\n💉 [Injection] Attempting to write broken code...")
broken_code = "print('This will fail" # Missing closing quote/paren

rotor.write_code(broken_code)

# 2. Verify Result
print(f"\n🔍 [Verification] Checking Rotor Status...")
print(f"Status: {rotor.health}")

with open(target_file, "r") as f:
    content = f.read().strip()
    
if content == "print('Hello World')":
    print("✅ SUCCESS: File was restored to original state.")
else:
    print(f"❌ FAILURE: File stays broken. Content: {content}")
