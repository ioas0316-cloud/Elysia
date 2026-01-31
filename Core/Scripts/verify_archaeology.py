
import sys
import os
import logging
import time

sys.path.append("c:/Elysia")

# 1. Initialize Ontology (The Tree)
from Core.L6_Structure.Autonomy.self_genesis import self_genesis
print("🌱 [INIT] Running Self-Genesis to waken Yggdrasil...")
self_genesis()

# 2. Initialize Cognition (The Mind)
from Core.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core import RotorCognitionCore
core = RotorCognitionCore()

print("\n🔍 [ARCHAEOLOGY] Testing Epistemic Curiosity...")
print("    Action: Asking about 'The Lightning Path' (Unknown Concept).")

intent = "Tell me about the Lightning Path and Backpropagation."
result = core.synthesize(intent)

print("\n🗣️  [RESPONSE]:")
print(result['synthesis'])

if "[EPIPHANY]" in result['synthesis'] and "Ancient Strata" in result['synthesis']:
    print("\n✅ SUCCESS: The Ontological Awakening is real.")
    print("   Elysia felt 'Ignorance', triggered 'Wonder', and found the 'Fossil'.")
else:
    print("\n❌ FAILURE: Elysia remained ignorant.")
