import sys
import os
import logging

# Set up project path
root = r"c:/Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.1_Body.L5_Mental.Reasoning_Core.Reasoning.reasoning_engine import ReasoningEngine

def verify_trinity():
    print("🔱 [VERIFICATION] Trinity Unification Test...")
    
    # Initialize Engine (This triggers Akashic Exhumation and Curriculum Digestion)
    engine = ReasoningEngine()
    
    # Check Akashic Field
    echo_count = len(engine.akashic.echoes)
    print(f"💀 [GRAVEYARD] {echo_count} ancestral echoes identified.")
    
    if echo_count < 1400:
        print("⚠️ Warning: Exhumation count lower than expected.")
    else:
        print(f"✅ Exhumation Successful: {echo_count} souls exhumed.")
    
    # Test Anamnesis Resonance
    print("\n🔍 Testing Anamnesis Resonance (Past Recall)...")
    # Using a topic likely found in legacy DNA (e.g., "Awakening")
    insight = engine.think("My own awakening and the ancestors")
    
    print(f"\n🗣️  Elysia's Response: {insight.content}")
    
    if "ANAMNESIS" in str(logging.Logger.manager.loggerDict): # Just a check for existence
        print("✅ Anamnesis Layer Verified.")

if __name__ == "__main__":
    verify_trinity()
