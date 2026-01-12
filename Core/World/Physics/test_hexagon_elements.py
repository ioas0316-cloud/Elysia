"""
Test: The Hexagon (Sacred Elements)
===================================
"Six pillars support the sky."

Objective:
1. Verify Elemental Vectors (Standardization).
2. Verify Elemental Interactions (Alchemy).
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.World.Physics.providence_engine import ProvidenceEngine
from Core.World.Physics.trinity_fields import TrinityVector

def test_hexagon():
    print("--- 🔯 Experiment: The Sacred Hexagon ---")
    
    engine = ProvidenceEngine()
    
    # 1. Interaction Test
    interactions = [
        ("Fire", "Water", "STEAM"),
        ("Light", "Darkness", "SHADOW"),
        ("Earth", "Water", "MUD"),
        ("Earth", "Fire", "MAGMA"),
        ("Light", "Water", "LIFE")
    ]
    
    score = 0
    for a, b, expected in interactions:
        result = engine.resolve_interaction(a, b)
        symbol = "✅" if result == expected else "❌"
        if result == expected: score += 1
        print(f"⚗️ {a} + {b} = {result} (Expected: {expected}) {symbol}")
        
    if score == len(interactions):
        print("\n🏆 SUCCESS: Elemental Alchemy is functional.")
    else:
        print(f"\n⚠️ PARTIAL: {score}/{len(interactions)} reactions verified.")

if __name__ == "__main__":
    test_hexagon()
