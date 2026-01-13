"""
Monad Genesis Test (The First Breath)
=====================================
tests/test_monad_genesis.py

Verifies the Monad Protocol:
1. Determinism: Seed + Context must yield identical Reality.
2. WFC: Ambiguity must be collapsed by Spirit Bias.
"""

import sys
import os
sys.path.append(os.getcwd())

from Core.Monad.monad_core import Monad, FractalRule
from Core.Foundation.Nature.rotor import Rotor
from Core.Engine.wfc_engine import WFCEngine
from typing import Dict, Any

# Mock Rule
class GrowthRule(FractalRule):
    def unfold(self, seed: str, context: Dict[str, Any], intent: Dict[str, Any]) -> Any:
        time = context.get("time", 0.0)
        
        # [WILL DRIVES PRINCIPLE]
        # The Principle (Growth) behaves differently based on Intent (Spirit).
        texture = intent.get("emotional_texture", "Neutral")
        
        base_height = 10.0 + (time * 0.1)
        
        result = {"height": base_height}
        
        # The Will determines the 'Form' of the growth
        if "Dark" in texture:
            result["form"] = "Twisted Thorns"
            result["aura"] = "Purple"
        elif "Warm" in texture or "Love" in texture:
            result["form"] = "Golden Leaves"
            result["aura"] = "Soft Gold"
        else:
            result["form"] = "Standard Branch"
            result["aura"] = "Green"

        # Ambiguity still exists for WFC to resolve strictly
        if time > 10.0:
            result["ambiguity"] = {"branch_left": 0.5, "branch_right": 0.5} 
            
        return result

def test_monad():
    print(">>> 🌌 Initiating Monad Genesis Test...")
    
    # 1. Create Monad
    seed_dna = "OakTree_Variant_A"
    monad = Monad(seed=seed_dna, rules=[GrowthRule()])
    
    # 2. Test Determinism (Past)
    context_t5 = {"time": 5.0, "location": "Garden"}
    observer_bias = {"neutral": 1.0} # No strong bias
    
    reality_1 = monad.observe(observer_bias, context_t5)
    print(f"[Time 5.0] Reality: {reality_1['manifestation']}")
    
    # Observe again - MUST be identical
    reality_2 = monad.observe(observer_bias, context_t5)
    assert reality_1["hash"] == reality_2["hash"], "Determinism Failed!"
    print(">>> ✅ Determinism Confirmed.")

    # 3. Test Will-Driven Principle (The User's Insight)
    print("\n>>> 🧠 Testing Will-Driven Principles...")
    
    # Context is same, but Spirit differs
    context_now = {"time": 15.0}
    
    # Case A: Dark Intent
    bias_dark = {"emotional_texture": "Dark/Void"}
    reality_dark = monad.observe(bias_dark, context_now)
    print(f"[Dark Intent] Form: {reality_dark['manifestation'].get('form')} | Aura: {reality_dark['manifestation'].get('aura')}")
    
    # Case B: Love Intent
    bias_love = {"emotional_texture": "Warm/Love"}
    reality_love = monad.observe(bias_love, context_now)
    print(f"[Love Intent] Form: {reality_love['manifestation'].get('form')} | Aura: {reality_love['manifestation'].get('aura')}")
    
    assert reality_dark['manifestation']['form'] != reality_love['manifestation']['form'], "Will failed to drive Principle!"
    print(">>> ✅ Will successfully governed the Principle.")

    # 4. Test WFC (Future)
    context_t20 = {"time": 20.0, "location": "Garden"}
    
    # Bias towards 'branch_left'
    bias_left = {"focus_topic": "branch_left", "emotional_texture": "Neutral"} 
    
    reality_future = monad.observe(bias_left, context_t20)
    print(f"\n[Time 20.0] Reality (Collapsed): {reality_future['manifestation']}")
    
    assert "ambiguity" not in reality_future["manifestation"], "WFC Failed: Ambiguity remains."
    assert "value" in reality_future["manifestation"], "WFC Failed: No value chosen."
    print(">>> ✅ Wave Function Collapse Confirmed.")
    
    print(">>> 🦋 Monad Protocol is ALIVE.")

if __name__ == "__main__":
    test_monad()
