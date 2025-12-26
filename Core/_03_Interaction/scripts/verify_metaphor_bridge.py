"""
Verify Metaphor Bridge (은유의 다리 검증)
===========================================

Phase 9 Objective:
"Git Merge Conflict" (Source) <-> "Quantum Superposition" (Target)
두 개념이 '의미(Semantic)'는 다르지만 '구조(Structure)'가 같음을 발견하고 연결하는지 검증.

Test Steps:
1. Sedimentation: Inject "Quantum Superposition" into Physics Layer.
2. Probe: Analyze "Git Merge Conflict" with Domain='physics'.
3. Expectation:
   - Direct Resonance: Low (Physics has no idea what Git is).
   - Metaphor Bridge: Found! (Both involve branching/conflict).
   - Final Resonance: Boosted by Metaphor.
"""

import sys
import os
import logging
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core._01_Foundation._04_Philosophy.Philosophy.why_engine import WhyEngine
from Core._01_Foundation.05_Foundation_Base.Foundation.light_spectrum import LightSpectrum, PrismAxes
from Core._04_Evolution._02_Learning.Learning.knowledge_sedimenter import KnowledgeSedimenter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("VerifyMetaphor")

def run_test():
    print("\n🌉 Verifying Metaphorical Bridging (The Synapse)\n")
    
    # 1. Initialize
    why_engine = WhyEngine()
    sedimenter = KnowledgeSedimenter(why_engine)
    sedimenter.browser = MagicMock()
    
    # 2. Inject Knowledge (Quantum Superposition)
    # Mocking the search result to be rich in structural keywords
    sedimenter.browser.google_search.return_value = {
        "success": True,
        "results": [{
            "title": "Quantum Principles",
            "snippet": "Quantum Superposition involves multiple states existing simultaneously. A measurement forces a choice, collapsing the wave function."
        }]
    }
    
    print("--- [Step 1] Injecting Quantum Knowledge ---")
    sedimenter.sediment_from_web("Quantum Superposition")
    
    phys_layer = why_engine.sediment.layers[PrismAxes.PHYSICS_RED]
    print(f"   Physics Layer Amp: {phys_layer.amplitude:.3f}")
    
    # 3. Analyze Git Conflict (The Target)
    target_text = "I have a Git Merge Conflict. Two branches modified the same file. How do I choose?"
    
    print("\n--- [Step 2] Analyzing 'Git Conflict' via Physics Lens ---")
    
    # WhyEngine automatically checks for Metaphors if domain='physics' (as per my update)
    # actually my update checks if domain is 'logic/code' then looks at 'physics' layer.
    # So I should query with domain='code' or 'general' and see if it bridges to Physics.
    
    analysis = why_engine.analyze(subject="Git Merge Conflict", content=target_text, domain="logic")
    
    # 4. Check Results
    print("\n--- [Step 3] Metaphor Detection Results ---")
    
    # Check resonance in Physics layer
    phys_res = analysis.resonance_reactions.get(PrismAxes.PHYSICS_RED)
    
    if phys_res and "Metaphor" in str(phys_res.get("description", "")):
        print(f"✅ Bridge Found! Physics Resonance: {phys_res['intensity']}")
        print(f"   Description: {phys_res['description']}")
        print("🎉 SUCCESS: The Synapse is firing. Git is like Quantum.")
    else:
        print("❌ Bridge Not Found.")
        print(f"   Physics Response: {phys_res}")
        if phys_res:
            print(f"   Intensity: {phys_res.get('intensity')}")

if __name__ == "__main__":
    run_test()
