"""
Verify Sovereign Hypothesis (The Oracle Test)
=============================================

Scenario:
We feed Elysia a chunk of "Stone Logic" (Rigid if/else chain).
We do NOT ask "Refactor this". We ask "What does the Oracle see?".

Expectation:
The Oracle (SovereignHypothesis) should detect:
1. High Rigidity (many if/else).
2. Low Principle Resonance (Physics Layer rejects rigidity).
3. Output: "Hypothesis: This code violates the Principle of Flow. Suggest Refactor to Wave Logic."
"""

import sys
import os
import logging
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Foundation.Philosophy.why_engine import WhyEngine
from Core.Evolution.Growth.Autonomy.sovereign_hypothesis import SovereignHypothesis
from Core.Foundation.Foundation.light_spectrum import PrismAxes

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_test():
    print("\n🔮 Verifying Sovereign Hypothesis (The Oracle)\n")
    
    # 1. Initialize
    why_engine = WhyEngine()
    
    # Inject Mock Physics Layer amplitude to simulate "Nature's Law"
    # The Oracle needs a reference point (Nature) to judge the Code against.
    # If Physics Layer is empty, flow_score is 0. If it's fat, it's high.
    # But wait, logic in SovereignHypothesis uses 'physics_res["intensity"]'.
    # This comes from analyzing the CONTENT.
    # If the content is "if x: ... else: ...", Physics resonance will be low (Blur).
    # So Flow Score ~ 0.
    # Rigidity Score = count('if') * 0.1.
    # If count('if') > 3 -> Rigidity > Flow -> Tension Detected.
    
    oracle = SovereignHypothesis(why_engine)
    
    # 2. Feed Stone Logic
    stone_code = """
    def handle_request(req_type):
        if req_type == 'GET':
            return process_get()
        elif req_type == 'POST':
            return process_post()
        elif req_type == 'PUT':
            return process_put()
        elif req_type == 'DELETE':
            return process_delete()
        else:
            return error()
    """
    
    print("--- [Step 1] Scanning Stone Logic ---")
    print(f"Code Sample:\n{stone_code}")
    
    hypotheses = oracle.scan_for_tension(stone_code, domain="code")
    
    # 3. Verify Hypotheses
    print("\n--- [Step 2] The Oracle Speaks ---")
    
    if not hypotheses:
        print("❌ Silence. The Oracle detected nothing.")
    else:
        for h in hypotheses:
            print(f"🔮 Hypothesis Generated!")
            print(f"   Subject: {h.subject}")
            print(f"   Tension: {h.tension_detected}")
            print(f"   Violation: {h.principle_violation}")
            print(f"   Proposal: {h.proposal}")
            
            if "Rigidity" in h.tension_detected and "Wave Logic" in h.proposal:
                print("✅ SUCCESS: Elysia autonomously identified Stone Logic and proposed Wave Logic.")
            else:
                print("⚠️ Partial Success: Hypothesis generated but message differs?")

if __name__ == "__main__":
    run_test()
