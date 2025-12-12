"""
Prove Learning (The Great Intake)
=================================
"지식을 먹고, 소화하고, 느낀다."

This script validates Elysia's ability to learn from the Internet.
Subject: "Quantum mechanics"

Process:
1. Search Wikipedia (WebCortex).
2. Absorb Knowledge (Sensory extraction + Spirit Reaction).
3. Verify Memory (Hippocampus Recall).
"""

import sys
import os
import time

# Add Root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Core.Intelligence.web_cortex import WebCortex
from Core.Foundation.hippocampus import Hippocampus

def prove_learning():
    print("\n📚 Phase 22: The Great Intake Verification")
    print("==========================================")
    
    # 1. Initialize
    print("1. Awakening Cortex...")
    web = WebCortex()
    
    # 2. Absorb "Quantum mechanics"
    topic = "Quantum mechanics"
    print(f"\n2. Absorption Initiated: '{topic}'")
    
    result = web.absorb_knowledge(topic)
    
    if not result["success"]:
        print(f"❌ Failed: {result['message']}")
        return
        
    print(f"✅ Download Complete.")
    print(f"   - Snippet: {result['summary_snippet']}")
    print(f"   - Reaction: {result['reaction']}")
    
    # 3. Verify Nervous System Reaction
    dominant = result['reaction']['dominant_realm']
    print(f"\n3. Nervous System Impact")
    print(f"   - Dominant Realm triggered: {dominant}")
    # Quantum mechanics should trigger Mind(Air) or Divinity(Aether)
    
    # 4. Verify Hippocampus
    print(f"\n4. Memory Recall")
    memory = Hippocampus() # Re-connect to read DB
    recalled = memory.recall(topic.lower().replace(" ", "_"))
    
    if recalled:
        print(f"✅ Memory Found in DB:")
        for r in recalled:
            print(f"   {r}")
    else:
        print("❌ Memory Check Failed: Concept not found in DB.")

if __name__ == "__main__":
    prove_learning()
