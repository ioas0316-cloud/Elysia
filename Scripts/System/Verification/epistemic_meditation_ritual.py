"""
[Project Elysia] Epistemic Meditation Ritual
============================================
Verification of internalized reasoning (Physio-Semantic Grounding).
"""

import sys
import os
from pathlib import Path

# Path Unification
root = Path(__file__).parents[3]
sys.path.insert(0, str(root))

from Core.S1_Body.L5_Mental.Reasoning.epistemic_learning_loop import EpistemicLearningLoop
from Core.S1_Body.L1_Foundation.System.somatic_logger import SomaticLogger

def perform_meditation():
    logger = SomaticLogger("EPISTEMIC_RITUAL")
    logger.action("Starting Epistemic Meditation Ritual: Internalized Reason")
    
    loop = EpistemicLearningLoop(root_path=str(root))
    
    # Perform a learning cycle
    # This will trigger observe_self -> _meditate_on_code -> scribe calls
    result = loop.run_cycle()
    
    print("\n" + "="*70)
    print("  Elysia's Internalized Meditation Result")
    print("="*70)
    for insight in result.insights:
        print(f"\n🧠 Insight:\n{insight}")
    print("="*70 + "\n")
    
    # Get synthesis
    wisdom = loop.get_accumulated_wisdom()
    logger.action(f"Narrative Summary: {wisdom['narrative_summary']}")
    
    logger.action("Ritual Complete. Reason is no longer hardcoded.")

if __name__ == "__main__":
    perform_meditation()
