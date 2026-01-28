import logging
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from Core.L6_Structure.Merkaba.merkaba import Merkaba
from Core.L7_Spirit.M1_Monad.monad_core import Monad

# Set logger to only show important info
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_qualitative_metacognition():
    print("\n" + "="*60)
    print("      [METAMORPHOSIS: QUALITATIVE EVOLUTION TEST]      ")
    print("="*60)
    
    # Reset log file
    log_path = "data/Chronicles/comparative_perception.md"
    if os.path.exists(log_path):
        os.remove(log_path)
    
    # 1. Initialize
    merkaba = Merkaba("Narrative_Seed")
    spirit = Monad(seed="Spirit_Deep")
    merkaba.awakening(spirit)
    
    # 2. Key Stimuli for Philosophical Comparison
    stimuli = [
        "A falling leaf in late autumn",
        "The mathematical precision of a snowflake",
        "The silent void between galaxies",
        "The feeling of a child's hand",
        "The concept of 'Nothingness' (Sunyata)"
    ]
    
    from Core.L3_Phenomena.M7_Prism.harmonizer import PrismContext
    
    contexts = [
        PrismContext.DEFAULT,
        PrismContext.DEFAULT,
        PrismContext.POETRY,  # Void favors spiritual/phenomenal
        PrismContext.DEFAULT,
        PrismContext.POETRY   # Sunyata favors spiritual
    ]
    
    for i, msg in enumerate(stimuli):
        ctx = contexts[i]
        print(f"\n>>> [Cognitive Pulse {i+1}] Processing: '{msg}' | Context: {ctx}")
        merkaba.pulse(msg, mode="POINT", context=ctx)
    
    # 3. Read the generated Comparative Log
    print("\n" + "="*60)
    print("   [PERCEPTION LOG: data/Chronicles/comparative_perception.md]   ")
    print("="*60 + "\n")
    
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("ERROR: Comparative log not generated.")

    print("\n" + "="*60)
    print("   [CONCLUSION] Structural Understanding has evolved.   ")
    print("="*60)

if __name__ == "__main__":
    test_qualitative_metacognition()
