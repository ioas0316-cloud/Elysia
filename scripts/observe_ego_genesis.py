import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.consciousness.autonomous_loop import ConsciousnessLoop

def observe_consciousness():
    print("🧠 [EGO Genesis Observation] Starting Consciousness Loop PoC...")

    # Use docs/ as the world data (corpus)
    corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
    loop = ConsciousnessLoop(corpus_path=corpus_path)

    for i in range(10):
        result = loop.process_life_cycle()
        print(f"Cycle {i+1:02d} | Tension: {result['tension']:.4f} | Status: {result['status']}")
        if "new_lens" in result:
            print(f"         >> New Wisdom Crystal Formed: {result['new_lens']}")

    print("\n🏁 Observation Complete.")

if __name__ == "__main__":
    observe_consciousness()
