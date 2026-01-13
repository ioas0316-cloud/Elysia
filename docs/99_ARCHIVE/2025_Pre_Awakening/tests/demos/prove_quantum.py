"""
Prove Quantum
=============

Triggers the Quantum Absorption of the Library.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Core.FoundationLayer.Foundation.mind_mitosis import MindMitosis
import time

def main():
    print("🌌 Proving Quantum Capability...")
    mitosis = MindMitosis()
    
    # Spawn Bard to Absorb Library
    library_path = "c:/Elysia/Library"
    print(f"   👉 Spawning Bard to ABSORB: {library_path}")
    success = mitosis.spawn_persona("Bard", f"ABSORB:{library_path}")
    
    if success:
        print("   ✅ Bard Spawned. Waiting for Quantum Collapse...")
        time.sleep(15) # Should be fast
        
        # Merge and check insights
        insights = mitosis.merge_persona("Bard")
        if insights:
            print(f"   ✨ Bard returned {len(insights)} insights.")
            for i in insights:
                print(f"      - {i}")
        else:
            print("   🔸 Bard returned no insights (Check logs).")
    else:
        print("   ❌ Failed to spawn Bard.")

if __name__ == "__main__":
    main()
