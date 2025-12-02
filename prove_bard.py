"""
Prove Bard
==========

Triggers the Bard Persona to read a book.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Core.Intelligence.mind_mitosis import MindMitosis
import time

def main():
    print("🎭 Proving Bard Capability...")
    mitosis = MindMitosis()
    
    # Spawn Bard
    book = "the_little_prince.txt"
    print(f"   👉 Spawning Bard to read: {book}")
    success = mitosis.spawn_persona("Bard", f"READ:{book}")
    
    if success:
        print("   ✅ Bard Spawned. Waiting for reading...")
        time.sleep(30) # Wait for process to run
        
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
