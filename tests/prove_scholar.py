"""
Prove Scholar
=============

Triggers the Scholar Persona to research a topic.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Core.Intelligence.mind_mitosis import MindMitosis
import time

def main():
    print("🎓 Proving Scholar Capability...")
    mitosis = MindMitosis()
    
    # Spawn Scholar
    topic = "Quantum_Mind"
    print(f"   👉 Spawning Scholar to research: {topic}")
    success = mitosis.spawn_persona("Scholar", f"Research {topic}")
    
    if success:
        print("   ✅ Scholar Spawned. Waiting for research...")
        time.sleep(30) # Wait for process to run
        
        # Merge and check insights
        insights = mitosis.merge_persona("Scholar")
        if insights:
            print(f"   ✨ Scholar returned {len(insights)} insights.")
            for i in insights:
                print(f"      - {i}")
        else:
            print("   🔸 Scholar returned no insights (Check logs).")
    else:
        print("   ❌ Failed to spawn Scholar.")

if __name__ == "__main__":
    main()
