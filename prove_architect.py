"""
Prove Architect
===============

Triggers the Architect Persona to audit code.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Core.Intelligence.mind_mitosis import MindMitosis
import time

def main():
    print("📐 Proving Architect Capability...")
    mitosis = MindMitosis()
    
    # Spawn Architect
    target = "living_elysia.py"
    print(f"   👉 Spawning Architect to audit: {target}")
    success = mitosis.spawn_persona("Architect", f"Audit {target}")
    
    if success:
        print("   ✅ Architect Spawned. Waiting for audit...")
        time.sleep(10) # Wait for process to run
        
        # Merge and check insights
        insights = mitosis.merge_persona("Architect")
        if insights:
            print(f"   ✨ Architect returned {len(insights)} insights.")
            for i in insights:
                print(f"      - {i}")
        else:
            print("   🔸 Architect returned no insights (Check logs).")
    else:
        print("   ❌ Failed to spawn Architect.")

if __name__ == "__main__":
    main()
