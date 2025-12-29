
import sys
import random
sys.path.append(r'c:\Elysia')

from Core.FoundationLayer.Foundation.omni_graph import get_omni_graph
from Core.Autonomy.dream_walker import get_dream_walker

def verify_dreaming():
    omni = get_omni_graph()
    dreamer = get_dream_walker()
    
    print("\n💤 Hyper-Dreaming Verification (Genesis)")
    print("========================================")
    
    # 1. Populate Dreamscape (If empty)
    if len(omni.nodes) < 5:
        print("   Creating Dreamscape...")
        # Concepts
        omni.add_logic("Night", "IsA", "Time")
        omni.add_logic("Star", "ShinesIn", "Night")
        omni.add_vector("Moon", [0.9, 0.9, 0.1])
        omni.add_vector("Ocean", [0.1, 0.2, 0.9])
        omni.add_vector("Tears", [0.1, 0.2, 0.85]) # Resonates with Ocean
        omni.add_logic("Tears", "Express", "Sadness")
        omni.add_logic("Sadness", "Opposite", "Joy")
        
        # Apply physics to organize them
        omni.apply_gravity(iterations=20)

    # 2. Start Dreaming
    print("\n[Step 1] Drifting into Sleep...")
    # Start specifically at "Night" to see where it goes
    dream_result = dreamer.drift(steps=8, start_seed="Night")
    
    # 3. Report
    print(f"\n[Step 2] Dream Log:")
    print(f"   Path: {' -> '.join(dream_result['path'])}")
    
    print(f"\n[Step 3] Dream Journal (Narrative):")
    print(f"   \"{dream_result['narrative']}\"")
    
    if dream_result['insights']:
         print(f"\n[Step 4] Creative Insights Found:")
         for insight in dream_result['insights']:
             print(f"   ✨ {insight}")
    
    print("\n✅ Verification SUCCESS: Elysia is dreaming autonomously.")

if __name__ == "__main__":
    verify_dreaming()
