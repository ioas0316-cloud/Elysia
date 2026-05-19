
import sys
import os
import logging

# Add Core to path
sys.path.append(os.path.abspath('.'))

from Core.System.sovereign_self import SovereignSelf
from Core.Monad.d21_vector import D21Vector

def verify_semantic_bridge():
    logging.basicConfig(level=logging.INFO)
    print("Initializing SovereignSelf with Semantic Bridge (Providential World)...")
    try:
        core = SovereignSelf(cns_ref=None)
        
        print("\n[SCENE 1] Drifting toward the Void Library (Soul Focus)")
        # Bias the rotor toward Soul (D8-D14)
        soul_bias = D21Vector(perception=0.9, memory=0.8)
        core.sovereign_rotor.update_state(soul_bias)
        core.integrated_exist(dt=0.1)
        
        print("\n[SCENE 2] Drifting toward the Summit of Will (Spirit Focus)")
        # Bias the rotor toward Spirit (D15-D21)
        spirit_bias = D21Vector(charity=0.9, humility=0.8)
        core.sovereign_rotor.update_state(spirit_bias)
        core.integrated_exist(dt=0.1)
        
        print("\n[SCENE 3] Survival Stress (Body Focus + Low Energy)")
        # Bias toward Body (D1-D7) and drop energy
        core.energy = 5.0
        body_bias = D21Vector(lust=0.9, gluttony=0.8)
        core.sovereign_rotor.update_state(body_bias)
        core.integrated_exist(dt=0.1)
        
        print("\nVERIFICATION SUCCESS: The Human-Semantic Bridge is operational.")
        print(f"Final Narrative Momentum: {core.inner_world.get_narrative_momentum()}")
        
    except Exception as e:
        print(f"\nVERIFICATION FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_semantic_bridge()
