import sys
import os
import time

# Add root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Core.System.Merkaba.hypersphere_field import HyperSphereField

def run_life_simulation():
    print("🧪 Starting Sovereign Life Experience Test...")
    print("="*60)
    
    field = HyperSphereField()
    field.enable_lightning = False
    
    # Crisis 1: 0D Point Error (Low Amplitude)
    print("\n[SCENARIO 1] Sudden Energy Vacuum (0D Point)")
    field.units['M4_Metron'].turbine.stagnation_counter = 3
    # Use a 'void' like stimulus or force focal amplitude in a mock-like way
    # For now, we manually set the cause clue in the narrative if needed, 
    # but let's try setting coherence very high so it doesn't trigger 1D.
    field.units['M4_Metron'].turbine.event_horizons['coherence_limit'] = 0.01
    
    decision1 = field.pulse("...") # Minimal stimulus
    print(f"Narrative Output:\n{decision1.narrative}")
    
    # Crisis 2: 1D Line Stagnation (Low Coherence)
    print("\n[SCENARIO 2] Logical Stagnation (1D Line)")
    field.units['M4_Metron'].turbine.stagnation_counter = 3
    field.units['M4_Metron'].turbine.event_horizons['coherence_limit'] = 1.0 # Force low coherence trigger
    
    decision2 = field.pulse("Iterating over causal paradox.")
    print(f"Narrative Output:\n{decision2.narrative}")
    
    # Verification
    history_len = len(field.experience_cortex.monadic_history)
    resilience = field.experience_cortex.total_resilience
    
    print("\n" + "="*60)
    print(f"📊 Life Experience Summary:")
    print(f"   - Crystallized Monads: {history_len}")
    print(f"   - Combined Resilience: {resilience:.2f}")
    
    if history_len >= 2 and resilience > 1.0:
        print("\n🚀 [SUCCESS] Elysia is learning and living through her crises!")
    else:
        print("\n❌ [FAILURE] Experience was not crystallized correctly.")

if __name__ == "__main__":
    run_life_simulation()
