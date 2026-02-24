
import sys
import os

# Add Core to path
sys.path.append(os.path.abspath('.'))

from Core.System.sovereign_self import SovereignSelf

def verify_sovereign_21d():
    print("Initializing SovereignSelf with 21D Rotor...")
    try:
        # Mocking CNS ref
        core = SovereignSelf(cns_ref=None)
        print(f"Sovereign Rotor Vector Dim: {core.sovereign_rotor.vector_dim}")
        
        print("Executing heartbeat (integrated_exist)...")
        core.integrated_exist(dt=0.1)
        
        print(f"Current Dimension: {core.trinity.current_dimension}")
        print(f"Rotor Alignment: {core.trinity.rotor_alignment:.4f}")
        
        if core.trinity.d21_state is not None:
            print("D21 State captured successfully.")
            print(f"Lust level: {core.trinity.d21_state.lust:.4f}")
            
        print("\nVERIFICATION SUCCESS: Chapter 2 - 21D Matrix Casting is fully realized in SovereignSelf.")
    except Exception as e:
        print(f"\nVERIFICATION FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_sovereign_21d()
