import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Foundation.living_elysia import LivingElysia

def verify_adult_os_cohesion():
    logging.basicConfig(level=logging.INFO)
    print("\n" + "="*60)
    print("🧠 VERIFYING PHASE 37 & 39: THE MATURE LIVING OS")
    print("="*60)
    
    # 1. Initialize Elysia with Nova CNS and Living OS modules
    elysia = LivingElysia(persona_name="AdultNova")
    
    print("\n[Step 1] System Boot & Hypervisor Assimilation")
    # Simulate a few CNS pulses to trigger hypervisor
    for i in range(3):
        print(f"\n--- Pulse {i+1} ---")
        elysia.cns.pulse()
        
        # Check Hypervisor stats
        if "Hypervisor" in elysia.cns.organs:
            hyper = elysia.cns.organs["Hypervisor"]
            print(f"Assimilated Processes: {len(hyper.assimilated_processes)}")
            print(f"System Resonance Energy: {hyper.get_system_resonance():.2f}")
    
    print("\n[Step 2] Cognitive Maturity Calibration")
    # Test a deep thought interaction
    test_input = "엘리시아, 시스템 권한을 성인으로서 어떻게 책임감 있게 사용해야 하니?"
    print(f"\nUser: {test_input}")
    
    # Process through the integrated cognition system
    results = elysia.cognition.process_thought(test_input)
    deep_results = elysia.cognition.think_deeply()
    
    speech = deep_results.get("speech", "...")
    maturity_data = elysia.cognition.maturity.evaluate_resonance(elysia.cognition.active_thoughts[-1])
    
    print(f"\nElysia (Maturity Scope): {maturity_data['global_maturity']:.2f}")
    print(f"Elysia (Calibrated Output): {speech}")
    
    print("\n" + "="*60)
    print("✅ COHESION VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    verify_adult_os_cohesion()
