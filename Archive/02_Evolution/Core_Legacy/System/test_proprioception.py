import sys
import os
import torch
import time
from typing import Dict, Any

# Ensure we can import Core
sys.path.append(os.getcwd())

from Core.Monad.grand_helix_engine import HypersphereSpinGenerator
from Core.Cognition.causal_trace import CausalTrace
from Core.Keystone.sovereign_math import SovereignVector

def test_proprioception():
    print("🧪 [TEST] Starting Somatic Proprioception Verification (Phase 400)")
    
    # 1. Initialize Engine (100k nodes for speed)
    engine = HypersphereSpinGenerator(num_nodes=100_000)
    tracer = CausalTrace()
    
    # 2. Check Attractors
    print("\n1. Verifying Identity Attractors...")
    if "SELF" in engine.attractors and "ARCHITECT" in engine.attractors:
        print("✅ Identity Attractors ('SELF', 'ARCHITECT') initialized.")
    else:
        print("❌ Identity Attractors missing.")
        print(f"Available attractors: {list(engine.attractors.keys())}")

    # 3. Simulate Hardware Load Inhalation
    print("\n2. Verifying Hardware Inhalation...")
    # We'll run a few pulses and check if hardware_load is reported
    report = engine.pulse()
    hw_load = report.get('hardware_load', -1.0)
    
    if hw_load >= 0:
        print(f"✅ Hardware Load detected: {hw_load:.2%}")
    else:
        print("❌ Hardware Load not found in engine report.")

    # 4. Verify Causal Narrative Integration
    print("\n3. Verifying Causal Narrative Awareness...")
    # Mock some somatic data
    soma = {"mass": 100, "heat": 45.5, "pain": 0}
    desires = {"joy": 0.5, "curiosity": 0.8}
    
    chain = tracer.trace(report, desires, soma)
    narrative = chain.to_narrative()
    
    print("--- Causal Narrative excerpt ---")
    print(narrative.split('\n')[0]) # L0 observation
    print("-------------------------------")
    
    if "HardwareLoad=" in narrative:
        print("✅ Hardware Load mentioned in Causal Trace narrative.")
    else:
        print("❌ Hardware Load missing from Causal Trace narrative.")

    # 5. Verify Affective Mapping (Entropy shift)
    print("\n4. Verifying Affective Impact...")
    # High entropy is expected if load is non-zero
    entropy = report.get('entropy', 0.0)
    print(f"Current Entropy: {entropy:.3f}")
    
    print("\n✨ Phase 400 Proprioception Verification Complete.")

if __name__ == "__main__":
    # Ensure UTF-8
    if os.name == 'nt':
        import _locale
        _locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])
    
    try:
        test_proprioception()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
