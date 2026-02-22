
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine import GrandHelixEngine

def test_solidification():
    print("🧬 [TEST] Initiating Merkaba Solidification Test...")
    
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = GrandHelixEngine(num_cells=10_000_000, device=device)
    
    # 2. Mutate a specific spot in the DNA (The 'Past')
    # We set a unique signature at index [50, 50]
    print("🖋️ [TEST] Inscribing a unique signature into the 10M manifold...")
    signature_val = 0.888 # Use float for 4D vector component
    engine.cells.permanent_q[50, 50, 1] = signature_val
    
    # 3. Solidify
    print("🕯️ [TEST] Calling Solidify (Crystallization)...")
    engine.solidify()
    
    # Check if files exist
    solid_path = engine.solid_path
    required_files = ["permanent_q.pt", "active_q.pt", "momentum.pt"]
    for f in required_files:
        f_path = os.path.join(solid_path, f)
        if os.path.exists(f_path):
            print(f"✅ Found solid file: {f}")
        else:
            print(f"❌ Missing solid file: {f}")
            return
            
    # 4. Kill and Resurrect
    print("💀 [TEST] Closing engine and instantiating a new one (Resurrection)...")
    del engine
    
    new_engine = GrandHelixEngine(num_cells=10_000_000, device=device)
    
    # 5. Verify Signature
    recovered_val = new_engine.cells.permanent_q[50, 50, 1].item()
    print(f"🔎 [TEST] Recovered Signature: {recovered_val}")
    
    if abs(recovered_val - signature_val) < 1e-5:
        print("🏆 [TEST] SUCCESS: The Merkaba has been solidified and resurrected!")
    else:
        print("❌ [TEST] FAILURE: The signature was lost in transit.")

if __name__ == "__main__":
    test_solidification()
