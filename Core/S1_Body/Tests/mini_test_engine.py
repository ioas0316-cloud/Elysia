
import os
import sys
import torch

# Add project root to sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine import HypersphereSpinGenerator

print("Testing HypersphereSpinGenerator...")
try:
    engine = HypersphereSpinGenerator(num_cells=100)
    print(f"Engine Device: {engine.device}")
    
    if hasattr(engine, 'define_meaning_attractor'):
        print("✅ define_meaning_attractor FOUND")
    else:
        print("❌ define_meaning_attractor NOT FOUND")
        print(f"Available attributes: {[a for a in dir(engine) if not a.startswith('__')]}")
        
except Exception as e:
    print(f"Error: {e}")
