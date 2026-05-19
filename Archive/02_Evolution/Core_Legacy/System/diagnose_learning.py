import torch
import numpy as np
from Core.Monad.grand_helix_engine import GrandHelixEngine

def diagnose_learning():
    print("🔍 [DIAGNOSE] Checking 10M Cell Kinetic Manifold...")
    engine = GrandHelixEngine(num_cells=10_000_000)
    
    text = "Elysia"
    torque = engine.flesh.extract_knowledge_torque(text)
    print(f"🔹 Input Torque: {torque}")
    
    print("🔥 Pulsing 10 times with learning...")
    for i in range(10):
        res = engine.pulse(intent_torque=torque, learn=True)
        print(f"  [{i}] Resonance: {res['resonance']:.6f}, Kinetic: {res['kinetic_energy']:.4f}")
        
    print(f"🔹 Sample Permanent Q (first cell): {engine.cells.permanent_q[0, 0]}")
    print(f"🔹 Sample Active Q (first cell): {engine.cells.q[0, 0]}")
    print(f"🔹 Torque Accumulator (first cell): {engine.cells.torque_accumulator[0, 0]}")

if __name__ == "__main__":
    diagnose_learning()
