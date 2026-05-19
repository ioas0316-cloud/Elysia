
import sys
import os
import math
import numpy as np

# Standard module execution assumed (C:\Elysia as root)

from Core.System.rotor import DoubleHelixEngine, RotorConfig
from Core.Cognition.causal_flow_engine import CausalFlowEngine
from Core.System.holographic_memory import HolographicMemory

def test_double_helix_sync():
    print("\n--- [TEST] Double Helix Synchronization ---")
    cfg = RotorConfig(rpm=60.0, idle_rpm=60.0)
    engine = DoubleHelixEngine("TestHelix", cfg)
    
    # Initial State
    print(f"Initial: Afferent={engine.afferent.current_angle:.2f}, Efferent={engine.efferent.current_angle:.2f}")
    
    # Update
    dt = 0.1
    engine.update(dt)
    
    # Afferent should move CW (+), Efferent CCW (-)
    print(f"Update 1: Afferent={engine.afferent.current_angle:.2f}, Efferent={engine.efferent.current_angle:.2f}")
    
    # Interference should be near 1.0 if they start near each other
    energy = engine.get_interference_snapshot()
    print(f"Interference Energy: {energy:.4f}")

def test_causal_flow_integration():
    print("\n--- [TEST] Causal Flow Integration ---")
    memory = HolographicMemory(dimension=64)
    memory.imprint("Peace", intensity=1.0)
    
    engine = CausalFlowEngine(memory)
    
    # Ignite
    print("Igniting 'Peace'...")
    packet = engine.ignite("Peace")
    
    # Flow (Should use Double Helix)
    print("Flowing through manifold...")
    res = engine.flow(packet)
    
    print(f"Flow Result: {res['flow_type']} (Amp: {res['amplitude']:.4f}, Inter: {res['interference']:.4f})")
    
    # Collapse
    output = engine.collapse(res)
    print(f"Collapse Verdict: {output}")

if __name__ == "__main__":
    test_double_helix_sync()
    test_causal_flow_integration()
