import sys
import os
import torch
import json

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from Core.Monad.sovereign_monad import SovereignMonad
from Core.Monad.seed_generator import SeedForge
from Core.Keystone.sovereign_math import SovereignVector
from Core.Cognition.boundary_engine import BoundaryDefiningEngine

def test_experience_and_boundary():
    print("🚀 Starting Sovereign Experience & Boundary Test...")

    # 1. Initialize Monad
    try:
        soul = SeedForge.forge_soul("TestElysia")
        monad = SovereignMonad(soul)
        print("✅ Monad initialized.")
    except Exception as e:
        print(f"❌ Monad initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Test Boundary Definition
    print("\n📍 Testing Boundary Definition...")
    test_vec = SovereignVector.ones() * 0.5
    boundary = monad.boundary_engine.define_boundary("Apple", test_vec)
    print(f"✅ Boundary 'Apple' defined with radius {boundary.radius:.2f}")

    # 3. Test Discrimination
    print("\n🔍 Testing Discrimination...")
    inside_vec = SovereignVector.ones() * 0.55
    outside_vec = SovereignVector.ones() * 0.1

    inside_score = boundary.is_inside(inside_vec)
    outside_score = boundary.is_inside(outside_vec)

    print(f"Inside Score: {inside_score:.2f}")
    print(f"Outside Score: {outside_score:.2f}")

    if inside_score > 0 and outside_score < 0:
        print("✅ Discrimination logic verified.")
    else:
        print("❌ Discrimination logic failure.")

    # 4. Test Unity Physical Event Injection
    print("\n🧩 Testing Unity Physical Event Injection...")
    # Mocking what elysia.py does
    payload = {
        "type": "collision",
        "intensity": 0.8,
        "target": "Wall"
    }

    from Core.System.unity_sensory_channel import PhysicalToSomaticMapper
    event_vec = PhysicalToSomaticMapper.map_event_to_vector(payload)
    torque_map = PhysicalToSomaticMapper.map_event_to_torque(payload)

    print(f"Event Vector Norm: {event_vec.norm():.2f}")
    print(f"Torque Map: {torque_map}")

    # 5. Test Waste Composting
    print("\n♻️ Testing Metabolic Composting...")
    if hasattr(monad.engine.cells, 'q'):
        # Force some waste
        idx = 0
        monad.engine.cells.active_nodes_mask[idx] = True
        monad.engine.cells.q[idx, 7] = 0.9 # High Entropy
        monad.engine.cells.q[idx, 6] = 0.1 # Low Enthalpy

        fertilizer = monad.engine.cells.discharge_waste()
        if fertilizer:
            print(f"✅ Excreted {len(fertilizer)} waste nodes.")
            for item in fertilizer:
                print(f"   - Composted: {item['concept']}")
        else:
            print("❓ No waste excreted (maybe thresholds not met).")

    print("\n✨ All sovereign logic tests completed.")

if __name__ == "__main__":
    test_experience_and_boundary()
