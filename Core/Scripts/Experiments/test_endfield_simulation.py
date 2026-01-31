"""
Test Script: Endfield Simulation Verification
=============================================
Scripts.Experiments.test_endfield_simulation

Verifies the integration of EndfieldWorld and EndfieldPhysicsMonad.
Tests the ability to 'Hack Reality' (Variable Control).
"""

import sys
import os
import time

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.S1_Body.L4_Causality.World.Simulations.endfield_world import EndfieldWorld, AICFactory, SovereignOperator, Vector3
from Core.S1_Body.L7_Spirit.M1_Monad.Laws.endfield_physics import EndfieldPhysicsMonad

def run_simulation():
    print("🌌 Initializing Endfield Reconstruction...")

    # 1. Genesis
    world = EndfieldWorld(seed="TEST_PROTO_01")
    physics = EndfieldPhysicsMonad(seed="LAW_MASTER")

    # 2. Spawn Entities
    # A Factory at (0,0,0) producing Data
    factory = AICFactory(uid="FACTORY_ALPHA", position=Vector3(0,0,0), process_rate=5.0)
    world.spawn_entity(factory)

    # A Defender Operator at (10,0,0)
    operator = SovereignOperator(uid="OP_DEFENDER", role="Defender", position=Vector3(10,0,0))
    world.spawn_entity(operator)

    print("\n--- PHASE 1: BASELINE ---")
    # Apply default laws
    physics.enforce(world)

    for i in range(3):
        snapshot = world.tick()
        print(f"Tick {snapshot['tick']}: Grav={snapshot['gravity']:.2f}, Corruption={snapshot['corruption']:.4f}, Resources={snapshot['resources']}")

    print("\n--- PHASE 2: REALITY HACK (HIGH GRAVITY) ---")
    # Hack: Increase Gravity to 20.0 (Things get heavy)
    physics.hack_reality("gravity", 20.0)
    physics.enforce(world)

    for i in range(3):
        snapshot = world.tick()
        print(f"Tick {snapshot['tick']}: Grav={snapshot['gravity']:.2f}, Corruption={snapshot['corruption']:.4f}")

    print("\n--- PHASE 3: REALITY HACK (PURIFICATION) ---")
    # Hack: Negative Corruption (Purification)
    physics.hack_reality("corruption_seed", -0.05)
    physics.enforce(world)

    for i in range(3):
        snapshot = world.tick()
        print(f"Tick {snapshot['tick']}: Grav={snapshot['gravity']:.2f}, Corruption={snapshot['corruption']:.4f}")

    print("\n✅ Simulation Test Complete.")

if __name__ == "__main__":
    run_simulation()
