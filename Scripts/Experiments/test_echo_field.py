"""
Test Script: Echo Field Verification
=============================================
Scripts.Experiments.test_echo_field

Verifies the Hybrid Genesis:
1. Endfield Structure (World/State)
2. Myeongjo Soul (Echoes/Action/Resonance)
"""

import sys
import os
import random

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.World.Simulations.echo_field_world import EchoFieldWorld, SovereignOperator, CorruptionEntity, Vector3
from Core.L7_Spirit.Monad.Laws.echo_field_physics import EchoFieldPhysicsMonad

def run_simulation():
    print("⚡ Initializing Project Echo Field...")

    # 1. Genesis
    world = EchoFieldWorld(seed="HYBRID_PROTO_01")
    physics = EchoFieldPhysicsMonad(seed="ACTION_LAW")

    # 2. Spawn Entities
    operator = SovereignOperator(uid="OP_VANGUARD", role="Vanguard", position=Vector3(0,0,0))
    world.spawn_entity(operator)

    enemy = CorruptionEntity(uid="NOISE_ERROR_404", position=Vector3(5,0,0), power=10.0)
    world.spawn_entity(enemy)

    print("\n--- PHASE 1: RESONANCE CHECK (PARRY) ---")
    # Simulate an attack at t=1.0
    attack_time = 1.0

    # Attempt 1: Too late (t=1.5, delta=0.5 > window 0.2)
    current_time = 1.5
    success = operator.parry(attack_time, current_time, world.state.resonance_window)
    print(f"Parry Attempt @ +0.5s (Window {world.state.resonance_window}s): {'CRITICAL' if success else 'MISS'}")

    # HACK: Widen the window using Monad
    print(">> HACKING: Widening Resonance Window to 1.0s...")
    physics.hack_reality("resonance_window", 1.0)
    physics.enforce(world)

    # Attempt 2: Same timing, but hacked window
    success = operator.parry(attack_time, current_time, world.state.resonance_window)
    print(f"Parry Attempt @ +0.5s (Window {world.state.resonance_window}s): {'CRITICAL' if success else 'MISS'}")

    print("\n--- PHASE 2: ECHO HUNT (LOOT) ---")
    # HACK: Increase Drop Rate to 100%
    print(">> HACKING: Maxing Echo Drop Rate...")
    physics.hack_reality("echo_drop_rate", 1.0)
    physics.enforce(world)

    # Defeat Enemy
    echo = enemy.defeat(world.state)
    if echo:
        print(f"✨ ENEMY DEFEATED! DROPPED: {echo.uid} ({echo.effect})")
        world.spawn_entity(echo)

        # Absorb
        power = echo.absorb()
        print(f"💪 ECHO ABSORBED! GAINED: {power}")
    else:
        print("❌ No Drop.")

    print("\n✅ Echo Field Test Complete.")

if __name__ == "__main__":
    run_simulation()
