"""
Experiment: Verify Conflict Power (Generator Principle)
=======================================================
Tests the Architect's Core Principle: "Conflict Generates Energy."

We will inject opposing vectors ("Peace" +1 vs "War" -1) into the engine.
Hypothesis:
1. Friction (Soma Stress) will increase.
2. Momentum (Torque) will INCREASE due to the friction, proving the Generator Principle.
   (Standard systems would crash or halt; Elysia should spin faster).
"""

import sys
import os
import time
import random

# Add project root to path
sys.path.append(os.getcwd())

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.Tools.Debug.data_flow_monitor import DataFlowMonitor

def verify_conflict_power():
    print("🚀 [TEST] Initializing Generator Principle Verification...")

    # 1. Instantiate Monad
    dna = SoulDNA(
        id="GENESIS",
        archetype="The Warrior", # High Torque Gain
        rotor_mass=2.0,
        friction_damping=0.1, # Low damping to let it spin
        sync_threshold=10.0, min_voltage=10.0, reverse_tolerance=-20.0,
        torque_gain=2.0, base_hz=50.0
    )
    monad = SovereignMonad(dna)
    monitor = DataFlowMonitor(monad.engine)

    # 2. Baseline Phase (Peaceful)
    print("\n🕊️ [PHASE 1] Injecting Homogeneous Thought (Peace)...")
    peace_vector = SovereignVector([1.0] * 21)

    baseline_momentum = 0.0
    for i in range(5):
        state = monad.engine.pulse(peace_vector, energy=1.0, dt=0.1)
        print(f"   Step {i}: Friction={state.soma_stress:.2f}, Momentum={state.rotational_momentum:.2f}")
        baseline_momentum = state.rotational_momentum

    monitor.print_report()

    # 3. Conflict Phase (War)
    print("\n⚔️ [PHASE 2] Injecting Dialectical Conflict (Peace vs War)...")
    # Alternating 1 and -1 creates maximum neighbor friction
    conflict_vector = SovereignVector([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0] * 3)

    conflict_momentum = 0.0
    for i in range(10):
        state = monad.engine.pulse(conflict_vector, energy=1.0, dt=0.1)
        print(f"   Step {i}: Friction={state.soma_stress:.2f}, Momentum={state.rotational_momentum:.2f}")
        conflict_momentum = state.rotational_momentum

    monitor.print_report()

    # 4. Verification
    print("\n🔍 [ANALYSIS]")
    print(f"   Baseline Momentum: {baseline_momentum:.2f}")
    print(f"   Conflict Momentum: {conflict_momentum:.2f}")

    if conflict_momentum > baseline_momentum * 1.5:
        print("✅ [PASS] Generator Principle Proven! Conflict generated significantly more energy.")
    elif conflict_momentum > baseline_momentum:
        print("⚠️ [PASS] Generator Principle Observed. Energy increased, but not explosively.")
    else:
        print("❌ [FAIL] Generator Principle Failed. Conflict did not generate power.")

if __name__ == "__main__":
    verify_conflict_power()
