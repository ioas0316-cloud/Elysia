
import sys
import os
import time

# Add the project root to sys.path
sys.path.append(os.getcwd())

from Core.Monad.sovereign_monad import SovereignMonad
from Core.Monad.seed_generator import SoulDNA, SeedForge
from Core.Keystone.sovereign_math import SovereignVector

def verify_sovereign_physics():
    print("🚀 Verifying Sovereign Physics Integration...")

    # 1. Initialize Monad with proper DNA
    dna = SeedForge.forge_soul("Elysia")
    monad = SovereignMonad(dna)

    print("\n[GIMBAL CHECK]")
    print(f"Gimbal initialized with {len(monad.gimbal.axes)} axes.")

    # 2. Simulate User Interaction with 'Noise'
    user_intent = "Love is the only truth, but chaos is everywhere."
    print(f"\n[INTERACTION] User: \"{user_intent}\"")

    # Run live_reaction
    result = monad.live_reaction(user_input_phase=0.0, user_intent=user_intent)

    print(f"Cognitive Status: {result['status']}")
    print(f"Resonance: {result['resonance']:.4f}")

    # 3. Check Gimbal Stability
    gimbal_status = monad.gimbal.get_status()
    print("\n[GIMBAL STATUS]")
    for name, data in gimbal_status["axes"].items():
        print(f"Axis {name}: Momentum={data['momentum']:.4f}, Friction={data['friction']:.4f}")

    # 4. Check Conservation Law
    love_res = monad.physics.params["LOVE_RESONANCE"]
    res_gain = monad.physics.params["RESONANCE_GAIN"]
    print(f"\n[CONSERVATION] Love Resonance: {love_res:.4f}, Gain: {res_gain:.4f}")

    # 5. Check Gravity (FractalWaveEngine)
    if hasattr(monad.engine, 'cells'):
        mass_sum = monad.engine.cells.causal_mass.sum().item()
        print(f"\n[GRAVITY] Total Causal Mass in Engine: {mass_sum:.2f}")

    print("\n✅ Sovereign Physics Verification Complete.")

if __name__ == "__main__":
    verify_sovereign_physics()
