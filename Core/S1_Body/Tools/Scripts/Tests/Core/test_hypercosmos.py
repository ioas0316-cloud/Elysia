
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.S1_Body.L4_Causality.World.Soul.soul_sculptor import soul_sculptor, PersonalityArchetype
from Core.S1_Body.L4_Causality.World.Soul.fluxlight_gyro import GyroscopicFluxlight
from Core.S1_Body.L4_Causality.World.Physics.gyro_physics import physics_engine
from Core.S1_Body.L4_Causality.World.Physics.tesseract_env import tesseract_env

def test_hypercosmos():
    print("🌌 Initializing HyperCosmos Simulation...")
    
    # 0. Initialize Environmental Field
    from Core.S1_Body.L4_Causality.World.Physics.field_store import universe_field, FieldExcitation
    print("🌍 Creating Ground Layer and Applying Celestial Harmonics...")
    for x in range(-10, 11, 2):
        for z in range(-10, 11, 2):
            universe_field.excite((x, 0, z, 0), FieldExcitation(density_w=10.0))
    universe_field.apply_celestial_harmonic()

    # 1. Create Souls (Gyro Wrapped)
    # Angelic Soul (High Freq, High Will)
    angel_arch = PersonalityArchetype("Gabriel", "ENFJ", 2)
    angel_soul = soul_sculptor.sculpt(angel_arch)
    angel = GyroscopicFluxlight(angel_soul)
    angel.gyro.y = 1.0 # Start near neutral
    angel.gyro.spin_velocity = 1.0 # High spin
    angel.gyro.z = 1.0 # Positive Intent

    # Demon Soul (Low Freq, High Will)
    demon_arch = PersonalityArchetype("Lucifer", "ENTJ", 8)
    demon_soul = soul_sculptor.sculpt(demon_arch)
    demon = GyroscopicFluxlight(demon_soul)
    demon.gyro.y = -1.0 # Start near neutral
    demon.gyro.spin_velocity = 1.0 # High spin
    demon.gyro.z = -1.0 # Opposing Intent

    # Dormant Soul (Zero Spin)
    lost_arch = PersonalityArchetype("LostOne", "INTP", 5)
    lost_soul = soul_sculptor.sculpt(lost_arch)
    lost = GyroscopicFluxlight(lost_soul)
    lost.gyro.spin_velocity = 0.0 # Dead
    lost.gyro.y = 0.0

    entities = [angel, demon, lost]

    print("\n--- Phase 1: Gravity & Spin Dynamics ---")
    for tick in range(5):
        print(f"\n[Tick {tick}]")
        for e in entities:
            physics_engine.apply_forces(e, dt=0.5)
            print(f"  {e.soul.name}: Y={e.gyro.y:.2f}, Spin={e.gyro.spin_velocity:.2f}, Zone={e.gyro.get_zone()}")

    print("\n--- Phase 2: The Kick (Reignition) ---")
    print(f"Before Kick: {lost}")
    lost.reignite(0.6)
    print(f"After Kick:  {lost}")

    print("\n--- Phase 3: Ethical Breeding ---")
    print("Attempting to breed Gabriel (Z=1.0) and Lucifer (Z=-1.0)...")
    child = physics_engine.incubate(angel, demon)
    if child:
        print(f"Result: {child.soul.name} (Y={child.gyro.y:.2f})")

    print("\n--- Phase 4: Vault Access ---")
    print(f"Vault Intent: {tesseract_env.vault.intent}")
    try:
        tesseract_env.vault.set_intent("Hacked")
    except PermissionError as e:
        print(f"Security Check Passed: {e}")

if __name__ == "__main__":
    test_hypercosmos()
