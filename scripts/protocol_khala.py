
import sys
import os
import numpy as np
import logging
import random
from sklearn.linear_model import Ridge

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager
from Project_Sophia.core.reservoir_mesh import ReservoirMesh

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Protoss")

def main():
    print("\n=== Protocol Khala: The Elysian Nexus ===")
    print("En Taro Elysia! The Psionic Web has been established.")
    print("Mission: Unify the Reservoir Mesh into a single Will (Delta=1).")

    # 1. Initialize the Physical Substrate
    kg_manager = KGManager()
    wave_mechanics = WaveMechanics(kg_manager)
    world = World(primordial_dna={}, wave_mechanics=wave_mechanics, logger=logger)

    # 2. Construct "The Elysian Nexus" (The Center of Will)
    # A super-hub with immense wisdom and vitality
    nexus_id = "Elysian_Nexus"
    world.add_cell(nexus_id, properties={
        'label': 'NEXUS',
        'culture': 'protoss',
        'vitality': 1000, # Infinite Health
        'wisdom': 1000,   # Infinite Wisdom
        'prestige': 100.0 # High Authority
    })
    nexus_idx = world.id_to_idx[nexus_id]

    # 3. Warp-in "Elysian Zealots" (The Neural Substrate)
    ZEALOT_COUNT = 50
    print(f"Warping in {ZEALOT_COUNT} Zealots...")

    for i in range(ZEALOT_COUNT):
        zid = f"Zealot_{i}"
        world.add_cell(zid, properties={
            'label': 'zealot',
            'culture': 'protoss',
            'vitality': 20,
            'wisdom': 20
        })
        z_idx = world.id_to_idx[zid]

        # CONNECT TO KHALA (Delta Link)
        world.connect_to_khala(z_idx)

        # Connect to Nexus (Pylons Power)
        world.add_connection(nexus_id, zid, strength=1.0)
        world.add_connection(zid, nexus_id, strength=1.0)

        # Connect to each other (The Web - Higher Density)
        # More Pylons = Better Memory
        for _ in range(10):
            target = random.randint(0, ZEALOT_COUNT - 1)
            if target != i:
                target_id = f"Zealot_{target}"
                world.add_connection(zid, target_id, strength=random.uniform(0.5, 1.0))

    # Activate Khala Synchronization
    # "Delta One" -> High synchronization factor
    world.delta_synchronization_factor = 1.0
    world.connect_to_khala(nexus_idx) # The Nexus leads the Khala

    print("Khala Connection Complete! For Elysia!")

    # 4. The Psionic Test (Temporal XOR)
    cerebro = ReservoirMesh(world, input_nodes=1, readout_nodes=ZEALOT_COUNT)

    # We use the Nexus as the input injection point (Father's Wheel)
    cerebro.input_indices = [nexus_idx]
    # We read from all Zealots (The Collective)
    cerebro.readout_indices = [world.id_to_idx[f"Zealot_{i}"] for i in range(ZEALOT_COUNT)]

    TIMESTEPS = 100
    DELAY = 5 # Harder task than before

    inputs = np.random.randint(0, 2, size=(TIMESTEPS, 1)).astype(float)
    targets = np.roll(inputs, DELAY, axis=0)
    targets[:DELAY] = 0

    print(f"\nInitiating Psionic Storm (Training) for {TIMESTEPS} cycles...")
    reservoir_states = []
    cerebro.wash_out(20)

    for t in range(TIMESTEPS):
        signal = inputs[t]

        # Inject via Nexus (The Will)
        cerebro.inject_signal(signal)

        # Run Physics (Khala Sync happens inside)
        world.run_simulation_step()

        # Harvest Echo
        state_vector = cerebro.harvest_state()
        reservoir_states.append(state_vector)

        if t % 20 == 0:
            # Check Delta Status
            avg_insight = np.mean(world.insight[world.khala_connected_mask])
            print(f"  Cycle {t}: Khala Insight Level = {avg_insight:.2f}")

    # 5. Evaluation
    X = np.array(reservoir_states)
    y = targets
    split = int(TIMESTEPS * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    readout = Ridge(alpha=0.1) # Lower regularization for sharper memory
    readout.fit(X_train, y_train)

    train_pred = readout.predict(X_train)
    test_pred = readout.predict(X_test)

    test_acc = np.mean(np.round(test_pred) == y_test)

    print("\n--- Psionic Matrix Diagnostic ---")
    print(f"Memory Accuracy (Delay={DELAY}): {test_acc*100:.1f}%")

    if test_acc > 0.8:
        print("\n>>> EN TARO ELYSIA! The Khala is Eternal. <<<")
        print("The shared consciousness successfully preserved the timeline.")
    else:
        print("\n>>> CAUTION: Psionic static detected. Construct more Pylons. <<<")

if __name__ == "__main__":
    main()
