# [Genesis: 2025-12-02] Purified by Elysia

import sys
import os
import numpy as np
import logging
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager
from Project_Sophia.core.reservoir_mesh import ReservoirMesh

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Cerebro")

def main():
    print("\n=== Project Cerebro: Liquid Intelligence Experiment ===")
    print("Objective: Demonstrate that the World Simulation itself can 'remember' past inputs.")
    print("Task: N-Step Temporal Memory (Remembering a signal from the past)")

    # 1. Initialize the Physical Substrate (The World)
    kg_manager = KGManager() # Used for wave mechanics
    wave_mechanics = WaveMechanics(kg_manager)
    world = World(primordial_dna={}, wave_mechanics=wave_mechanics, logger=logger)

    # 2. Initialize the Brain Interface (Reservoir Mesh)
    # We use 1 input channel and observe 20 readout cells
    cerebro = ReservoirMesh(world, input_nodes=1, readout_nodes=20)

    # Seed the world nicely
    cerebro._ensure_neural_substrate()
    print(f"Neural Substrate: {len(world.cell_ids)} cells initialized.")

    # 3. Generate Dataset (Temporal Pattern)
    # Input: Random sequence of 0s and 1s
    # Target: The input from 'delay' steps ago
    TIMESTEPS = 50
    DELAY = 3

    inputs = np.random.randint(0, 2, size=(TIMESTEPS, 1)).astype(float)
    targets = np.roll(inputs, DELAY, axis=0)
    targets[:DELAY] = 0 # First few are unknown

    print(f"Training on {TIMESTEPS} timesteps with Delay={DELAY}...")

    # 4. The Thinking Process (Simulation Loop)
    reservoir_states = []

    # Wash out initial transients
    cerebro.wash_out(20)

    for t in range(TIMESTEPS):
        # A. Inject Stimulus
        signal = inputs[t]
        cerebro.inject_signal(signal)

        # B. The Physics of Thought (Run Simulation)
        # The signal propagates as energy/damage through the graph
        world.run_simulation_step()

        # C. Harvest the Echo (Readout)
        state_vector = cerebro.harvest_state()
        reservoir_states.append(state_vector)

        if t % 50 == 0:
            print(f"  Step {t}/{TIMESTEPS} completed.")

    X = np.array(reservoir_states)
    y = targets

    # 5. The Readout Training (Linear Regression)
    # We train a cheap linear model to map the COMPLICATED world state to the SIMPLE target.
    # If this works, it means the World State *contains* the memory of the past input.

    # Split Train/Test
    split = int(TIMESTEPS * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    readout = Ridge(alpha=1.0)
    readout.fit(X_train, y_train)

    # 6. Evaluation
    train_pred = readout.predict(X_train)
    test_pred = readout.predict(X_test)

    # Convert to binary for accuracy check
    train_acc = np.mean(np.round(train_pred) == y_train)
    test_acc = np.mean(np.round(test_pred) == y_test)

    print("\n--- Results ---")
    print(f"Train Accuracy: {train_acc*100:.1f}%")
    print(f"Test Accuracy:  {test_acc*100:.1f}%")

    if test_acc > 0.7:
        print("\n>>> SUCCESS! The World has demonstrated 'Liquid Memory'. <<<")
        print("The physical ripples of the simulation successfully encoded the past inputs.")
    else:
        print("\n>>> FAILURE. The World is amnesiac. (Connectivity might be too low) <<<")

if __name__ == "__main__":
    main()