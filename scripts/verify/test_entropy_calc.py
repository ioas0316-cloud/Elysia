import numpy as np
from synaptic_architecture.field import CrystallizationField

def test_entropy():
    field = CrystallizationField(64)

    # 1. Initial State (No activation)
    e0 = field.calculate_entropy()
    print(f"Initial Entropy (No activation): {e0:.4f}")

    # 2. Random Activation (High Entropy)
    field.activation = np.random.rand(64, 64).astype(np.float32)
    e1 = field.calculate_entropy()
    print(f"Random Activation Entropy: {e1:.4f}")

    # 3. Focused Activation (Low Entropy)
    field.activation = np.zeros((64, 64), dtype=np.float32)
    field.activation[32, 32] = 100.0
    e2 = field.calculate_entropy()
    print(f"Focused Activation Entropy: {e2:.4f}")

    # 4. Focused Activation + High Conductance (Even Lower)
    field.conductance[32, 32] = 10.0
    e3 = field.calculate_entropy()
    print(f"Focused + High Conductance Entropy: {e3:.4f}")

    assert e1 > e2 > e3
    print("Entropy calculation verified.")

if __name__ == "__main__":
    test_entropy()
