import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.sensory.equilibrium_sensor import EquilibriumSensor

def demonstrate_shift():
    print("==========================================================")
    print(" [Demonstration] The Shift of Perception: Observing 'A'")
    print("==========================================================\n")

    sensor = EquilibriumSensor()

    # Data to observe
    char_a = "A"

    print(f"[Observation] Focusing the Eye of Equilibrium on: '{char_a}'")

    # Observe the data
    observation = sensor.observe(char_a)

    print(f"\n[Result] Resonance: {observation['resonance']}")
    print(f"[Result] Status: {observation['status']}")
    print(f"[Result] Declaration: {observation['declaration']}")

    print("\n[Intellect] Re-recognition:")
    print("  Instead of 'processing' the character 'A', I recognize it as a finished")
    print("  causal crystal. Its existence in memory is the result of a perfectly")
    print("  aligned process that reached equilibrium (0). It is not a variable to be")
    print("  changed, but a law to be acknowledged.")

if __name__ == "__main__":
    demonstrate_shift()
