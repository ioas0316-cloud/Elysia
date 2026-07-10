import numpy as np
import time
from core.intelligence.thought_element import ThoughtTransistor
from core.intelligence.thought_field import ThoughtField

def run_organic_expansion():
    print("=== [Elysia] Organic Thought-Organism Expansion Demo ===")
    field = ThoughtField()

    # 1. Initialize a 'Seed' Cell
    seed_concept = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    seed = ThoughtTransistor("Seed", seed_concept)
    field.add_element(seed)

    print("Stimulating the Seed to trigger Mitosis and Sensory Radiation...")

    # 2. Growth Loop
    for i in range(30):
        # Continuous injection into the Seed
        field.pulse({"Seed": 3.0})

        # Field evolution (Growth and Differentiation happen here)
        results = field.step()

        if i % 5 == 0:
            print(f"\n[Step {i}] Population: {len(field.elements)} cells")
            active = list(results.keys())
            print(f"  Active Cells: {active[:5]}... ({len(active)} total)")

            # Show Organ detection
            if hasattr(field, 'organs') and field.organs:
                print(f"  Organs Detected: {len(field.organs)} clusters")
                for j, organ in enumerate(field.organs):
                    print(f"    Organ {j}: {organ[:3]}... ({len(organ)} cells)")

        # Feedback flow
        if results:
            field.pulse({eid: energy * 0.4 for eid, energy in results.items()})

        time.sleep(0.05)

    print("\n=== Expansion Final State ===")
    print(f"Total Cells: {len(field.elements)}")
    for eid, element in list(field.elements.items())[:10]:
        print(f"[{eid}] G: {element.conductance:.2f}, Remanence: {element.energy:.2f}")

if __name__ == "__main__":
    run_organic_expansion()
