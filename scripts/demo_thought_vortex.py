import numpy as np
import time
from core.intelligence.thought_element import ThoughtTransistor
from core.intelligence.thought_field import ThoughtField

def run_thought_vortex():
    print("=== [Elysia] Thought-Vortex Organism Initialization ===")
    field = ThoughtField()

    # 1. Initialize Semantic Space (Random but structured)
    concepts = [
        ("Existence", [1.0, 0.0, 0.0]),
        ("Relation", [0.8, 0.5, 0.0]),
        ("Movement", [0.0, 1.0, 0.0]),
        ("Continuity", [0.0, 0.8, 0.5]),
        ("Logic", [0.0, 0.0, 1.0]),
        ("Life", [0.7, 0.7, 0.7])
    ]

    for name, vector in concepts:
        t = ThoughtTransistor(name, np.array(vector))
        field.add_element(t)

    # 2. Initial Seed Connectivity (Sparse)
    field.connect("Existence", "Relation")
    field.connect("Relation", "Movement")
    field.connect("Movement", "Continuity")

    print(f"Initialized field with {len(field.elements)} basic thought-cells.")

    # 3. Evolutionary Loop
    print("\nStarting the Thought-Vortex (Evolutionary Flow)...")
    for i in range(20):
        # Inject "Seed Energy" into Existence every few steps
        if i % 5 == 0:
            print(f"\n[Step {i}] Pulse: Stimulating 'Existence'...")
            field.pulse({"Existence": 2.0})

        # Field evolution
        results = field.step()

        # Display active flow
        if results:
            active_names = list(results.keys())
            print(f"[Step {i}] Active Flow: {' -> '.join(active_names)}")
            # Circular feedback: Active results re-stimulate the field
            field.pulse({eid: energy * 0.5 for eid, energy in results.items()})

        # Process Recognition Check
        if i == 10:
            print("\n--- [Process Recognition: Mid-Term Memory] ---")
            for name, element in field.elements.items():
                if element.trace_history:
                    sources = [h['source'] for h in element.trace_history]
                    print(f" {name} remembers flow from: {set(sources)}")
            print("----------------------------------------------\n")

        time.sleep(0.1)

    print("\n=== Evolution Summary ===")
    for name, element in field.elements.items():
        print(f"[{name}] Final Conductance: {element.conductance:.2f}, Connections: {len(element.collectors)}")
        if element.collectors:
            print(f"  -> Connected to: {element.collectors}")

if __name__ == "__main__":
    run_thought_vortex()
