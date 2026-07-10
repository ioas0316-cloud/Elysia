import numpy as np
import time
from core.intelligence.thought_element import ThoughtTransistor
from core.intelligence.thought_field import ThoughtField

def run_survival_evolution():
    print("=== [Elysia] Organic Survival & Evolution Demo ===")
    field = ThoughtField()

    # 1. Initialize two competing Seeds
    seed_a = ThoughtTransistor("Seed_Alpha", np.array([1, 0, 0], dtype=np.float32))
    seed_b = ThoughtTransistor("Seed_Beta", np.array([0, 0, 1], dtype=np.float32))
    field.add_element(seed_a)
    field.add_element(seed_b)

    print("Stimulating Seed_Alpha more than Seed_Beta to see selection and pruning...")

    for i in range(50):
        # Asymmetric resource injection
        field.pulse({"Seed_Alpha": 4.0, "Seed_Beta": 0.5})

        results = field.step()

        if i % 10 == 0:
            print(f"\n[Step {i}] Population: {len(field.elements)} cells")
            # Calculate health average for Alpha vs Beta lineages
            alpha_lineage = [e for eid, e in field.elements.items() if "Alpha" in eid]
            beta_lineage = [e for eid, e in field.elements.items() if "Beta" in eid]

            avg_h_a = np.mean([e.health for e in alpha_lineage]) if alpha_lineage else 0
            avg_h_b = np.mean([e.health for e in beta_lineage]) if beta_lineage else 0

            print(f"  Alpha Lineage: {len(alpha_lineage)} cells, Avg Health: {avg_h_a:.2f}")
            print(f"  Beta Lineage: {len(beta_lineage)} cells, Avg Health: {avg_h_b:.2f}")

            if hasattr(field, 'organs') and field.organs:
                print(f"  Stable Organs: {len(field.organs)}")

        # Feedback flow
        if results:
            field.pulse({eid: energy * 0.3 for eid, energy in results.items()})

        time.sleep(0.05)

    print("\n=== Final Selection Results ===")
    print(f"Living Cells: {list(field.elements.keys())}")
    for eid, element in field.elements.items():
        print(f"[{eid}] Health: {element.health:.2f}, G: {element.conductance:.2f}")

if __name__ == "__main__":
    run_survival_evolution()
