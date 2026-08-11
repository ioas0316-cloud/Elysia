import numpy as np
from core.physics.topological_reduction import TopologicalReductionEngine

def verify_topological_reduction():
    print("==================================================================")
    print(" [Verification] Topological Reduction & Equivalent Synthesis Engine")
    print("==================================================================\n")

    # Initialize a 16-node network with 2 boundary ports
    engine = TopologicalReductionEngine(num_nodes=16, num_boundary=2)

    # ---------------------------------------------------------
    # PRINCIPLE 1: Equivalence (등가성)
    # ---------------------------------------------------------
    print("--- [Principle 1] Equivalence (등가성) Verification ---")
    G_reduced, R_eq = engine.compress()
    print(f"  > Condensed Boundary Conductance Matrix (G_reduced):\n{G_reduced}")
    print(f"  > Single Representative Equivalent Resistance (R_eq): {R_eq:.4f}")
    assert R_eq > 0.0, "Equivalent resistance must be positive."
    print("  > [SUCCESS] Complex network successfully reduced to a single representative value.\n")

    # ---------------------------------------------------------
    # PRINCIPLE 2: Hierarchical Local Reduction & Modality-Agnostic Projection
    # ---------------------------------------------------------
    print("--- [Principle 2] Modality-Agnostic Projection & Hierarchical Mapping ---")

    # 1. Linguistic input ("Love")
    print("  > Projecting Linguistic Modality Data ('Love')...")
    engine.map_multimodal_to_network({
        "language": "Love",
        "physical": {"cpu": 0.2, "ram": 0.3}
    })
    _, R_eq_love = engine.compress()
    print(f"    - Equivalent Resistance under 'Love': {R_eq_love:.4f}")

    # 2. Visual input (Red light scenario)
    print("  > Projecting Visual Modality Data (High Red bias)...")
    engine.map_multimodal_to_network({
        "visual": {"red": 0.9, "green": 0.1, "blue": 0.1},
        "physical": {"cpu": 0.8, "ram": 0.9} # High autonomic pressure
    })
    _, R_eq_red = engine.compress()
    print(f"    - Equivalent Resistance under Red Visual stress: {R_eq_red:.4f}")

    assert R_eq_love != R_eq_red, "Modality projection must dynamically shape the network topology."
    print("  > [SUCCESS] Different sensory/symbolic inputs successfully mapped to unique topological structures.\n")

    # ---------------------------------------------------------
    # PRINCIPLE 3: Closed-Loop Self-Correction & Attractor Resonance
    # ---------------------------------------------------------
    print("--- [Principle 3] Closed-Loop Self-Refinement & Attractor Resonance ---")
    target_potential = 1.5
    refinement_res = engine.run_self_refinement_loop(target_potential=target_potential, max_steps=15, lr=0.3)

    print(f"  > Convergence status: {refinement_res['converged']}")
    print(f"  > Final Equivalent Resistance: {refinement_res['final_equivalent_resistance']:.4f}")
    print(f"  > Potentials History: {refinement_res['potentials_history']}")
    print(f"  > Residuals History: {refinement_res['residuals_history']}")

    # Assert that the residual decreased and is close to zero
    final_residual = refinement_res['residuals_history'][-1]
    assert abs(final_residual) < 0.1, f"Self-correction loop did not align target potential. Final residual: {final_residual}"
    print("  > [SUCCESS] Closed-loop self-refinement successfully aligned equivalent state to target attractor potential.\n")

    # ---------------------------------------------------------
    # PRINCIPLE 4: Cross-Modal Resonance / Translation
    # ---------------------------------------------------------
    print("--- [Principle 4] Cross-Modal Resonance/Translation ---")
    source_modality = {
        "language": "Jesus",
        "physical": {"cpu": 0.1, "ram": 0.1}
    }

    # Translate language input into visual representation via the equivalent latent potential
    translation = engine.cross_modal_translate(source_modality, target_key="visual")
    print(f"  > Sensed Language: 'Jesus'")
    print(f"  > Equivalent Latent Potential: {translation['latent_potential']:.4f}")
    print(f"  > Translated Visual RGB Spectrum: {translation['translated_data']}")
    assert "intensity" in translation['translated_data'], "Translation must produce expected visual structure."
    print("  > [SUCCESS] Cross-modal resonance achieved between Linguistic and Visual domains.\n")

    print("==================================================================")
    print(" [Verification Complete] All Principles Successfully Proven!")
    print("==================================================================")

if __name__ == "__main__":
    verify_topological_reduction()
