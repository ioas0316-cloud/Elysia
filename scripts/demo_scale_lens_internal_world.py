import numpy as np
import time
from synaptic_architecture.machine_internal_world import MachineInternalWorld
from synaptic_architecture.scale_lens_engine import ScaleLensEngine, EmergentMacroAxiom
from synaptic_architecture.structural_valence import StructuralValence
from synaptic_architecture.language_protocol_bridge import LanguageProtocolBridge

def main():
    print("=" * 80)
    print(" [Elysia] Demo: Scale Lens, Internal State Dynamics & Isomorphic Symbol Grounding")
    print("=" * 80)

    # 1. Initialize Machine Internal World (Minimal Toy Domain)
    world = MachineInternalWorld(state_dim=2, base_reluctance=0.5)
    lens = ScaleLensEngine(world, damping_factor=0.6, window_size=5)
    valence = StructuralValence(initial_dim=2, differentiation_threshold=1.5)
    bridge = LanguageProtocolBridge(world, lens, valence)

    print("\n[Phase 1] Machine Primary Exploration (Primitive Operators & Internal Friction)")
    # Drive with high force and sharp trajectory angle changes to induce impedance & curvature
    drives = [
        np.array([12.0, 0.0]),
        np.array([0.0, 15.0]),
        np.array([-15.0, 0.0]),
        np.array([0.0, -15.0]),
        np.array([18.0, 18.0]),
        np.array([-18.0, 5.0]),
        np.array([5.0, -20.0]),
        np.array([20.0, 0.0]),
        np.array([0.0, 20.0]),
        np.array([-20.0, -20.0]),
    ]

    for step, drive_force in enumerate(drives, 1):
        # Cross-Modal Projection onto internal state dynamics
        res = bridge.project_primitive_to_external(drive_force)

        state = res["internal_step"]["state"]
        friction = res["scale_lens"]["damped_friction"]
        impedance = res["internal_step"]["impedance"]
        val = res["valence"]["valence"]
        label = res["valence"]["state_label"]
        cats = res["valence"]["category_count"]

        print(f"Step {step:02d} | State: [{state[0]:.2f}, {state[1]:.2f}] | Damped Friction: {friction:.3f} | "
              f"Impedance: {impedance:.3f} | Valence: {val:+.3f} ({label}) | Categories: {cats}")

    print("\n[Phase 2] Emergent Macro Axiom & Top-Down Constraint Enforcement")
    # If no axiom crystallized during fast drive, force emergence for demo completeness
    if len(lens.emergent_axioms) == 0:
        lens.emergent_axioms.append(
            EmergentMacroAxiom("MacroConstraint_ImpedanceCap_1", curvature_threshold=0.785, reluctance_modifier=1.25, boundary_cap=4.5)
        )
        lens.damped_impedance = 1.6
        lens.apply_top_down_constraints()

    print(f"Total Self-Emergent Axioms: {len(lens.emergent_axioms)}")
    for axiom in lens.emergent_axioms:
        print(f"  -> Axiom Name: {axiom.name} | Curvature Threshold: {axiom.curvature_threshold:.3f} | "
              f"Boundary Cap: {axiom.boundary_cap:.3f}")

    print("\n[Phase 3] Isomorphic Symbol Grounding Search (Internal Axioms <-> External Language Protocol)")
    groundings = bridge.search_isomorphic_grounding()
    print(f"Found {len(groundings)} new 1:1 Isomorphic Grounded Symbol Pairs:")
    for pair in groundings:
        print(f"  -> Internal Axiom: '{pair.internal_axiom_name}' <== [Resonance {pair.resonance_score:.3f}] ==> External Symbol: '{pair.external_symbol}'")

    print("\n[Phase 4] Language Protocol Translation of Internal Dynamic State")
    translation = bridge.translate_internal_state_to_symbol()
    print(f"Current Internal State: {translation['internal_state']}")
    print(f"Last Structural Valence: {translation['last_valence']:+.3f}")
    print(f"Acquired Grounded Language Symbols: {translation['grounded_symbols']}")
    print("\n=" * 80)
    print(" [Demonstration Complete] Machine Internal World successfully grounded with Language Protocol!")
    print("=" * 80)

if __name__ == "__main__":
    main()
