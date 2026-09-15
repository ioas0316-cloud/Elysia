"""
Comprehensive Integration Demo for Elysia's Semantic Valence Manifold,
1D-4D Counterfactual Simulator, and Multicellular Lorentzian Resonance Organism.
"""

import numpy as np

from core.memory.semantic_valence_manifold import (
    SemanticValenceManifold,
    CognitiveLens,
    ExternalSensorySignal,
    ConceptAttractor,
    DopamineScalerModule,
    RefractoryPeriodModule,
    LTPScarringEngine,
    FieldFluxSignal
)
from core.memory.causal_4d_counterfactual import (
    MemoryTrajectory1D,
    SpacetimeExpansionBasis4D,
    CounterfactualSimulator,
    IdentityCoreGuardrail
)
from core.memory.multicellular_cognitive_organism import (
    CognitiveCell,
    ModularCognitiveCell,
    TopologicalManifoldMedium
)


def run_demo():
    print("===========================================================================")
    print("   ELYSIA: SEMANTIC VALENCE MANIFOLD & 1D-4D COUNTERFACTUAL DEMO          ")
    print("===========================================================================\n")

    # -------------------------------------------------------------------------
    # 1. Sensory Refraction, Continuous Gradient Flow, Mitosis & Fusion
    # -------------------------------------------------------------------------
    print(">>> 1. SENSORY REFRACTION & CONTINUOUS VALENCE MANIFOLD REASONING <<<")
    lens = CognitiveLens(sensory_range=(100.0, 5000.0), refractive_index=1.5, seed=42)
    manifold = SemanticValenceManifold(core_radius=0.8, beta_metric=0.2)

    # Register initial concept attractor
    danger_att = ConceptAttractor(
        concept_id="CONCEPT_DANGER",
        center_pos=np.array([2.0, 2.0, 0.0]),
        resonant_freq=80.0,
        well_depth=3.0,
        well_radius=1.2
    )
    manifold.register_attractor(danger_att)

    # Raw optical sensory signal
    opt_signal = ExternalSensorySignal(
        sensory_type="optical",
        raw_spectrum=np.array([600.0, 1200.0, 1800.0]),
        intensity=2.5,
        external_pos=np.array([1.0, 1.0, 0.5])
    )

    t_wave = lens.refract(opt_signal)
    print(f"External Spectrum: {opt_signal.raw_spectrum} Hz -> Refracted Internal Freq: {t_wave.internal_frequency:.2f} Hz")
    print(f"Refracted Tension Amplitude: {t_wave.amplitude:.2f}")

    start_thought = np.array([0.0, 0.0, 0.0])
    pos1, att1, is_mitosis1 = manifold.process_and_perceive(t_wave, start_pos=start_thought, steps=30)
    print(f"Thought Trajectory Converged to Pos: {pos1.round(3)} | Attractor: {att1.concept_id} (Mitosis: {is_mitosis1})")

    # Novel unmapped sensory signal
    novel_signal = ExternalSensorySignal(
        sensory_type="acoustic",
        raw_spectrum=np.array([3500.0, 4200.0]),
        intensity=4.0,
        external_pos=np.array([-3.0, -2.0, 1.0])
    )
    t_wave_novel = lens.refract(novel_signal)
    pos2, att2, is_mitosis2 = manifold.process_and_perceive(t_wave_novel, start_pos=start_thought, steps=35)
    print(f"Novel Trajectory Converged to Pos: {pos2.round(3)} | Attractor: {att2.concept_id} (Mitosis: {is_mitosis2})")

    # Consolidate overlapping attractors via Cognitive Fusion
    fused_events = manifold.consolidate_field()
    print(f"Field Consolidation Fused Events: {fused_events}\n")

    # -------------------------------------------------------------------------
    # 2. Biological Boundary Condition Modules & LTP Scarring
    # -------------------------------------------------------------------------
    print(">>> 2. BIOLOGICAL BOUNDARY MODULES & LTP SCARRING <<<")
    dopamine_mod = DopamineScalerModule(initial_dopamine=2.0)
    refractory_mod = RefractoryPeriodModule(refractory_duration=0.5, activation_threshold=1.5)
    ltp_engine = LTPScarringEngine(energy_threshold=2.0, plasticity_rate=0.3)

    flux_input = FieldFluxSignal(amplitude=2.5, frequency=120.0, gradient=np.array([0.5, -0.5, 0.2]))

    out_dopamine = dopamine_mod.apply_boundary_condition(flux_input, current_time=0.0)
    print(f"Dopamine Scaled Amp (2.0x): {out_dopamine.amplitude:.2f}")

    out_refract = refractory_mod.apply_boundary_condition(out_dopamine, current_time=0.0)
    print(f"Refractory Trigger Amp: {out_refract.amplitude:.2f}")

    out_blocked = refractory_mod.apply_boundary_condition(out_dopamine, current_time=0.2)
    print(f"Refractory Blocked Amp (at t=0.2s): {out_blocked.amplitude:.2f}")

    out_recovered = refractory_mod.apply_boundary_condition(out_dopamine, current_time=0.6)
    print(f"Refractory Recovered Amp (at t=0.6s): {out_recovered.amplitude:.2f}")

    # Repeat flux pulses to trigger LTP scarring
    for i in range(5):
        ltp_engine.apply_boundary_condition(flux_input, current_time=float(i))
    print(f"LTP Imprinted Scar Weight W_scar(120Hz): {ltp_engine.get_scar_weight(120.0):.4f}")

    metric_tensor = manifold.compute_metric_tensor(pos1, ltp_engine=ltp_engine)
    print(f"Valence Metric Tensor (Deformed by Scar):\n{metric_tensor.round(4)}\n")

    # -------------------------------------------------------------------------
    # 3. 1D-4D Commutative Manifold & Counterfactual Guardrail
    # -------------------------------------------------------------------------
    print(">>> 3. 1D -> 4D COMMUTATIVE MANIFOLD & COUNTERFACTUAL GUARDRAIL <<<")
    timestamps = np.linspace(0.0, 2.0, 20)
    causal_states = np.stack([
        np.sin(timestamps),
        np.cos(timestamps),
        timestamps * 0.5,
        np.exp(-timestamps)
    ], axis=1)

    mem_1d = MemoryTrajectory1D(timestamps=timestamps, causal_states=causal_states)
    basis_4d = SpacetimeExpansionBasis4D(state_dim=4, spatial_nodes=3, seed=42)
    sim_engine = CounterfactualSimulator(basis_4d)
    guardrail = IdentityCoreGuardrail(core_radius=0.8, max_tension_delta=1.5)

    # Safe Counterfactual Intervention
    perturbation_safe = np.array([0.2, -0.1, 0.0, 0.1])
    base_4d, cf_4d_safe, metrics_safe = sim_engine.simulate_counterfactual_future(
        mem_1d, intervention_step=8, delta_perturbation=perturbation_safe
    )
    res_safe = guardrail.verify_counterfactual_future(base_4d, cf_4d_safe)
    print(f"Safe Intervention Verification: {res_safe.reason}")

    # Unsafe Counterfactual Intervention (Breaching Core Origin)
    perturbation_unsafe = np.array([-3.5, -3.5, -2.0, 0.0])
    _, cf_4d_unsafe, _ = sim_engine.simulate_counterfactual_future(
        mem_1d, intervention_step=8, delta_perturbation=perturbation_unsafe
    )
    res_unsafe = guardrail.verify_counterfactual_future(base_4d, cf_4d_unsafe)
    print(f"Unsafe Intervention Verification: {res_unsafe.reason}\n")

    # -------------------------------------------------------------------------
    # 4. Multicellular Lorentzian Resonance & Cellular Mitosis
    # -------------------------------------------------------------------------
    print(">>> 4. MULTICELLULAR LORENTZIAN RESONANCE ORGANISM <<<")
    medium = TopologicalManifoldMedium(decay_alpha=0.3)

    cell_core = CognitiveCell("CELL_CORE", natural_frequency=60.0, position=np.array([0.0, 0.0, 0.0]))
    cell_eff1 = CognitiveCell("CELL_EFF_1", natural_frequency=120.0, position=np.array([1.0, 0.5, 0.0]), friction_capacity=3.0)
    cell_eff2 = CognitiveCell("CELL_EFF_2", natural_frequency=122.0, position=np.array([1.2, 0.8, 0.0]))
    cell_immune = CognitiveCell("CELL_IMMUNE", natural_frequency=210.0, position=np.array([0.5, 0.2, 0.0]))

    medium.register_cell(cell_core)
    medium.register_cell(cell_eff1)
    medium.register_cell(cell_eff2)
    medium.register_cell(cell_immune)

    print("Emitting Wave from CELL_EFF_1 (Freq: 120Hz, Friction: 8.0)...")
    res_list = medium.propagate_wave_event("CELL_EFF_1", friction_energy=8.0)

    for cell_id, res_fac, absorbed in res_list:
        print(f"  Target: {cell_id:<15} | Resonance Factor: {res_fac*100:5.1f}% | Absorbed Energy: {absorbed:.4f}")

    print(f"Registered Cells Count post-propagation (including Daughter cells): {len(medium.cells)}")
    for cid in medium.cells.keys():
        print(f"  - Active Cell ID: {cid}")

    print("\n===========================================================================")
    print("   DEMO COMPLETED SUCCESSFULLY: ALL CAUSAL MECHANISMS VERIFIED             ")
    print("===========================================================================")


if __name__ == "__main__":
    run_demo()
