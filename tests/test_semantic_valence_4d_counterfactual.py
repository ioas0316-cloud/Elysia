"""
Unit tests for Elysia's Semantic Valence Manifold, 1D-4D Counterfactual Simulator,
and Multicellular Lorentzian Resonance Organism.
"""

import numpy as np
import pytest

from core.memory.semantic_valence_manifold import (
    SemanticValenceManifold,
    CognitiveLens,
    ExternalSensorySignal,
    ConceptAttractor,
    CognitiveMitosisEngine,
    CognitiveFusionEngine,
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
    TopologicalManifoldMedium,
    TensionWave
)


class TestSemanticValenceManifold:
    def test_composite_potential_and_gradient(self):
        manifold = SemanticValenceManifold(core_radius=0.8)
        pos = np.array([2.0, 1.0, 0.0])

        v_val = manifold.compute_potential(pos)
        grad_v = manifold.compute_gradient(pos)

        assert isinstance(v_val, float)
        assert grad_v.shape == (3,)
        assert not np.isnan(v_val)
        assert not np.isnan(grad_v).any()

    def test_core_weight_divergence_near_rcore(self):
        manifold = SemanticValenceManifold(core_radius=0.8)
        pos_far = np.array([3.0, 0.0, 0.0])
        pos_near = np.array([0.805, 0.0, 0.0])

        w_far = manifold.calculate_w_core(pos_far)
        w_near = manifold.calculate_w_core(pos_near)

        assert w_near > w_far
        assert w_near >= 1e3

    def test_valence_weighted_metric_tensor(self):
        manifold = SemanticValenceManifold()
        pos = np.array([1.5, 0.5, 0.0])

        g_tensor = manifold.compute_metric_tensor(pos)
        assert g_tensor.shape == (3, 3)
        assert g_tensor[0, 0] > 0.0

    def test_cognitive_lens_refraction(self):
        lens = CognitiveLens(sensory_range=(100.0, 5000.0), refractive_index=1.5)
        signal = ExternalSensorySignal(
            sensory_type="optical",
            raw_spectrum=np.array([500.0, 1000.0, 1500.0]),
            intensity=2.0,
            external_pos=np.array([1.0, 2.0, 3.0])
        )

        t_wave = lens.refract(signal)
        assert isinstance(t_wave.internal_frequency, float)
        assert t_wave.amplitude > 0.0
        assert t_wave.refraction_dir.shape == (3,)

    def test_cognitive_mitosis_and_fusion(self):
        manifold = SemanticValenceManifold()
        lens = CognitiveLens(refractive_index=1.5)

        signal = ExternalSensorySignal("optical", np.array([3000.0]), 3.0, np.array([5.0, 5.0, 0.0]))
        t_wave = lens.refract(signal)

        start_pos = np.array([0.0, 0.0, 0.0])
        converged_pos, att, is_mitosis = manifold.process_and_perceive(t_wave, start_pos, steps=20)

        assert is_mitosis is True
        assert att.concept_id.startswith("CONCEPT_MITO_")
        assert len(manifold.attractors) == 1

        # Add an adjacent attractor for fusion test
        att_adj = ConceptAttractor(
            concept_id="CONCEPT_ADJ",
            center_pos=att.center_pos + np.array([0.2, 0.2, 0.0]),
            resonant_freq=att.resonant_freq + 2.0,
            well_depth=2.0,
            well_radius=0.8
        )
        manifold.register_attractor(att_adj)

        fused_events = manifold.consolidate_field()
        assert len(fused_events) == 1


class TestBiologicalBoundaryModules:
    def test_dopamine_scaler(self):
        mod = DopamineScalerModule(initial_dopamine=2.5)
        signal = FieldFluxSignal(amplitude=1.0, frequency=60.0, gradient=np.array([1.0, 0.0, 0.0]))

        out = mod.apply_boundary_condition(signal, current_time=0.0)
        assert out.amplitude == 2.5
        assert np.array_equal(out.gradient, np.array([2.5, 0.0, 0.0]))

    def test_refractory_period(self):
        mod = RefractoryPeriodModule(refractory_duration=0.5, activation_threshold=1.0)
        signal = FieldFluxSignal(amplitude=2.0, frequency=60.0, gradient=np.array([1.0, 0.0, 0.0]))

        out1 = mod.apply_boundary_condition(signal, current_time=1.0)
        assert out1.amplitude == 2.0

        out2 = mod.apply_boundary_condition(signal, current_time=1.2)
        assert out2.amplitude == 0.0

        out3 = mod.apply_boundary_condition(signal, current_time=1.6)
        assert out3.amplitude == 2.0

    def test_ltp_scarring_engine(self):
        engine = LTPScarringEngine(energy_threshold=1.0, plasticity_rate=0.2)
        signal = FieldFluxSignal(amplitude=2.0, frequency=100.0, gradient=np.array([1.0, 1.0, 1.0]))

        engine.apply_boundary_condition(signal, current_time=0.0)
        engine.apply_boundary_condition(signal, current_time=1.0)
        engine.apply_boundary_condition(signal, current_time=2.0)

        scar_weight = engine.get_scar_weight(100.0)
        assert scar_weight > 0.0


class TestCausal4DCounterfactual:
    def test_1d_to_4d_expansion(self):
        timestamps = np.linspace(0.0, 1.0, 10)
        states = np.random.randn(10, 4)
        traj1d = MemoryTrajectory1D(timestamps=timestamps, causal_states=states)

        basis = SpacetimeExpansionBasis4D(state_dim=4, spatial_nodes=3)
        manifold4d = basis.expand_1d_to_4d(traj1d)

        assert manifold4d.shape == (10, 3, 4)
        assert np.array_equal(manifold4d[:, :, 3], np.tile(timestamps[:, None], (1, 3)))

    def test_counterfactual_simulation_and_guardrail(self):
        timestamps = np.linspace(0.0, 1.0, 10)
        states = np.ones((10, 4)) * 0.5
        traj1d = MemoryTrajectory1D(timestamps=timestamps, causal_states=states)

        basis = SpacetimeExpansionBasis4D(state_dim=4, spatial_nodes=3, spatial_offset=2.0)
        sim = CounterfactualSimulator(basis)
        guardrail = IdentityCoreGuardrail(core_radius=0.8, max_tension_delta=1.5)

        # Safe perturbation
        base_4d, cf_4d_safe, metrics = sim.simulate_counterfactual_future(
            traj1d, intervention_step=4, delta_perturbation=np.array([0.1, 0.0, 0.0, 0.0])
        )
        res_safe = guardrail.verify_counterfactual_future(base_4d, cf_4d_safe)
        assert res_safe.is_approved is True

        # Core breaching perturbation (shifting coordinates directly towards origin 0,0,0)
        # Baseline coords are around 2.0; shifting state so coords drop below 0.8
        basis_zero_offset = SpacetimeExpansionBasis4D(state_dim=4, spatial_nodes=3, spatial_offset=0.2)
        sim_zero = CounterfactualSimulator(basis_zero_offset)
        base_4d_zero, cf_4d_breaching, _ = sim_zero.simulate_counterfactual_future(
            traj1d, intervention_step=4, delta_perturbation=np.array([-0.2, -0.2, -0.2, 0.0])
        )
        res_breach = guardrail.verify_counterfactual_future(base_4d_zero, cf_4d_breaching)
        assert res_breach.is_approved is False
        assert "Existential Breach" in res_breach.reason


class TestMulticellularCognitiveOrganism:
    def test_lorentzian_resonance(self):
        cell = CognitiveCell("CELL_1", natural_frequency=100.0, position=np.array([0.0, 0.0, 0.0]), quality_factor=10.0)
        wave_exact = TensionWave("SOURCE", frequency=100.0, amplitude=2.0, origin_pos=np.array([0.0, 0.0, 0.0]))
        wave_off = TensionWave("SOURCE", frequency=200.0, amplitude=2.0, origin_pos=np.array([0.0, 0.0, 0.0]))

        r_exact = cell.calculate_resonance_factor(wave_exact.frequency)
        r_off = cell.calculate_resonance_factor(wave_off.frequency)

        assert abs(r_exact - 1.0) < 1e-5
        assert r_off < 0.2

    def test_manifold_medium_propagation_and_mitosis(self):
        medium = TopologicalManifoldMedium(decay_alpha=0.2)
        cell1 = CognitiveCell("CELL_1", natural_frequency=100.0, position=np.array([0.0, 0.0, 0.0]))
        cell2 = CognitiveCell("CELL_2", natural_frequency=100.0, position=np.array([0.1, 0.0, 0.0]), friction_capacity=0.5)

        medium.register_cell(cell1)
        medium.register_cell(cell2)

        medium.propagate_wave_event("CELL_1", friction_energy=8.0)

        # CELL_2 should undergo mitosis due to friction >= capacity
        assert len(medium.cells) > 2
        daughter_keys = [k for k in medium.cells.keys() if "DAUGHTER" in k]
        assert len(daughter_keys) >= 1
