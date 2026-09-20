"""
Unit and Integration Tests for Ecological Synesthetic Causal Manifold.

Tests:
1. EcologicalReceptorMatrix: Multi-modal signal extraction and wave tensor packing.
2. SynestheticCausalField: Phase interference, superposition, phase locking, and wave dynamics.
3. BodySchemaEvaluator: Efference Copy generation, prediction error calculation, self vs external shock boundary classification, and interoceptive strain accumulation.
4. MultiScalePhaseLossEngine: Multi-scale loss calculation, Ricci flow dual backpropagation.
5. DynamicGraphRewriter: Hypergraph construction, Categorical Double Pushout (DPO) topological surgeries (Node Unfolding, Edge Rewiring, Scale Elevation).
6. End-to-End Integration: Complete closed-loop workflow.
"""

import pytest
import math
import torch
import numpy as np

from core.sensory.ecological_receptor_matrix import EcologicalReceptorMatrix, SensoryModality
from core.physics.synesthetic_causal_field import SynestheticCausalField
from core.sensory.body_schema_evaluator import BodySchemaEvaluator
from core.physics.multiscale_phase_loss import MultiScalePhaseLossEngine
from core.topology.dynamic_graph_rewriter import DynamicGraphRewriter, SurgeryType


def test_ecological_receptor_matrix():
    dim = 32
    matrix = EcologicalReceptorMatrix(dimension=dim)

    # Test Audio extraction
    audio_sig = matrix.extract_audio_wave({"amplitude": 2.0, "frequency": 440.0})
    assert audio_sig.modality == SensoryModality.AUDIO
    assert len(audio_sig.amplitude) == dim

    # Test Visual Optic Flow extraction
    visual_sig = matrix.extract_optic_flow({"optic_flow": [1.0, 2.0, 0.0], "density": 1.5})
    assert visual_sig.modality == SensoryModality.VISUAL_OPTIC_FLOW
    assert visual_sig.wavevector.shape == (dim, 3)

    # Test System Impedance extraction
    imp_sig = matrix.extract_impedance_friction({"latency_ms": 25.0, "impedance": 0.5})
    assert imp_sig.modality == SensoryModality.IMPEDANCE_FRICTION
    assert imp_sig.phase_shift > 0

    # Test Process and Pack
    waveform_tensor = matrix.process_and_pack(
        audio_input={"amplitude": 1.0},
        visual_input={"optic_flow": [0, 1, 0]},
        system_input=10.0
    )
    assert waveform_tensor.amplitude_tensor.shape == torch.Size([dim])
    assert waveform_tensor.frequency_tensor.shape == torch.Size([dim])


def test_synesthetic_causal_field():
    dim = 32
    matrix = EcologicalReceptorMatrix(dimension=dim)
    waveform = matrix.process_and_pack(audio_input=1.0)

    field = SynestheticCausalField(dimension=dim)
    field.inject_waveform(waveform)

    # Test time evolution & phase coherence
    lock_state1 = field.evolve_time(dt=0.001)
    assert lock_state1.phase_coherence > 0.8 # High coherence on small dt
    assert lock_state1.constructive_energy > 0

    # Test friction accumulation
    friction_vec = torch.ones(dim) * 0.5
    field.accumulate_friction(friction_vec)
    assert torch.all(field.accumulated_friction_scar > 0)


def test_body_schema_evaluator():
    dim = 32
    evaluator = BodySchemaEvaluator(dimension=dim, self_boundary_threshold=1.0)

    action_vec = torch.randn(dim)
    eff_copy = evaluator.generate_efference_copy("act_test", action_vec, timestamp=0.0)

    assert eff_copy.predicted_sensory_outcome.shape == torch.Size([dim])

    # Test self-agency feedback
    self_input = eff_copy.predicted_sensory_outcome + torch.randn(dim) * 0.05
    state_self = evaluator.evaluate_afferent_feedback(self_input)
    assert state_self.is_self_agency is True
    assert state_self.is_external_shock is False

    # Test external shock feedback
    shock_input = torch.randn(dim) * 5.0
    state_shock = evaluator.evaluate_afferent_feedback(shock_input, external_friction_impedance=2.0)
    assert state_shock.is_external_shock is True
    assert state_shock.internal_strain > 0


def test_multiscale_phase_loss_engine():
    dim = 32
    engine = MultiScalePhaseLossEngine(dimension=dim, critical_friction_threshold=1.5)

    micro = torch.randn(dim, requires_grad=True)
    macro = torch.randn(dim, requires_grad=True)
    prev = torch.randn(dim)
    f_pred = torch.randn(dim)
    omega_ext = torch.randn(dim) * 3.0

    loss_comp = engine(micro, macro, prev, f_pred, omega_ext)

    assert loss_comp.l_total.item() > 0
    assert loss_comp.l_scale.item() >= 0
    assert loss_comp.l_causal.item() >= 0
    assert loss_comp.l_friction.item() >= 0

    # Test dual backprop
    engine.execute_dual_backprop(loss_comp, [micro, macro])
    assert micro.grad is not None


def test_dynamic_graph_rewriter():
    dim = 32
    rewriter = DynamicGraphRewriter(dimension=dim)

    # Build initial hypergraph
    n_micro = rewriter.add_node("micro_n", torch.randn(dim), scale=0.0)
    n_macro = rewriter.add_node("macro_n", torch.randn(dim), scale=1.0)
    edge = rewriter.add_edge("macro_n", "micro_n")
    hyper = rewriter.add_hyperedge("h1", ["micro_n", "macro_n"], macro_scale=1.0)

    assert len(rewriter.nodes) == 2
    assert len(rewriter.edges) == 1
    assert len(rewriter.hyperedges) == 1

    # Execute DPO rewriting on high shock
    shock = torch.randn(dim) * 4.0
    surgeries = rewriter.execute_double_pushout_rewriting(shock, friction_magnitude=5.0)

    assert len(surgeries) > 0
    assert len(rewriter.surgery_history) == len(surgeries)


def test_end_to_end_manifold_integration():
    dim = 32
    receptor_matrix = EcologicalReceptorMatrix(dimension=dim)
    causal_field = SynestheticCausalField(dimension=dim)
    body_evaluator = BodySchemaEvaluator(dimension=dim)
    loss_engine = MultiScalePhaseLossEngine(dimension=dim)
    graph_rewriter = DynamicGraphRewriter(dimension=dim)

    # 1. Action & Efference copy
    act = torch.randn(dim)
    eff = body_evaluator.generate_efference_copy("act_1", act, timestamp=0.0)

    # 2. Field Injection & Evolution
    tensor = receptor_matrix.process_and_pack(audio_input=10.0, visual_input=[1, 0, 0], system_input=5.0)
    causal_field.inject_waveform(tensor)
    field_state = causal_field.evolve_time(0.01)
    assert field_state.phase_coherence > 0

    # 3. External Shock & Evaluation
    shock = torch.randn(dim) * 3.0
    body_state = body_evaluator.evaluate_afferent_feedback(shock, external_friction_impedance=1.0)
    assert body_state.is_external_shock is True

    # 4. Loss & Dual Backprop
    micro = causal_field.get_real_potential().requires_grad_()
    macro = torch.randn(dim, requires_grad=True)
    loss_comp = loss_engine(micro, macro, micro.detach() * 0.9, micro, shock)
    loss_engine.execute_dual_backprop(loss_comp, [micro, macro])

    # 5. DPO Surgery
    surgeries = graph_rewriter.execute_double_pushout_rewriting(shock, friction_magnitude=loss_comp.l_friction.item())
    assert isinstance(surgeries, list)
