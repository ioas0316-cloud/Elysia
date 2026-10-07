"""
Comprehensive Unit Tests for Wisdom-Causal Loss, Phase Resonance, & Meta-Dimensional Leap Engines
"""

import pytest
import math
import torch

from core.physics.wisdom_causal_loss import WisdomCausalLossEngine
from core.physics.fractal_cell_resonance_engine import FractalCellResonanceEngine
from core.sensory.unified_phase_transducer import UnifiedPhaseTransducerEngine
from core.topology.observation_horizon_boundary import ObservationHorizonBoundaryEngine
from core.consciousness.autonomous_phase_rearrangement import AutonomousPhaseRearrangementEngine
from core.consciousness.meta_dimensional_leap import MetaDimensionalLeapEngine


def test_wisdom_causal_loss_engine():
    engine = WisdomCausalLossEngine(num_scales=5, dimension=16)
    micro_shock = torch.randn(16)
    scale_tensors = [torch.randn(16) for _ in range(5)]
    phase_field = torch.randn(16)

    loss_out = engine(micro_shock, scale_tensors, phase_field)

    assert loss_out.l_wisdom_total.item() > 0.0
    assert loss_out.l_cascade.item() >= 0.0
    assert loss_out.l_macro_deform.item() >= 0.0
    assert loss_out.l_entropy.item() >= 0.0
    assert loss_out.l_trinity.item() >= 0.0
    assert loss_out.trinity_regularity > 0.0

    # Test backpropagation execution
    micro_params = [nn_param for nn_param in engine.parameters() if nn_param.requires_grad]
    engine.execute_wisdom_backprop(loss_out, micro_params)


def test_fractal_cell_resonance_engine():
    engine = FractalCellResonanceEngine(dimension=16)
    external_wave = torch.randn(16)

    out = engine(external_wave)
    cell_state = out["cell_state"]
    res_result = out["resonance_result"]

    assert cell_state.sin_val.shape == (16,)
    assert cell_state.cos_val.shape == (16,)
    assert cell_state.tan_val.shape == (16,)
    assert res_result.phase_error >= 0.0

    # Test imagination wave synthesis
    virtual_offset = torch.randn(16) * 0.1
    imagine_wave = engine.synthesize_imagination_wave(virtual_offset)
    assert imagine_wave.shape == (16,)

    # Test scientific reasoning verification
    macro_wave = torch.randn(16)
    is_valid, phase_err, inter_pat = engine.verify_scientific_reasoning(imagine_wave, macro_wave)
    assert isinstance(is_valid, bool)
    assert phase_err >= 0.0


def test_unified_phase_transducer_engine():
    engine = UnifiedPhaseTransducerEngine(dimension=16)

    vision = torch.randn(8, 8)
    audio = torch.randn(16)
    shear = torch.randn(16)
    normal = torch.ones(16)
    potential = torch.randn(16)
    imagination = torch.randn(16)

    out = engine(
        vision_input=vision,
        audio_input=audio,
        shear_input=shear,
        normal_input=normal,
        potential_input=potential,
        imagination_input=imagination
    )

    assert out.theta_sensory.shape == (16,)
    assert out.bivector_10d.shape == (10,)
    assert out.r_rotor_5d.shape == (5, 5)
    assert out.state_v5d_transformed.shape == (5,)
    assert out.closed_loop_resonance >= 0.0


def test_observation_horizon_boundary_engine():
    engine = ObservationHorizonBoundaryEngine(dimension=16)
    obs = torch.randn(16)

    out = engine(obs)
    horizon = out["horizon_state"]
    bell_res = out["bell_result"]

    assert horizon.w_seen.shape == (16,)
    assert horizon.w_unseen.shape == (16,)
    assert horizon.boundary_tension_dW.shape == (16,)
    assert 0.0 <= horizon.horizon_ratio <= 1.0

    # Test Tsirelson's bound Bell correlation (S = 2.828...)
    assert pytest.approx(bell_res.correlation_S, 0.01) == 2.8284
    assert bell_res.is_quantum_nonlocal is True


def test_autonomous_phase_rearrangement_engine():
    engine = AutonomousPhaseRearrangementEngine(dimension=16, stress_threshold=1.0)
    shock = torch.randn(16) * 4.0

    out = engine(shock)
    history = out["rearrangement_history"]

    assert len(history) == 4
    assert history[0].step_index == 1
    assert history[1].step_index == 2
    assert history[2].step_index == 3
    assert history[3].step_index == 4
    assert out["final_phase_field"].shape == (16,)
    assert out["final_rotor_state_5d"].shape == (5,)


def test_meta_dimensional_leap_engine():
    engine = MetaDimensionalLeapEngine(base_dimension=16, subsumption_threshold=2.0)
    turb = torch.randn(16) * 3.0
    shock = torch.randn(16) * 3.0

    result = engine(phase_turbulence=turb, boundary_stress=3.5, shock_field=shock)

    assert result.lower_contradiction_energy > 2.0
    assert result.is_meta_leap_achieved is True
    assert result.meta_dimension_index == 32 # 16 + 16
    assert result.meta_rule_tensor.shape == (5,)
    assert 0.0 <= result.instantaneous_coherence <= 1.0
