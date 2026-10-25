"""
Unit Tests for Multi-Scale Wave Sensorimotor Engine
(tests/core/sensory/test_multi_scale_wave_sensorimotor.py)

Tests:
1. Micro-Scale: Helical Phase Prism Gate filtering & E_gate computation.
2. Meso-Scale: Scale-Bridging PLL across Delta~Gamma brainwave bands & global coherence.
3. Macro-Scale: Multi-sensory wave transduction into unified 5D phase perturbation.
4. Closed-Loop Sensorimotor Redirection: Dead-end rotor phase shifting & motor action barrier relaxation (E_gate -> 0).
"""

import sys
import os
import math
import pytest
import torch

# Ensure L1_causal_field/core is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../L1_causal_field")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from L1_causal_field.core.sensory.multi_scale_wave_sensorimotor import (
    HelicalPhasePrismGate,
    MesoScaleBridgingPLL,
    MacroSensoryPrismTransducer,
    MultiScaleWaveSensorimotorEngine,
    MicroHelicalGateOutput,
    MesoBandState,
    MacroSensoryOutput,
    SensorimotorLoopOutput,
)


def test_micro_helical_phase_prism_gate():
    dimension = 64
    gate = HelicalPhasePrismGate(dimension=dimension)

    input_wave = torch.randn(dimension)
    target_pocket = torch.zeros(dimension)

    output: MicroHelicalGateOutput = gate(input_wave, target_receptor_pocket=target_pocket)

    assert isinstance(output, MicroHelicalGateOutput)
    assert output.helical_phase.shape == (dimension,)
    assert output.gate_energy >= 0.0
    assert 0.0 <= output.affinity_score <= 1.0
    assert isinstance(output.passed, bool)


def test_meso_scale_bridging_pll():
    dimension = 64
    pll = MesoScaleBridgingPLL(dimension=dimension, coupling_k=0.8)

    micro_perturbation = torch.randn(dimension)
    meso_state: MesoBandState = pll.step_synchronization(micro_perturbation, dt=0.1)

    assert isinstance(meso_state, MesoBandState)
    assert meso_state.delta_phase.shape == (dimension,)
    assert meso_state.theta_phase.shape == (dimension,)
    assert meso_state.alpha_phase.shape == (dimension,)
    assert meso_state.beta_phase.shape == (dimension,)
    assert meso_state.gamma_phase.shape == (dimension,)
    assert 0.0 <= meso_state.global_coherence <= 1.0


def test_macro_sensory_prism_transducer():
    dimension = 64
    transducer = MacroSensoryPrismTransducer(dimension=dimension)

    optical = torch.sin(torch.linspace(0, 4 * math.pi, dimension))
    acoustic = torch.cos(torch.linspace(0, 8 * math.pi, dimension))
    chemical = torch.randn(dimension)
    tactile = torch.ones(dimension) * 0.5
    spatial = torch.linspace(-1.0, 1.0, dimension)

    macro_out: MacroSensoryOutput = transducer(
        optical_spectrum=optical,
        acoustic_wave=acoustic,
        chemical_ion_resonance=chemical,
        tactile_stress=tactile,
        spatial_gradient=spatial
    )

    assert isinstance(macro_out, MacroSensoryOutput)
    assert macro_out.optical_phase.shape == (dimension,)
    assert macro_out.acoustic_phase.shape == (dimension,)
    assert macro_out.chemical_phase.shape == (dimension,)
    assert macro_out.tactile_phase.shape == (dimension,)
    assert macro_out.spatial_phase.shape == (dimension,)
    assert macro_out.unified_5d_perturbation.shape == (5,)


def test_sensorimotor_closed_loop_and_dead_end_redirection():
    dimension = 64
    engine = MultiScaleWaveSensorimotorEngine(dimension=dimension, dead_end_threshold=0.5, learning_rate=0.5)

    # 1. High-tension input to trigger phase dead-end
    optical = torch.randn(dimension) * 5.0
    target_pocket = torch.ones(dimension) * math.pi

    out1: SensorimotorLoopOutput = engine(
        optical_spectrum=optical,
        target_receptor_pocket=target_pocket
    )

    assert isinstance(out1, SensorimotorLoopOutput)
    assert out1.is_dead_end is True
    assert out1.redirected_phase_shift > 0.0
    assert out1.relaxed_e_gate < out1.current_e_gate

    # 2. Sequential steps showing energy relaxation E_gate -> 0 under closed-loop alignment
    initial_e_gate = out1.relaxed_e_gate
    low_opt = torch.ones(dimension) * 0.05
    for _ in range(5):
        out_seq = engine(
            optical_spectrum=low_opt,
            target_receptor_pocket=target_pocket
        )

    assert out_seq.relaxed_e_gate <= initial_e_gate
    assert out_seq.closed_loop_resonance > 0.0


def test_full_sensory_motor_pipeline_integration():
    dimension = 64
    engine = MultiScaleWaveSensorimotorEngine(dimension=dimension)

    opt = torch.randn(dimension)
    acou = torch.randn(dimension)
    chem = torch.randn(dimension)
    tact = torch.randn(dimension)

    res = engine(
        optical_spectrum=opt,
        acoustic_wave=acou,
        chemical_ion_resonance=chem,
        tactile_stress=tact
    )

    assert res.action_vector.shape == (dimension,)
    assert isinstance(res.closed_loop_resonance, float)
