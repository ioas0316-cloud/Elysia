import numpy as np
import pytest
from core.physics.telos_attractor_field import TelosAttractorField
from core.physics.spectral_spectrum_tuner import SpectralSpectrumTuner
from core.physics.causal_compiler_engine import CausalCompilerEngine
from core.physics.topological_loom_os import TopologicalLoomOS
from simulators.telos_causal_framework_sim import TelosCausalFrameworkSimulator

def test_intent_wave_reception():
    field = TelosAttractorField(dim=8)
    intent = np.ones(8)
    res = field.receive_intent_wave(intent_vector=intent, amplitude=0.5, chromatic_bias=np.array([0.2, 0.9, 0.1]))
    assert "new_telos_center" in res
    assert res["shift_magnitude"] > 0

def test_spectral_spectrum_tuner():
    tuner = SpectralSpectrumTuner(dim=8, baseline_autonomy=0.5)
    stimulus = np.array([2.0, -1.0, 0.5, 0.0, 1.0, -0.5, 0.0, 1.0])
    harmony = np.zeros(8)
    res = tuner.evaluate_external_stimulus(stimulus, harmony)
    assert res["friction"] > 0
    assert tuner.autonomy_level >= 0.5

    curv = tuner.tune_field_curvature(np.eye(8))
    assert np.trace(curv) > 8.0

def test_phase_transition():
    field = TelosAttractorField(dim=8)
    compiler = CausalCompilerEngine(state_dim=8, telos_field=field)
    res = compiler.trigger_phase_transition(external_friction=6.0, threshold=5.0)
    assert res["phase_transition_triggered"] is True

def test_loom_mesh_adaptation():
    loom = TopologicalLoomOS(fabric_shape=(8, 8))
    loom.inject_warp_logic(0, np.ones(8) * 5.0)
    loom.inject_weft_data(0, np.ones(8) * 5.0)
    loom.weave_step()
    res = loom.adapt_topological_mesh(friction_threshold=0.1)
    assert res["mesh_reconfigured"] is True

def test_telos_causal_framework_sim():
    sim = TelosCausalFrameworkSimulator(dim=8)
    comp = sim.compare_frameworks()
    assert comp["zero_friction_achieved"] is True
    assert "autonomy_level" in comp["telos_causal"]
