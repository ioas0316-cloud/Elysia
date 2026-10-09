import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_core.biological_cognitive_engine import (
    MacroValueManifold,
    HJBTopDownAttentionPipeline,
    BiologicalCognitiveEngine,
    PyFallbackAutonomicTensionController
)

try:
    import autonomic_tension_cpp
    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False


def test_autonomic_controller_py_fallback():
    controller = PyFallbackAutonomicTensionController(kp=1.2, ki=0.1, kd=0.4, th=0.5)
    micro_phase_error = torch.randn(1, 16, 32, 32)
    wave_field = torch.randn(1, 16, 32, 32)
    dt = 0.016

    relaxed_field, symp_weight, parasymp_weight = controller.step(micro_phase_error, wave_field, dt)

    assert relaxed_field.shape == wave_field.shape
    assert 0.0 <= symp_weight <= 1.0
    assert 0.0 <= parasymp_weight <= 1.0
    assert abs((symp_weight + parasymp_weight) - 1.0) < 1e-5


@pytest.mark.skipif(not HAS_CPP_EXT, reason="autonomic_tension_cpp C++ extension not installed")
def test_autonomic_controller_cpp_extension():
    controller = autonomic_tension_cpp.AutonomicTensionController(1.2, 0.1, 0.4, 0.5)
    micro_phase_error = torch.randn(1, 16, 32, 32)
    wave_field = torch.randn(1, 16, 32, 32)
    dt = 0.016

    relaxed_field, symp_weight, parasymp_weight = controller.step(micro_phase_error, wave_field, dt)

    assert relaxed_field.shape == wave_field.shape
    assert 0.0 <= symp_weight <= 1.0
    assert 0.0 <= parasymp_weight <= 1.0
    assert abs((symp_weight + parasymp_weight) - 1.0) < 1e-5


def test_macro_value_manifold():
    manifold = MacroValueManifold(channels=16)
    state = torch.randn(1, 16, 32, 32)
    out = manifold(state)

    assert out.shape == state.shape
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_hjb_top_down_attention_pipeline():
    manifold = MacroValueManifold(channels=16)
    pipeline = HJBTopDownAttentionPipeline(channels=16)
    current_state = torch.randn(1, 16, 32, 32)

    sculpted, mask = pipeline(manifold, current_state)

    assert sculpted.shape == current_state.shape
    assert mask.shape == (1, 1, 32, 32)
    assert abs(mask.sum().item() - 1.0) < 1e-4


def test_biological_cognitive_engine_step():
    B, C, H, W = 1, 16, 32, 32
    engine = BiologicalCognitiveEngine(channels=C, height=H, width=W)
    prev_state = torch.randn(B, C, H, W)
    raw_sensory_wave = torch.randn(B, C, H, W)
    dt = 0.016

    output = engine.step(raw_sensory_wave, prev_state, dt)

    assert "next_state" in output
    assert "sensory_mask" in output
    assert "sympathetic_weight" in output
    assert "parasympathetic_weight" in output
    assert "phase_error_norm" in output

    assert output["next_state"].shape == (B, C, H, W)
    assert output["sensory_mask"].shape == (B, 1, H, W)
    assert isinstance(output["phase_error_norm"], float)


def test_biological_cognitive_engine_unidirectional_loop():
    B, C, H, W = 1, 8, 16, 16
    engine = BiologicalCognitiveEngine(channels=C, height=H, width=W)
    current_state = torch.randn(B, C, H, W)
    dt = 0.016

    for t in range(10):
        raw_wave = torch.randn(B, C, H, W)
        if t == 5:
            raw_wave += 10.0 * torch.randn(B, C, H, W) # disturbance
        res = engine.step(raw_wave, current_state, dt)
        current_state = res["next_state"]
        assert current_state.shape == (B, C, H, W)
