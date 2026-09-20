import pytest
import numpy as np
from core.sensory.multimodal_cognitive_frontend import (
    BioTransductionDeconstructor,
    AsynchronousPhaseLockTransducer,
    DualAxisDisentangler,
    MultimodalCognitiveFrontend,
)


def test_bio_transduction_deconstructor_visual():
    deconstructor = BioTransductionDeconstructor(spectral_resolution=32)
    rgb_flat = np.array([255, 0, 0])
    spectral_flat = deconstructor.deconstruct_visual(rgb_flat)
    assert spectral_flat.shape == (32,)
    assert np.all(spectral_flat >= 0.0)

    rgb_img = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_img[:, :, 0] = 200
    spectral_img = deconstructor.deconstruct_visual(rgb_img)
    assert spectral_img.shape == (10, 10, 32)


def test_bio_transduction_deconstructor_audio():
    deconstructor = BioTransductionDeconstructor(frequency_bands=16)
    audio_wave = np.sin(np.linspace(0, 2 * np.pi * 440, 44100))
    audio_field = deconstructor.deconstruct_audio(audio_wave)
    assert audio_field.shape == (16, 2)


def test_bio_transduction_deconstructor_text():
    deconstructor = BioTransductionDeconstructor()
    text = "Chernobyl valve opening sacrifice for humanity"
    vec = deconstructor.deconstruct_text(text)
    assert vec.shape == (16,)
    assert pytest.approx(np.linalg.norm(vec), abs=1e-5) == 1.0


def test_asynchronous_phase_lock_transducer():
    transducer = AsynchronousPhaseLockTransducer(num_modalities=3, coupling_strength=1.5)
    initial_phases = np.array([0.0, 1.5, -1.5])
    natural_freqs = np.array([1.0, 1.0, 1.0])

    locked_phases, order_param = transducer.synchronize(
        initial_phases, natural_freqs, steps=200, dt=0.01
    )
    assert order_param > 0.8  # Strong phase-locking order parameter
    assert len(locked_phases) == 3


def test_dual_axis_disentangler_orthogonality():
    disentangler = DualAxisDisentangler(feature_dim=16)
    raw_combined = np.random.randn(32)
    res = disentangler.disentangle(raw_combined)

    axis_a = res["axis_a_topology"]
    axis_b = res["axis_b_qualia"]
    dot_prod = res["orthogonality_dot_product"]

    assert axis_a.shape == (16,)
    assert axis_b.shape == (16,)
    assert pytest.approx(dot_prod, abs=1e-6) == 0.0  # Strict orthogonality


def test_multimodal_cognitive_frontend_integration():
    frontend = MultimodalCognitiveFrontend(feature_dim=16)
    rgb_image = np.array([100, 150, 200])
    audio_wave = np.sin(np.linspace(0, 10, 100))
    text_input = "Duty and sacrifice over self-preservation"

    output = frontend.process_multimodal_input(rgb_image, audio_wave, text_input)

    assert "axis_a_topology" in output
    assert "axis_b_qualia" in output
    assert "phase_lock_order_parameter" in output
    assert pytest.approx(output["orthogonality_dot_product"], abs=1e-6) == 0.0
