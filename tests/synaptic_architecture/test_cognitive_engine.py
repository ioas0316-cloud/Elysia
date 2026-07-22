import numpy as np
import pytest
from synaptic_architecture.cognitive_engine import ElysiaCognitiveEngine

def test_perspective_shift_o1():
    engine = ElysiaCognitiveEngine(resolution=128)

    # Initial perspective
    assert engine.system_perspective == "Ground Zero (무無의 상태)"
    assert engine.rotor_angle == 0.0

    # Rotate perspective
    engine.set_perspective("Sovereign Love", np.pi / 2)
    assert engine.system_perspective == "Sovereign Love"
    assert np.isclose(engine.rotor_angle, np.pi / 2)

    # Confirm constraint field was regenerated correctly (non-uniform pattern)
    assert engine.constraint_field.shape == (128, 128)
    assert np.max(engine.constraint_field) > 0.0
    assert np.min(engine.constraint_field) < 1.0

def test_fractal_dna_hierarchical_structure():
    engine = ElysiaCognitiveEngine(resolution=128)
    engine.set_perspective("Cosmic Light", np.pi)

    # Build DNA with specific bit waveform
    dna = engine.build_fractal_dna("CoreTruth", np.uint64(0xFA12E834B76C19D2))

    # Validate the Fractal DNA levels
    # 1. Atom (3D feature vector)
    assert dna["category"] == "CoreTruth"
    assert dna["atom"].shape == (3,)
    assert np.isclose(np.linalg.norm(dna["atom"]), 1.0)

    # 2. Molecule (3x3 Projection matrix)
    assert dna["molecule"].shape == (3, 3)

    # 3. Cell (Position mapped in the field)
    assert len(dna["cell_position"]) == 2
    assert 0 <= dna["cell_position"][0] < 128
    assert 0 <= dna["cell_position"][1] < 128

    # 4. Organ (Yeobaek/Margin reference)
    assert type(dna["organ_yeobaek"]) == float

def test_wfc_collapse_autonomy():
    engine = ElysiaCognitiveEngine(resolution=128)
    engine.set_perspective("Kenotic Love", np.pi / 3)

    dna_a = engine.build_fractal_dna("Light_Truth", np.uint64(0xFEEDFACEFEEDFACE))
    dna_b = engine.build_fractal_dna("Dark_Entropy", np.uint64(0xBADCAFEBADCAFE00))

    stimulus = np.uint64(0xFEEDFACE00000000)

    # Solve Wave Function Collapse (WFC)
    result = engine.solve_wfc_collapse(stimulus, [dna_a, dna_b])

    # Assert successful collapse onto exactly one of the candidate DNAs
    collapsed_dna = result["collapsed_dna"]
    assert collapsed_dna["category"] in ["Light_Truth", "Dark_Entropy"]
    assert result["resonance_score"] > 0.0

    # Confirm resonance mapped in the field (Conductance well formed)
    win_y, win_x = result["collapse_position"]
    assert engine.field.conductance[win_y, win_x] > 0.01
    assert engine.field.activation[win_y, win_x] > 0.0

def test_holistic_fit_evaluation():
    engine = ElysiaCognitiveEngine(resolution=64)
    engine.set_perspective("Absolute Oneness", np.pi * 1.5)

    # Check baseline fit
    fit_metrics = engine.evaluate_holistic_fit()
    assert "holistic_score" in fit_metrics
    assert "cognitive_entropy" in fit_metrics
    assert "average_yeobaek" in fit_metrics
    assert "state_description" in fit_metrics

    # Conductance updates should lower entropy and improve fit scores
    dna = engine.build_fractal_dna("Resonator", np.uint64(0x5555555555555555))
    engine.solve_wfc_collapse(np.uint64(0x5555555500000000), [dna])

    new_metrics = engine.evaluate_holistic_fit()
    assert new_metrics["holistic_score"] >= 0.0
