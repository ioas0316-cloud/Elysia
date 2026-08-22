"""
Unit Test Suite for Scale Lens Engine and Machine Internal World Architecture.
"""

import pytest
import numpy as np

from synaptic_architecture.machine_internal_world import MachineInternalWorld
from synaptic_architecture.scale_lens_engine import ScaleLensEngine
from synaptic_architecture.structural_valence import StructuralValence
from synaptic_architecture.language_protocol_bridge import LanguageProtocolBridge


class TestMachineInternalWorld:
    """Test suite for MachineInternalWorld dynamics."""

    def test_initialization(self):
        world = MachineInternalWorld(grid_size=16)
        state = world.get_state()
        assert state["current_pos"] == [0.0, 0.0]
        assert state["mean_remanence"] == 0.0
        assert state["internal_entropy"] == 0.1

    def test_push_against_resistance(self):
        world = MachineInternalWorld(grid_size=16, reluctance_coeff=0.2)
        moved_norm, friction = world.push_against_resistance(0.2, 0.3)
        assert moved_norm > 0.0
        assert friction >= 0.01
        assert world.get_state()["mean_remanence"] > 0.0

    def test_tune_frequency(self):
        world = MachineInternalWorld(grid_size=16)
        resonance = world.tune_frequency(frequency=2.0, phase=0.0)
        assert 0.0 <= resonance <= 1.0

    def test_homeostatic_drive(self):
        world = MachineInternalWorld(grid_size=16, homeostatic_target=0.5)
        err = world.apply_homeostatic_drive()
        assert err >= 0.0

    def test_probe_friction(self):
        world = MachineInternalWorld(grid_size=16)
        ext_signal = np.ones((16, 16), dtype=np.float32)
        probe = world.probe_friction(ext_signal)
        assert "spatial_friction" in probe
        assert "cross_modal_friction" in probe
        assert "total_impedance" in probe


class TestScaleLensEngine:
    """Test suite for ScaleLensEngine dynamics."""

    def test_initialization(self):
        lens = ScaleLensEngine(num_cells=256)
        assert len(lens.micro_phase) == 256
        assert len(lens.macro_potential) == 256

    def test_process_time_scale_lens(self):
        lens = ScaleLensEngine(num_cells=256, hysteresis_thresh=0.5)
        impulse = np.ones(256, dtype=np.float32) * 0.1
        metrics = lens.process_time_scale_lens(external_micro_impulse=impulse)
        assert 0.0 <= metrics["mean_coherence"] <= 1.0
        assert metrics["active_precipitated_cells"] >= 0

    def test_top_down_constraint(self):
        lens = ScaleLensEngine(num_cells=256)
        lens.macro_potential[10:20] = 0.8
        delta = lens.apply_top_down_constraint()
        assert delta >= 0.0

    def test_counterfactual_simulation(self):
        lens = ScaleLensEngine(num_cells=256)
        impulses = [np.ones(256, dtype=np.float32) * 0.02 for _ in range(3)]
        cf_res = lens.run_counterfactual_simulation(impulses, horizon_steps=3)
        assert cf_res["horizon_steps"] == 3
        assert len(cf_res["coherence_trajectory"]) == 3


class TestStructuralValence:
    """Test suite for StructuralValence evaluator."""

    def test_evaluate_valence(self):
        valence_eval = StructuralValence()
        val = valence_eval.evaluate_valence(resonance_score=0.8, friction=0.1, homeostatic_alignment=0.9)
        assert val > 0.0

    def test_category_differentiation(self):
        valence_eval = StructuralValence(friction_threshold=0.3)
        pos = np.array([0.5, 0.5])
        res = valence_eval.check_category_differentiation(pos, friction=0.5)
        assert res["differentiated"] is True
        assert res["total_categories"] == 2


class TestLanguageProtocolBridge:
    """Test suite for LanguageProtocolBridge."""

    def test_align_internal_to_external_symbol(self):
        bridge = LanguageProtocolBridge()
        res = bridge.align_internal_to_external_symbol(
            macro_potential_mean=0.6,
            coherence_mean=0.8,
            friction_mean=0.1,
            valence=0.4,
        )
        assert "grounded_symbol" in res
        assert res["isomorphism_score"] > 0.0

    def test_inter_subjective_mirror_resonance(self):
        bridge = LanguageProtocolBridge()
        p1 = np.array([0.5, 0.8, 0.2], dtype=np.float32)
        p2 = np.array([0.5, 0.8, 0.2], dtype=np.float32)
        res = bridge.inter_subjective_mirror_resonance(p1, p2)
        assert res["mirror_resonance"] == pytest.approx(1.0, abs=1e-4)
        assert res["coordination_aligned"] is True
