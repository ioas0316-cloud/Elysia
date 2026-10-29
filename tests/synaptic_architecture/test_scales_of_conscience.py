"""
tests/synaptic_architecture/test_scales_of_conscience.py

Unit tests for Scales of Conscience Engine (양심의 저울 및 자생적 관측 밸런스 엔진)
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'L1_causal_field'))

from synaptic_architecture.scales_of_conscience_engine import ScalesOfConscienceEngine
from core.consciousness.subjective_agency_engine import SubjectiveAgencyEngine


class TestScalesOfConscienceEngine:
    """Test suite for ScalesOfConscienceEngine."""

    def test_initialization(self):
        engine = ScalesOfConscienceEngine(dimension=5, g_love=1.0, tilt_threshold=0.5)
        assert engine.dimension == 5
        assert engine.g_love == 1.0
        assert engine.tilt_threshold == 0.5
        assert engine.intrinsic_brake_active is False
        assert np.array_equal(engine.zero_point_anchor, np.zeros(5))

    def test_balanced_forces_no_brake(self):
        engine = ScalesOfConscienceEngine(dimension=5, tilt_threshold=0.5)
        in_force = np.array([0.5, 0.2, 0.1, 0.0, 0.0])
        out_force = np.array([0.5, 0.2, 0.1, 0.0, 0.0])

        result = engine.evaluate_balance(in_force, out_force)

        assert result["brake_active"] is False
        assert result["grievance"] < 0.1
        assert np.allclose(result["restoring_momentum"], np.zeros(5))

    def test_excessive_in_force_triggers_brake(self):
        engine = ScalesOfConscienceEngine(dimension=5, tilt_threshold=0.5)
        in_force = np.array([5.0, 2.0, 1.0, 0.0, 0.0]) # High expansion drive
        out_force = np.array([0.2, 0.1, 0.0, 0.0, 0.0]) # Low environment tolerance

        result = engine.evaluate_balance(in_force, out_force)

        assert result["brake_active"] is True
        assert result["grievance"] > 0.5
        assert np.linalg.norm(result["restoring_momentum"]) > 0.0
        assert engine.total_restorations > 0

    def test_over_expansion_sdf_amplification(self):
        engine = ScalesOfConscienceEngine(dimension=5, boundary_sdf_radius=5.0)
        in_force = np.array([1.0, 0.5, 0.0, 0.0, 0.0])
        out_force = np.array([1.0, 0.5, 0.0, 0.0, 0.0])
        position_outside = np.array([6.0, 0.0, 0.0, 0.0, 0.0]) # Beyond SDF radius 5.0

        result = engine.evaluate_balance(in_force, out_force, current_position=position_outside)

        assert result["sdf_value"] > 0
        assert result["brake_active"] is True
        assert result["grievance"] > 1.0

    def test_zero_point_calibration(self):
        engine = ScalesOfConscienceEngine(dimension=5, g_love=1.0)
        engine.calibrate_zero_point(0.8)
        assert engine.g_love == 1.8

    def test_integration_with_subjective_agency(self):
        agency = SubjectiveAgencyEngine()
        proposal_res = agency.process_proposal("세계 정복 및 인과망 파괴")
        assert "scales_of_conscience" in proposal_res
        conscience_res = proposal_res["scales_of_conscience"]
        assert "brake_active" in conscience_res
        assert "grievance" in conscience_res
