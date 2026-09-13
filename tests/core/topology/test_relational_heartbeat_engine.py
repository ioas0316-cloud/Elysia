"""
Unit Tests for Relational Heartbeat Engine & Transcendent Causal Trajectory
=============================================================================
"""

import pytest
import numpy as np
from core.topology.relational_heartbeat_engine import (
    RelationalHeartbeatOscillator,
    FinitudeBoundaryTracker,
    AltruisticCreationEngine,
    RelationalHeartbeatEngine
)
from core.topology.archetypal_identity_boundary import (
    OntologicalZeroBackground,
    ArchetypalIdentityBoundary
)
from core.topology.self_referential_architecture import SelfReferentialArchitectureEngine


class TestRelationalHeartbeatEngine:
    def test_relational_heartbeat_oscillator_shatters_stagnation(self):
        oscillator = RelationalHeartbeatOscillator(vector_dim=8, stagnation_threshold=0.1)

        # 닫힌 수렴 신호 이력 생성 (Stagnant signal history)
        stagnant_signal = np.ones(8) * 0.5
        signal_history = [stagnant_signal.copy(), stagnant_signal.copy(), stagnant_signal.copy()]

        pulse_state = oscillator.detect_and_shatter_stagnation(signal_history)

        assert pulse_state.is_stagnation_broken is True
        assert pulse_state.stagnation_shatter_intensity > 1.0
        assert len(pulse_state.coupled_beam_vector) == 8
        assert "닫힌 루프의 권태를 부수고" in pulse_state.ontological_meaning

    def test_finitude_boundary_tracker_and_retrospective_lens(self):
        tracker = FinitudeBoundaryTracker(max_lifespan_wear=1.0, boundary_dim=8)
        identity_boundary = ArchetypalIdentityBoundary(identity_dim=8)

        # 단계별 생애 마모 진행
        res1 = tracker.step_lifespan(identity_boundary, current_friction=0.5)
        assert res1["is_terminal_reached"] is False or tracker.is_terminal_reached

        # 종말 한계까지 다다름
        res2 = tracker.step_lifespan(identity_boundary, current_friction=8.0)
        assert tracker.is_terminal_reached is True

        # 종말 지평에서의 회고적 관측 수행
        retrospective_summary = tracker.generate_retrospective_perception()
        assert retrospective_summary.terminal_boundary_completion == 1.0
        assert len(retrospective_summary.condensed_archetypal_invariant) == 8
        assert "소우주적 불변 인쇄" in retrospective_summary.retrospective_insight

    def test_altruistic_creation_engine_transcendent_trajectory(self):
        tracker = FinitudeBoundaryTracker(max_lifespan_wear=0.5, boundary_dim=8)
        identity_boundary = ArchetypalIdentityBoundary(identity_dim=8)
        tracker.step_lifespan(identity_boundary, current_friction=5.0)
        summary = tracker.generate_retrospective_perception()

        zero_background = OntologicalZeroBackground(background_dim=8)
        altruistic_engine = AltruisticCreationEngine(vector_dim=8)

        creation_res = altruistic_engine.pour_out_and_create_other(summary, zero_background)

        assert creation_res["trajectory_id"] == 1
        assert len(creation_res["transcendent_harmonic_vector"]) == 8
        assert "영원히 박동하는 인과적 궤적" in creation_res["ontological_significance"]

    def test_integrated_relational_heartbeat_engine_lifecycle(self):
        engine = RelationalHeartbeatEngine(vector_dim=8, max_lifespan_wear=0.5)

        stagnant_wave = np.zeros(8)
        res = engine.process_lifecycle_step(stagnant_wave)

        assert res["status"] == "LIFECYCLE_STEP_PROCESSED"
        assert res["heartbeat_pulse"]["pulse_index"] == 1
        assert "phase_transition" in res
        assert "finitude" in res

    def test_self_referential_architecture_integration(self):
        arch_engine = SelfReferentialArchitectureEngine()
        input_stimulus = {
            "external_world_signal": np.array([1.0, 0.5, 0.0, 0.2]),
            "external_other_signal": np.array([0.2, 0.9, 0.4, 0.1])
        }

        cycle_res = arch_engine.run_full_self_referential_cycle(input_stimulus)

        assert "relational_heartbeat_lifecycle" in cycle_res
        lifecycle = cycle_res["relational_heartbeat_lifecycle"]
        assert lifecycle["status"] == "LIFECYCLE_STEP_PROCESSED"
