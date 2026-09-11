"""
Unit tests for Teleological Hierarchy & Dimensional Leap Engine
===================================================================
"""

import pytest
import numpy as np

from core.consciousness.teleological_hierarchy import (
    TeleologicalHierarchyEngine,
    PurposeStatus,
    MacroPurpose,
    SubGoalNode
)


class TestTeleologicalHierarchyEngine:

    def setup_method(self):
        self.engine = TeleologicalHierarchyEngine(
            dimension=8,
            friction_critical_threshold=0.5,
            leap_threshold=0.65
        )

    def test_initialization(self):
        state = self.engine.get_hierarchy_state()
        assert state["dimension"] == 8
        assert state["total_macro_purposes"] == 1
        assert state["total_sub_goals"] == 0
        assert state["active_macro_purpose"] == "macro_root_causal_truth"

    def test_register_subgoal(self):
        target = np.ones(8, dtype=np.float32) * 0.5
        subgoal = self.engine.register_subgoal(
            subgoal_id="sub1",
            description="Initial Subgoal Test",
            target_state=target
        )
        assert subgoal.id == "sub1"
        assert subgoal.status == PurposeStatus.SUBGOAL_ACTIVE
        assert self.engine.get_hierarchy_state()["total_sub_goals"] == 1

    def test_execute_substrate_step_and_wave_observation(self):
        target = np.ones(8, dtype=np.float32) * 0.5
        self.engine.register_subgoal("sub1", "Step Test", target)

        action = np.ones(8, dtype=np.float32) * 0.1
        perturbation = np.array([0.02] * 8, dtype=np.float32)
        res = self.engine.execute_substrate_step_and_observe("sub1", action, perturbation)

        assert res["step"] == 1
        assert "distance_to_target" in res
        assert "friction_coefficient" in res
        assert "wave_observation" in res
        assert res["wave_observation"]["phase_delta_mean"] >= 0.0

    def test_teleological_relabeling_and_horizon_expansion(self):
        """특정 목표 도달 시 결코 Terminate하지 않고, Sub-goal로 재라벨링 및 상위 Macro-Purpose 지평이 무한 확장되는지 검증."""
        target = np.ones(8, dtype=np.float32) * 0.2
        self.engine.register_subgoal("sub_target_reach", "Reach Target Subgoal", target)

        # Action을 크게 주어 target에 도달시킴 (distance < 0.2)
        action = np.ones(8, dtype=np.float32) * 0.19
        res = self.engine.execute_substrate_step_and_observe("sub_target_reach", action)

        assert res["escalation_event"] is not None
        esc = res["escalation_event"]
        assert esc["relabeled_subgoal_id"] == "sub_target_reach"
        assert esc["new_status"] == PurposeStatus.RELABELED_AS_TERRAIN.value
        assert "spawned_next_subgoal_id" in esc

        # 상위 Macro-Purpose 지평이 무한 확장되었는지 확인
        state = self.engine.get_hierarchy_state()
        assert state["total_macro_purposes"] > 1
        assert state["total_sub_goals"] > 1

    def test_dimensional_leap_on_high_stress_and_subsequent_steps(self):
        """하위 차원의 마찰 및 사각지대 한계 직면 시, 새로운 관조의 축(Axis D -> D+1)이 자발적으로 발아하고 후속 단계가 정상 구동되는지 검증."""
        target = np.ones(8, dtype=np.float32) * 10.0
        self.engine.register_subgoal("sub_high_stress", "High Stress Target", target)

        # 아주 큰 마찰과 섭동 부여
        action = np.ones(8, dtype=np.float32) * 2.0
        perturbation = np.ones(8, dtype=np.float32) * 1.5

        res = self.engine.execute_substrate_step_and_observe("sub_high_stress", action, perturbation)

        # 차원 도약 발생 여부 검증
        if res["leaped"]:
            assert res["leap_event"]["new_dimension"] == 9
            assert self.engine.dimension == 9
            assert self.engine.leap_module.sprouted_axes_count == 1
            assert self.engine.wave_observer.dimension == 9

            # 차원 도약 이후 후속 Step 실행 검증 (새로운 차원 9에 맞춰 섭동 관측이 오류 없이 작동하는지 확인)
            action_post_leap = np.ones(9, dtype=np.float32) * 0.5
            perturbation_post_leap = np.ones(9, dtype=np.float32) * 0.1
            res_post = self.engine.execute_substrate_step_and_observe("sub_high_stress", action_post_leap, perturbation_post_leap)
            assert res_post["step"] == 2
            assert res_post["wave_observation"]["is_real_time_observed"] is True
