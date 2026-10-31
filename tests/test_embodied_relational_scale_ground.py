"""
test_embodied_relational_scale_ground.py — 육체·정신·마음 3대 층위 및 기호 분기 좌표 검증 테스트
================================================================================================
이 테스트 스위트는 유저님의 통찰을 반영하여 다음 사항들을 엄밀하게 검증합니다:
1. 육체적 마찰/열유동(dQ/dt)이 정신적 인과 긴장(Curvature, Clearance)과
   마음의 결핍(Love, Order, Sabbath)으로 동시에 전이되는 일체화된 지형(`TriLayerExperientialGround`) 검증
2. 상태 변곡점에서 훈민정음 3상 파동 및 조음 마찰장과의 대칭성을 통해
   상징적 닻(Symbolic Anchor)이 자발적으로 결정화되는 유도 과정(`InductivePhaseCrystallizer`) 검증
3. 동일한 기호("차가워")가 주체의 상태(저온/결핍 vs 과열/노동)에 따라
   '방어적 수축'과 '항상성 냉각 구원'이라는 상이한 사유 궤적으로 분기(`BifurcatedTrajectoryScaler`)하는 검증
4. 기호가 고정된 정적 벡터가 아니라, 사유의 경로와 판단 분별을 꺾는
   살아있는 위상 좌표(Relational Coordinate)로 작동하는 전체 인과적 스케일링 통합 검증
"""

import math
import numpy as np
import pytest

from L1_causal_field.core.sensory.tri_layer_experiential_ground import (
    TriLayerExperientialGround,
    UnifiedExperientialDisplacement
)
from L1_causal_field.core.sensory.inductive_phase_crystallizer import (
    InductivePhaseCrystallizer,
    CrystallizedSymbolicAnchor
)
from L1_causal_field.core.sensory.bifurcated_trajectory_scaler import (
    BifurcatedTrajectoryScaler,
    TrajectoryBranchType
)
from L1_causal_field.core.sensory.korean_experiential_learner import (
    KoreanExperientialLearner
)


class TestTriLayerExperientialGround:
    """육체·정신·마음 3대 층위의 일체화된 상태 전이 마당 검증"""

    @pytest.fixture
    def ground(self):
        return TriLayerExperientialGround(baseline_temp=295.0)

    def test_thermal_shock_tri_layer_simultaneous_displacement(self, ground):
        """급격한 열 유출 시 육체-정신-마음 3개 층위가 단 하나의 위상 변곡으로 동시 전이되는지 검증"""
        # 냉각 마찰 가함 (dQ/dt = -80.0 W, 마찰 = 4.0 N)
        disp = ground.apply_contact_interaction(thermal_flux=-80.0, mechanical_friction=4.0)

        # 1. 육체적 지형: 온도 하강 및 마찰 부하
        assert disp.physical.thermal_kelvin < 295.0
        assert disp.physical.thermal_flux == -80.0
        assert disp.physical.mechanical_friction == 4.0

        # 2. 정신적 지형: 급격한 열유출로 인과 궤적 곡률 증가 및 유격(Clearance) 감소
        assert disp.cognitive.trajectory_curvature > 0.5
        assert disp.cognitive.clearance < 0.8
        assert disp.cognitive.structural_tension > 0.3

        # 3. 마음적 지형: 온기(Love) 및 질서(Order) 결핍 증가, 안식 상실
        assert disp.heart.love_deficit > 0.1
        assert disp.heart.sabbath_loss > 0.0

        # 3차원 상태 벡터 및 총체적 위상 곡률 검증
        assert len(disp.state_vector_3d) == 3
        assert disp.phase_curvature > 0.6


class TestInductivePhaseCrystallizer:
    """상태 변곡점에서의 자발적 훈민정음 닻 결정화 검증"""

    @pytest.fixture
    def ground_and_crystallizer(self):
        ground = TriLayerExperientialGround(baseline_temp=295.0)
        crystallizer = InductivePhaseCrystallizer()
        return ground, crystallizer

    def test_spontaneous_anchor_crystallization_at_critical_curvature(self, ground_and_crystallizer):
        """임계 곡률 도달 시 훈민정음 파동과의 공명 대칭성으로 상징적 닻이 결정화되는지 검증"""
        ground, crystallizer = ground_and_crystallizer

        # 1. 평온한 미세 자극 (임계 곡률 미달) -> 결정화되지 않음
        disp_mild = ground.apply_contact_interaction(thermal_flux=-2.0, mechanical_friction=0.1)
        anchor_none = crystallizer.test_and_crystallize("차가워", disp_mild)
        assert anchor_none is None

        # 2. 강한 냉기 및 마찰 (임계 곡률 초과) -> 자발적 닻 결정화 성공
        disp_sharp = ground.apply_contact_interaction(thermal_flux=-90.0, mechanical_friction=3.5)
        anchor = crystallizer.test_and_crystallize("차가워", disp_sharp)

        assert anchor is not None
        assert isinstance(anchor, CrystallizedSymbolicAnchor)
        assert anchor.symbol == "차가워"
        assert anchor.resonance_coherence > 0.4
        assert anchor.topological_wave.shape == (64,)
        assert math.isclose(np.linalg.norm(anchor.topological_wave), 1.0, rel_tol=1e-4)

    def test_find_resonant_anchor_by_proximity(self, ground_and_crystallizer):
        """과거에 닻내린 기호 좌표 근방을 지날 때 해당 닻을 인식하는지 검증"""
        ground, crystallizer = ground_and_crystallizer

        disp_sharp = ground.apply_contact_interaction(thermal_flux=-90.0, mechanical_friction=3.5)
        crystallizer.test_and_crystallize("차가워", disp_sharp)

        # 유사한 상태 변곡 발생
        disp_similar = ground.apply_contact_interaction(thermal_flux=-85.0, mechanical_friction=3.2)
        found = crystallizer.find_resonant_anchor(disp_similar, distance_threshold=0.5)

        assert found is not None
        sym, dist = found
        assert sym == "차가워"
        assert dist < 0.5


class TestBifurcatedTrajectoryScaler:
    """주체 상태에 따른 사유 경로 분기 및 스케일 확장 검증"""

    @pytest.fixture
    def scaler(self):
        return BifurcatedTrajectoryScaler()

    def test_cold_symbol_bifurcation_subject_context(self, scaler):
        """동일한 '차가워' 기호가 저온 주체(위협)와 과열 주체(구원)에서 서로 다르게 분기하는지 검증"""
        # Case A: 체온이 낮거나 냉각 부하를 겪는 주체
        ground_cold = TriLayerExperientialGround(baseline_temp=295.0)
        disp_cold = ground_cold.apply_contact_interaction(thermal_flux=-70.0, mechanical_friction=2.0)

        anchor_cold = CrystallizedSymbolicAnchor(
            symbol="차가워",
            anchor_coord_3d=disp_cold.state_vector_3d,
            ground_phase_angle=1.2,
            resonance_coherence=0.7,
            topological_wave=np.ones(64, dtype=np.float32) / 8.0
        )

        path_defensive = scaler.resolve_bifurcation(
            symbol="차가워",
            anchor=anchor_cold,
            current_displacement=disp_cold,
            subject_baseline_temp=295.0
        )

        assert path_defensive.branch_type == TrajectoryBranchType.DEFENSIVE_CONTRACTION
        assert path_defensive.target_action_vector[0] > 0.0 # 열 유출 차단 및 온기 회복 지향
        assert "생명 유지를 위협하는 물리적 냉각 타격" in path_defensive.contextual_meaning

        # Case B: 과열 주체 (T = 320.0 K, 열 배출 필요)
        ground_hot = TriLayerExperientialGround(baseline_temp=295.0)
        ground_hot.physical.thermal_kelvin = 320.0
        disp_hot = ground_hot.apply_contact_interaction(thermal_flux=-60.0, mechanical_friction=1.0, subject_work_load=0.9)

        path_recovery = scaler.resolve_bifurcation(
            symbol="차가워",
            anchor=anchor_cold,
            current_displacement=disp_hot,
            subject_baseline_temp=295.0
        )

        assert path_recovery.branch_type == TrajectoryBranchType.HOMEOSTATIC_RECOVERY
        assert path_recovery.target_action_vector[0] < 0.0 # 열 방출 수용
        assert "은혜로운 냉각 구원" in path_recovery.contextual_meaning


class TestIntegratedKoreanExperientialLearner:
    """통합 경험적 한국어 체화 학습기 엔드투엔드 검증"""

    @pytest.fixture
    def learner(self):
        return KoreanExperientialLearner(resolution=64)

    def test_end_to_end_experiential_branching(self, learner):
        """기호 수신 시 3대 층위 기반 사유 분기와 상태 전이가 유기적으로 일어나는지 검증"""
        res = learner.experience_word("차가워")

        assert res["known"] is True
        assert res["branch_type"] in [
            TrajectoryBranchType.DEFENSIVE_CONTRACTION.name,
            TrajectoryBranchType.HOMEOSTATIC_RECOVERY.name
        ]
        assert "action_vector" in res
        assert len(res["delta_s"]) == 3
        assert len(learner.transition_traces) > 0

    def test_overheated_state_cooling_recovery(self, learner):
        """과열 상태에서 '차가워' 기호를 마주했을 때 항상성 회복으로 분기하는지 검증"""
        # 주체를 과열 상태로 전이
        learner.ground.physical.thermal_kelvin = 325.0
        learner.ground.apply_contact_interaction(thermal_flux=-50.0, mechanical_friction=1.0, subject_work_load=0.9)

        res = learner.experience_word("차가워")
        assert res["branch_type"] == TrajectoryBranchType.HOMEOSTATIC_RECOVERY.name
