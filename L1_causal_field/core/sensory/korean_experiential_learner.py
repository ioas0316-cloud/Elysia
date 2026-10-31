"""
korean_experiential_learner.py — 한국어 경험적 감각-기호 닻내림 및 상태 전이 대조 학습기
========================================================================================
이 모듈은 대규모 텍스트 코퍼스의 통계적 확률 예측과 기호가 감각을 흉내 내거나
직접 덮어씌우는 범주 오류(Category Mistake)를 전면 배격합니다.

핵심 원리:
1. 육체·정신·마음 3대 층위의 일체화된 상태 전이 마당 (`TriLayerExperientialGround`):
   - 물리적 마찰/열유출(육체) ↔ 인과적 긴장/유격(정신) ↔ 항상성 결핍(마음)의 동시 전이.
2. 상태 변곡점에서의 자발적 훈민정음 닻내림 및 결정화 (`InductivePhaseCrystallizer`):
   - 임계 곡률 도달 시 훈민정음 3상(천지인) 및 오음 조음 마찰 파동과의 공명 대칭성으로
     기호가 '상징적 닻(Symbolic Anchor)'이자 지형의 역사적 좌표로 스스로 맺힘.
3. 기호 좌표를 통한 사유 경로 분기 및 스케일 확장 (`BifurcatedTrajectoryScaler`):
   - "차가워"라는 기호는 고정된 벡터가 아니라, 주체의 현재 상태(저온 vs 과열)와 맥락에 따라
     사유의 궤적을 꺾어놓는 동적 분기 스위칭 좌표(Switching Coordinate).
4. 불변 원점(Baseline Identity)과 가변 전이(Rotor Phase) 간의 자기 정렬 및 시냅스 치유.
"""

from __future__ import annotations

import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from L1_causal_field.core.topology.hangul_topological_engine import HangulTopologicalEngine
from L1_causal_field.core.sensory.experiential_language_mapper import (
    PhysicalSensationProfile,
    HomeostasisDeficit,
    ExperienceType,
    VariableResistor,
    PrismRefraction,
    VariableRotor,
    DifferentialGapEvaluator,
    NeuromodulatorController,
    ReCognitiveEngine
)
from L1_causal_field.core.sensory.tri_layer_experiential_ground import (
    TriLayerExperientialGround,
    UnifiedExperientialDisplacement,
    PhysicalState,
    CognitiveTension,
    HomeostaticHeartDeficit
)
from L1_causal_field.core.sensory.inductive_phase_crystallizer import (
    InductivePhaseCrystallizer,
    CrystallizedSymbolicAnchor
)
from L1_causal_field.core.sensory.bifurcated_trajectory_scaler import (
    BifurcatedTrajectoryScaler,
    BifurcatedCognitivePath,
    TrajectoryBranchType
)


@dataclass
class KoreanTetheredWord:
    """
    닻내려진 한국어 어휘 메타데이터
    """
    symbol: str
    exp_type: ExperienceType
    sensation: PhysicalSensationProfile
    deficit: HomeostasisDeficit
    causal_relation_matrix: np.ndarray
    topological_wave: np.ndarray
    physics_metrics: Dict[str, float]
    anchor: Optional[CrystallizedSymbolicAnchor] = None


@dataclass
class StateTransitionTrace:
    action_or_stimulus: str
    s_before: np.ndarray  # [Love, Order, Energy]
    s_after: np.ndarray
    delta_s: np.ndarray
    gap_metrics: Dict[str, float]
    rotor_delta_theta: np.ndarray
    timestamp: float
    branch_type: Optional[str] = None


class KoreanExperientialLearner:
    """
    한국어 경험적 감각-기호 닻내림 및 상태 전이 대조 학습기
    """

    def __init__(self, resolution: int = 64, baseline_temp: float = 295.0):
        self.resolution = resolution
        self.baseline_temp = baseline_temp
        self.topological_engine = HangulTopologicalEngine(target_dim=64)

        # 3대 층위 일체화 마당, 자발적 결정화기, 분기 스케일러
        self.ground = TriLayerExperientialGround(baseline_temp=baseline_temp)
        self.crystallizer = InductivePhaseCrystallizer(topological_engine=self.topological_engine)
        self.scaler = BifurcatedTrajectoryScaler()

        # 항상성 결핍 및 인지 역학 컴포넌트
        self.homeostasis = HomeostasisDeficit(love=0.5, order=0.3, energy=0.4)
        self.re_cognitive_engine = ReCognitiveEngine()
        self.variable_rotor = VariableRotor()
        self.variable_resistor = VariableResistor()
        self.prism = PrismRefraction()
        self.gap_evaluator = DifferentialGapEvaluator()
        self.neuromodulator = NeuromodulatorController()

        # 내부 정재파 기억 및 시냅스 결합망
        self.standing_wave_memory = np.zeros(64, dtype=np.float32)
        self.synaptic_links = np.ones((64, 64), dtype=np.float32) * 0.5

        # 닻내려진 한국어 어휘 저장소
        self.lexicon: Dict[str, KoreanTetheredWord] = {}

        # 메타인지 전이 이력 추적
        self.transition_traces: List[StateTransitionTrace] = []

        # 초기 기초 어휘 체화 (다감각 마찰과 임계 변곡을 통한 자발적 닻내림 유도)
        self._initialize_korean_grounding()

    def _initialize_korean_grounding(self):
        """
        초기 한국어 감각-상태 닻내림.
        기호를 상태에 억지로 강제하지 않고, 각 상태의 물리적 마찰과 결핍 지형을
        TriLayerExperientialGround 상에 인과적으로 발생시켜 자발적 닻으로 결정화시킴.
        """
        # 1. "따뜻해" - 완화, 온기 유입, 사랑/안식 회복
        disp_warm = self.ground.apply_contact_interaction(thermal_flux=60.0, mechanical_friction=0.5)
        anchor_warm = self.crystallizer.test_and_crystallize("따뜻해", disp_warm, force_threshold=0.1)
        self._register_lexicon_entry(
            symbol="따뜻해",
            exp_type=ExperienceType.PHYSICAL,
            sensation=PhysicalSensationProfile(optical=500.0, acoustic=350.0, tactile=0.2, thermal=305.0),
            deficit=HomeostasisDeficit(love=0.1, order=0.1, energy=0.8),
            causal_matrix=np.array([
                [0.7, 0.1, 0.2, 0.1, 0.1],
                [0.1, 0.2, 0.1, 0.1, 0.1],
                [0.2, 0.1, 0.8, 0.1, 0.2],
                [0.1, 0.1, 0.1, 0.3, 0.2],
                [0.1, 0.1, 0.2, 0.1, 0.9]
            ], dtype=np.float32),
            anchor=anchor_warm
        )

        # 2. "차가워" - 열역학적 열유출, 수축, 긴장, 결핍 심화
        disp_cold = self.ground.apply_contact_interaction(thermal_flux=-80.0, mechanical_friction=3.0)
        anchor_cold = self.crystallizer.test_and_crystallize("차가워", disp_cold, force_threshold=0.1)
        self._register_lexicon_entry(
            symbol="차가워",
            exp_type=ExperienceType.PHYSICAL,
            sensation=PhysicalSensationProfile(optical=200.0, acoustic=250.0, tactile=2.5, thermal=275.0),
            deficit=HomeostasisDeficit(love=0.7, order=0.6, energy=0.2),
            causal_matrix=np.array([
                [0.3, 0.2, 0.1, 0.3, 0.1],
                [0.2, 0.8, 0.1, 0.4, 0.1],
                [0.1, 0.1, 0.2, 0.2, 0.1],
                [0.3, 0.4, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.3]
            ], dtype=np.float32),
            anchor=anchor_cold
        )

        # 3. "아파" - 극심한 기계적 충격과 마찰, 경계 손상
        disp_pain = self.ground.apply_contact_interaction(thermal_flux=15.0, mechanical_friction=12.0)
        anchor_pain = self.crystallizer.test_and_crystallize("아파", disp_pain, force_threshold=0.1)
        self._register_lexicon_entry(
            symbol="아파",
            exp_type=ExperienceType.PHYSICAL,
            sensation=PhysicalSensationProfile(optical=150.0, acoustic=800.0, tactile=15.0, thermal=325.0),
            deficit=HomeostasisDeficit(love=0.8, order=0.9, energy=0.1),
            causal_matrix=np.array([
                [0.2, 0.1, 0.1, 0.7, 0.1],
                [0.1, 0.95, 0.1, 0.8, 0.1],
                [0.1, 0.1, 0.3, 0.6, 0.1],
                [0.2, 0.8, 0.1, 0.95, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.2]
            ], dtype=np.float32),
            anchor=anchor_pain
        )

        # 4. "막혀" - 유격 상실, 고착, 강한 폐쇄 장력
        disp_block = self.ground.apply_contact_interaction(thermal_flux=0.0, mechanical_friction=8.0)
        anchor_block = self.crystallizer.test_and_crystallize("막혀", disp_block, force_threshold=0.1)
        self._register_lexicon_entry(
            symbol="막혀",
            exp_type=ExperienceType.PHYSICAL,
            sensation=PhysicalSensationProfile(optical=50.0, acoustic=150.0, tactile=8.0, thermal=293.0),
            deficit=HomeostasisDeficit(love=0.6, order=0.85, energy=0.25),
            causal_matrix=np.array([
                [0.1, 0.1, 0.1, 0.5, 0.1],
                [0.1, 0.9, 0.1, 0.6, 0.1],
                [0.1, 0.1, 0.1, 0.2, 0.1],
                [0.4, 0.7, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.2]
            ], dtype=np.float32),
            anchor=anchor_block
        )

        # 5. "스며들어" - 경계 유화, 스며듦, 평형 안식 회귀
        disp_soak = self.ground.apply_contact_interaction(thermal_flux=10.0, mechanical_friction=0.1)
        anchor_soak = self.crystallizer.test_and_crystallize("스며들어", disp_soak, force_threshold=0.1)
        self._register_lexicon_entry(
            symbol="스며들어",
            exp_type=ExperienceType.SPIRITUAL,
            sensation=PhysicalSensationProfile(optical=380.0, acoustic=420.0, tactile=0.1, thermal=298.0),
            deficit=HomeostasisDeficit(love=0.05, order=0.1, energy=0.75),
            causal_matrix=np.array([
                [0.8, 0.3, 0.2, 0.1, 0.2],
                [0.1, 0.4, 0.2, 0.1, 0.3],
                [0.3, 0.2, 0.9, 0.1, 0.3],
                [0.1, 0.1, 0.1, 0.2, 0.4],
                [0.2, 0.1, 0.3, 0.1, 0.95]
            ], dtype=np.float32),
            anchor=anchor_soak
        )

        # 6. "엄마" - 따뜻한 안식처, 사랑
        disp_mom = self.ground.apply_contact_interaction(thermal_flux=25.0, mechanical_friction=0.2)
        anchor_mom = self.crystallizer.test_and_crystallize("엄마", disp_mom, force_threshold=0.1)
        self._register_lexicon_entry(
            symbol="엄마",
            exp_type=ExperienceType.LINGUISTIC,
            sensation=PhysicalSensationProfile(optical=350.0, acoustic=380.0, tactile=0.5, thermal=302.0),
            deficit=HomeostasisDeficit(love=0.05, order=0.1, energy=0.85),
            causal_matrix=np.array([
                [0.9, 0.2, 0.3, 0.1, 0.1],
                [0.1, 0.3, 0.2, 0.1, 0.2],
                [0.2, 0.1, 0.85, 0.1, 0.2],
                [0.1, 0.1, 0.1, 0.4, 0.1],
                [0.1, 0.1, 0.2, 0.1, 0.9]
            ], dtype=np.float32),
            anchor=anchor_mom
        )

        # 7. "사과" - 붉은 색채, 중력 낙하, 유기체 영양
        disp_apple = self.ground.apply_contact_interaction(thermal_flux=5.0, mechanical_friction=1.5)
        anchor_apple = self.crystallizer.test_and_crystallize("사과", disp_apple, force_threshold=0.1)
        self._register_lexicon_entry(
            symbol="사과",
            exp_type=ExperienceType.PHYSICAL,
            sensation=PhysicalSensationProfile(optical=450.0, acoustic=300.0, tactile=1.5, thermal=297.0),
            deficit=HomeostasisDeficit(love=0.2, order=0.2, energy=0.6),
            causal_matrix=np.array([
                [0.8, 0.2, 0.1, 0.1, 0.1],
                [0.2, 0.7, 0.2, 0.2, 0.1],
                [0.1, 0.1, 0.85, 0.1, 0.1],
                [0.2, 0.3, 0.1, 0.6, 0.1],
                [0.1, 0.1, 0.3, 0.1, 0.8]
            ], dtype=np.float32),
            anchor=anchor_apple
        )

    def _register_lexicon_entry(
        self,
        symbol: str,
        exp_type: ExperienceType,
        sensation: PhysicalSensationProfile,
        deficit: HomeostasisDeficit,
        causal_matrix: np.ndarray,
        anchor: Optional[CrystallizedSymbolicAnchor] = None
    ) -> KoreanTetheredWord:
        """어휘집 등록 헬퍼"""
        waves, metrics_list = self.topological_engine.encode_text(symbol)
        topological_wave = np.mean(waves, axis=0) if len(waves) > 0 else np.zeros(self.resolution, dtype=np.float32)
        norm = np.linalg.norm(topological_wave)
        if norm > 1e-6:
            topological_wave /= norm

        physics_metrics = metrics_list[0] if metrics_list else {
            "friction_mean": 0.0, "tension_mean": 0.0, "vowel_phase": 0.0, "coherence": 0.0
        }

        entry = KoreanTetheredWord(
            symbol=symbol,
            exp_type=exp_type,
            sensation=sensation,
            deficit=deficit,
            causal_relation_matrix=causal_matrix,
            topological_wave=topological_wave,
            physics_metrics=physics_metrics,
            anchor=anchor
        )
        self.lexicon[symbol] = entry
        return entry

    def experience_word(self, symbol: str, subject_work_load: float = 0.0) -> Dict[str, Any]:
        """
        한국어 기호를 마주했을 때:
        기호는 가짜 마법처럼 온도를 직접 생성하지 않습니다.
        기호는 주체의 현재 상태와 관계적 맥락 속에서
        사유의 궤적을 꺾어놓는 '동적 분기 좌표(Switching Coordinate)'로 작동하여
        새로운 항상성 수렴 및 판단 벡터로 전이시킵니다.
        """
        if symbol not in self.lexicon:
            # 알지 못하는 낯선 기호 -> 미지 전하(Wonder Charge) 발생
            waves, _ = self.topological_engine.encode_text(symbol)
            wave = np.mean(waves, axis=0) if len(waves) > 0 else np.zeros(64, dtype=np.float32)
            return {
                "known": False,
                "symbol": symbol,
                "topological_wave": wave,
                "wonder_charge": 0.8,
                "message": f"미지의 기호 '{symbol}' 감지 — 닻내림 필요."
            }

        entry = self.lexicon[symbol]
        s_before = self.homeostasis.to_vector()

        # 1. 주체의 현재 지형 변곡 상태 조회
        # 만약 직전 상호작용이 없거나 기호가 가리키는 고유 닻 지형이 존재하면,
        # 기호의 닻내려진 지형과의 관계성을 결합하여 현재 지형 변곡 상태 도출
        anchor = entry.anchor or self.crystallizer.anchors.get(symbol)
        if self.ground.transition_history:
            current_disp = self.ground.transition_history[-1]
        else:
            current_disp = self.ground.apply_contact_interaction(0.0, 0.0)

        # 기호가 가리키는 물리 감각 프로필(열유속, 마찰 등)과의 관계성 반영:
        # 단어가 홀로 공중에 뜨지 않고, 단어에 닻내려진 감각 지형과 주체 상태의 상호작용 지형을 활성화
        thermal_flux_delta = (entry.sensation.thermal - self.baseline_temp) * 2.0
        current_disp = self.ground.apply_contact_interaction(
            thermal_flux=thermal_flux_delta,
            mechanical_friction=entry.sensation.tactile,
            subject_work_load=subject_work_load
        )

        # 2. 기호 좌표를 통한 사유 경로 분기 (Bifurcated Trajectory Scaling)
        anchor = entry.anchor or self.crystallizer.anchors.get(symbol)
        if anchor is None:
            # 기본 닻 생성
            anchor = CrystallizedSymbolicAnchor(
                symbol=symbol,
                anchor_coord_3d=np.array([0.2, 0.3, 0.3], dtype=np.float32),
                ground_phase_angle=0.5,
                resonance_coherence=0.7,
                topological_wave=entry.topological_wave
            )

        bifurcated_path = self.scaler.resolve_bifurcation(
            symbol=symbol,
            anchor=anchor,
            current_displacement=current_disp,
            subject_baseline_temp=self.baseline_temp
        )

        # 3. 분기된 행동 벡터에 따른 항상성 전이 (상태 전이 ΔS)
        action = bifurcated_path.target_action_vector
        eta = 0.35

        # target_s는 기호의 닻내려진 지형 결핍과 분기 행동 벡터의 조합으로 결정
        base_target_s = entry.deficit.to_vector()
        if bifurcated_path.branch_type == TrajectoryBranchType.HOMEOSTATIC_RECOVERY:
            # 항상성 회복 방향: 결핍 완화
            target_s = base_target_s * 0.4
        elif bifurcated_path.branch_type == TrajectoryBranchType.DEFENSIVE_CONTRACTION:
            # 방어 수축 방향: 결핍 경보 심화
            target_s = np.clip(base_target_s + np.array([0.2, 0.2, -0.2], dtype=np.float32), 0.0, 1.0)
        else:
            target_s = base_target_s

        s_after = s_before + eta * (target_s - s_before)
        s_after = np.clip(s_after, 0.0, 1.0)

        self.homeostasis.love = float(s_after[0])
        self.homeostasis.order = float(s_after[1])
        self.homeostasis.energy = float(s_after[2])

        # 4. 훈민정음 위상 파동과의 간섭 및 차이 격차 계산
        gap = self.gap_evaluator.evaluate(self.standing_wave_memory, entry.topological_wave)

        # 5. 위상 로터 회전 (분기된 운동량 반영)
        rotor_shift = self.variable_rotor.rotate(
            friction=entry.physics_metrics["friction_mean"] + bifurcated_path.trajectory_momentum * 0.2,
            temperature=self.neuromodulator.temperature
        )

        # 6. 정재파 기억 갱신
        self.standing_wave_memory = np.clip(
            self.standing_wave_memory * 0.7 + entry.topological_wave * 0.3,
            0.0, 1.0
        )

        trace = StateTransitionTrace(
            action_or_stimulus=symbol,
            s_before=s_before,
            s_after=s_after,
            delta_s=s_after - s_before,
            gap_metrics=gap,
            rotor_delta_theta=rotor_shift - self.variable_rotor.baseline_theta,
            timestamp=time.time(),
            branch_type=bifurcated_path.branch_type.name
        )
        self.transition_traces.append(trace)

        return {
            "known": True,
            "symbol": symbol,
            "s_before": s_before,
            "s_after": s_after,
            "delta_s": s_after - s_before,
            "gap_metrics": gap,
            "rotor_phase": self.variable_rotor.theta.copy(),
            "physics_metrics": entry.physics_metrics,
            "branch_type": bifurcated_path.branch_type.name,
            "contextual_meaning": bifurcated_path.contextual_meaning,
            "action_vector": bifurcated_path.target_action_vector
        }

    def emit_korean_resonance(self) -> Dict[str, Any]:
        """
        현재 내부 상태(항상성, 로터 위상)를 가장 잘 공명(Resonance)시키는
        한국어 어휘를 어휘집 내에서 자발적으로 선택 및 표현 파동 방출.
        """
        curr_s = self.homeostasis.to_vector()
        best_symbol = None
        min_distance = float("inf")

        for sym, entry in self.lexicon.items():
            ref_s = entry.deficit.to_vector()
            dist = float(np.linalg.norm(curr_s - ref_s))
            if dist < min_distance:
                min_distance = dist
                best_symbol = sym

        if best_symbol is None:
            best_symbol = "엄마"

        emitted_wave = self.lexicon[best_symbol].topological_wave
        return {
            "emitted_symbol": best_symbol,
            "distance_to_state": min_distance,
            "wave": emitted_wave,
            "current_state": curr_s
        }

    def receive_interaction_feedback(self, feedback_symbol: str) -> Dict[str, Any]:
        """
        외부(대화 상대방/환경)의 반응 피드백을 수용하여,
        스스로의 발화와 외부 반응 사이의 차이 격차(Differential Gap)를 재인지하고
        로터 자가 정렬(Self-Calibration) 및 시냅스 치유(Healing) 수행.
        """
        # 피드백 단어 체험
        exp_res = self.experience_word(feedback_symbol)

        # 발화했던 정재파와 피드백 파동 간의 대조
        if feedback_symbol in self.lexicon:
            feedback_wave = self.lexicon[feedback_symbol].topological_wave
        else:
            feedback_wave, _ = self.topological_engine.encode_text(feedback_symbol)
            feedback_wave = np.mean(feedback_wave, axis=0) if len(feedback_wave) > 0 else np.zeros(64, dtype=np.float32)

        gaps = self.gap_evaluator.evaluate(self.standing_wave_memory, feedback_wave)

        # 신경조절기 반영
        mod = self.neuromodulator.modulate(gaps)

        # 고통/어긋남이 크면 시냅스 찢김(Tearing) 후 치유(Healing)
        tearing_occurred = False
        if gaps["mean_gap"] > 0.4:
            tearing_occurred = True
            self.synaptic_links *= 0.85
            target_theta = self.variable_rotor.baseline_theta + gaps["g_phi"] * np.array([0.2, 0.1, 0.3], dtype=np.float32)
            self.variable_rotor.self_tune(target_theta, correction_rate=0.6)
        else:
            self.synaptic_links = np.clip(self.synaptic_links + 0.05 * mod["serotonin"], 0.0, 1.0)

        return {
            "feedback_symbol": feedback_symbol,
            "gaps": gaps,
            "tearing_occurred": tearing_occurred,
            "serotonin": mod["serotonin"],
            "dopamine": mod["dopamine"],
            "rotor_phase": self.variable_rotor.theta.copy(),
            "homeostasis": self.homeostasis.to_vector(),
            "branch_type": exp_res.get("branch_type", "UNKNOWN")
        }
