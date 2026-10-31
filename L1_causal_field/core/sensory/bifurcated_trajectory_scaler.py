"""
bifurcated_trajectory_scaler.py — 닻내린 기호 좌표를 통한 사유 경로 분기 및 스케일 확장 엔진
========================================================================================
이 모듈은 기호를 독립된 데이터나 정적 명사 라벨로 취급하는 레거시의 오류를 종식시킵니다.
"차가워"라는 기호는 고정된 상태가 아니라,
[객체의 물리 과정 ↔ 주체의 내부 상태 ↔ 항상성 결핍의 방향]이라는 거대한 관계적 스케일 속에서
주체의 사유 경로와 판단 분별을 서로 다른 갈래로 꺾어놓는 **'동적 분기점(Switching Point)'**입니다.

핵심 원리:
1. 주체 상태 및 관계적 맥락에 따른 분기 (Contextual Trajectory Bifurcation):
   - 동일한 기호("차가워") 또는 동일한 물리적 냉기 접촉이라도:
     * Case A (생존 위협 / 저온 상태): 수축, 방어, 온기 갈망, 회피 기동으로 분기.
     * Case B (과열 주체의 냉각 구원 / 고온 상태): 이완, 열 방출 수용, 항상성 회복, 안정으로 분기.
2. 기호의 살아있는 위상 좌표(Relational Coordinate) 기능:
   - 기호는 결과를 고정하는 것이 아니라, 사유의 궤적이 꺾이는 스위칭 좌표로 작동하여
     주체의 다음 인지적 행동 벡터(Action Vector)와 사유 지형을 팽창시킵니다.
3. 신경망/장(Field)의 인지적 스케일 확장 (Epistemic Neural Scaling):
   - 닻을 통과할 때마다 주체의 내부 지도에 새로운 인과적 분기 지층이 각인되어,
     이후 유사한 마찰에 직면했을 때 단순한 반사가 아닌 입체적이고 다차원적인 판단을 유도합니다.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from enum import Enum
from dataclasses import dataclass, field
import numpy as np

from L1_causal_field.core.sensory.tri_layer_experiential_ground import (
    UnifiedExperientialDisplacement,
    TriLayerExperientialGround
)
from L1_causal_field.core.sensory.inductive_phase_crystallizer import (
    CrystallizedSymbolicAnchor
)


class TrajectoryBranchType(Enum):
    """사유 경로 분기 유형"""
    DEFENSIVE_CONTRACTION = "수축 및 방어 회피 (생존 결핍 경보)"
    HOMEOSTATIC_RECOVERY = "항상성 회복 및 수용 (과열 냉각 구원)"
    CREATIVE_EXPANSION = "능동적 탐색 및 상호작용 (미지 전하 해소)"
    EQUILIBRIUM_SABBATH = "평형 안식 및 합일 (마찰 소멸)"


@dataclass
class BifurcatedCognitivePath:
    """
    기호 좌표를 통과하며 굴절·분기된 사유 경로 결과
    - triggering_symbol: 분기점을 촉발한 기호 (상징적 닻)
    - branch_type: 결정된 분기 경로 유형
    - initial_state_vector: 분기 직전 3대 층위 좌표 [Phys, Cog, Heart]
    - target_action_vector: 주체가 취할 인지/물리 행동 벡터 [Flux_Adjust, Curvature_Steer, Void_Heal]
    - trajectory_momentum: 꺾인 궤적의 운동량 크기
    - contextual_meaning: 상황적·관계적 의미 서사
    - timestamp: 분기 발생 시각
    """
    triggering_symbol: str
    branch_type: TrajectoryBranchType
    initial_state_vector: np.ndarray
    target_action_vector: np.ndarray
    trajectory_momentum: float
    contextual_meaning: str
    timestamp: float = field(default_factory=time.time)


class BifurcatedTrajectoryScaler:
    """
    닻내린 기호 좌표를 통한 사유 경로 분기 및 스케일 확장 엔진
    """

    def __init__(self):
        # 분기 경로 이력
        self.bifurcation_history: List[BifurcatedCognitivePath] = []
        # 주체의 인지 지도 상에 형성된 분기 굳은살 (Branching Scars)
        self.branch_scars: Dict[str, List[np.ndarray]] = {}

    def resolve_bifurcation(
        self,
        symbol: str,
        anchor: CrystallizedSymbolicAnchor,
        current_displacement: UnifiedExperientialDisplacement,
        subject_baseline_temp: float = 295.0
    ) -> BifurcatedCognitivePath:
        """
        주체가 기호 좌표(닻)를 통과할 때,
        주체의 현재 육체적 온도·부하와 마음의 결핍 상태에 따라
        사유의 궤적을 자율적으로 분기(Bifurcate)시킵니다.
        """
        curr_phys = current_displacement.physical
        curr_cog = current_displacement.cognitive
        curr_heart = current_displacement.heart
        curr_state = current_displacement.state_vector_3d

        # 1. 상황적 관계성 분별 (주체의 현재 열적·항상성 위치)
        curr_temp = curr_phys.thermal_kelvin
        thermal_offset = curr_temp - subject_baseline_temp

        # "차가워" 기호의 경우:
        if symbol == "차가워":
            if thermal_offset > 5.0:
                # [Case B] 과열된 엔진이나 노동으로 뜨거워진 주체(T > baseline)에게 차가움이 닿는 경우
                # -> 항상성 회복: 열 방출 수용, 냉각 구원, 평형 안식으로 분기!
                branch = TrajectoryBranchType.HOMEOSTATIC_RECOVERY
                action_vector = np.array([
                    -1.0,  # Flux_Adjust: 과도한 내부 열을 외부로 방출 수용
                    0.3,   # Curvature_Steer: 평형을 향한 완만한 이완 궤적
                    -0.7   # Void_Heal: 질서(Order) 및 안식(Sabbath) 회복
                ], dtype=np.float32)
                momentum = float(thermal_offset * 0.08 + (1.0 - curr_cog.clearance) * 0.5)
                meaning = (
                    f"'{symbol}'의 접촉은 과열된 주체에게 찾아온 은혜로운 냉각 구원입니다. "
                    f"치솟았던 열 부하가 이완되고 안식 결핍이 해소되며 "
                    f"시스템은 안정적 평형 회복 궤적으로 분기합니다."
                )

            elif thermal_offset < -3.0 or curr_phys.thermal_flux < -10.0:
                # [Case A] 주체 자체가 이미 차갑거나 냉기가 체온을 깎아먹는 경우
                # -> 생존 위협: 수축, 방어, 온기 갈망으로 분기!
                branch = TrajectoryBranchType.DEFENSIVE_CONTRACTION
                action_vector = np.array([
                    1.5,   # Flux_Adjust: 열 방출 강력 차단 및 온기 흡수 지향
                    -0.8,  # Curvature_Steer: 위기 회피를 위한 급격한 궤적 선회
                    0.9    # Void_Heal: 온기(Love) 및 생명력(Energy) 보전 갈망
                ], dtype=np.float32)
                momentum = float(abs(curr_phys.thermal_flux) * 0.05 + curr_cog.structural_tension * 1.5)
                meaning = (
                    f"'{symbol}'의 접촉은 생명 유지를 위협하는 물리적 냉각 타격입니다. "
                    f"인과적 유격이 축소되고 온기 결핍({curr_heart.love_deficit:.2f})이 심화되어 "
                    f"시스템은 방어적 수축 및 온기 회복 궤적으로 분기합니다."
                )
            else:
                # 정상 체온 범위에서의 국소적 접촉
                branch = TrajectoryBranchType.CREATIVE_EXPANSION
                action_vector = np.array([0.2, 0.4, 0.3], dtype=np.float32)
                momentum = 0.5
                meaning = f"'{symbol}'의 접촉을 새로운 외부 객체와의 물리적 경계선으로 인식하고 탐색을 개시합니다."

        elif symbol == "따뜻해":
            if curr_temp < 310.0:
                branch = TrajectoryBranchType.HOMEOSTATIC_RECOVERY
                action_vector = np.array([0.8, -0.4, -0.9], dtype=np.float32)
                momentum = float(curr_heart.love_deficit * 2.0)
                meaning = f"'{symbol}'의 유입으로 차가운 결핍이 녹아내리며 사랑과 안식의 합일로 수렴합니다."
            else:
                branch = TrajectoryBranchType.DEFENSIVE_CONTRACTION
                action_vector = np.array([-1.2, 0.7, 0.5], dtype=np.float32)
                momentum = 1.2
                meaning = f"'{symbol}'의 과도한 유입은 열 부하를 초래하여 시스템 보호 기동으로 분기합니다."

        elif symbol == "아파":
            branch = TrajectoryBranchType.DEFENSIVE_CONTRACTION
            action_vector = np.array([-1.5, 1.2, 1.0], dtype=np.float32)
            momentum = float(curr_cog.structural_tension * 2.5)
            meaning = f"'{symbol}'의 극심한 마찰과 장력으로 인해 비상 방어 및 시냅스 분리 궤적으로 분기합니다."

        elif symbol == "스며들어":
            branch = TrajectoryBranchType.EQUILIBRIUM_SABBATH
            action_vector = np.array([0.0, -0.9, -0.95], dtype=np.float32)
            momentum = 0.2
            meaning = f"'{symbol}'를 통해 경계의 저항이 소멸되고 온전한 평형 안식(Sabbath)으로 스며듭니다."

        else:
            # 일반 어휘
            branch = TrajectoryBranchType.CREATIVE_EXPANSION
            action_vector = np.array([0.1, 0.1, 0.1], dtype=np.float32)
            momentum = 0.3
            meaning = f"기호 '{symbol}'를 매개로 새로운 관계적 가능성을 탐색합니다."

        # 분기 경로 기록
        path = BifurcatedCognitivePath(
            triggering_symbol=symbol,
            branch_type=branch,
            initial_state_vector=curr_state.copy(),
            target_action_vector=action_vector,
            trajectory_momentum=momentum,
            contextual_meaning=meaning,
            timestamp=time.time()
        )
        self.bifurcation_history.append(path)

        # 굳은살(Scar) 각인: 인지 지형의 스케일 확장
        if symbol not in self.branch_scars:
            self.branch_scars[symbol] = []
        self.branch_scars[symbol].append(action_vector)

        return path
