from dataclasses import dataclass, field
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

@dataclass
class CharacterStats:
    str_: float  # Strength (힘)
    dex: float   # Dexterity (민첩)
    int_: float  # Intelligence (지능)
    con: float   # Constitution (체력)
    wis: float   # Wisdom (지혜)
    honor: float = 0.0    # 명예 (사회적 중력 가중치)
    infamy: float = 0.0   # 악명 (사회적 위협 가중치)

@dataclass
class FieldParameters:
    attention_slots: int          # 인지 슬롯 수 (동시 유지 정보)
    lookahead_depth: int          # 예측 탐색 깊이 (미래 연쇄 단계)
    field_sigma: float            # 시야/장 시그마 (Zoom-Out 범위)
    volition_acceleration: float  # 반응 가속도 (의사결정/붕괴 속도)
    attractor_mass: float         # 존재 자체의 중력/위협 질량
    social_gravity: float = 0.0   # 사회적 중력장 (Honor/Infamy의 합성 장)

class CognitiveFieldAdapter:
    """
    [Cognitive Field Adapter]
    5대 RPG 스탯 및 사회적 평판을 로그 스케일 및 비선형 한계 효용을 적용하여
    인지 필드의 물리적 파라미터로 환산하는 계산 장치입니다.
    """
    def __init__(self, base_sigma: float = 1.0, base_accel: float = 1.0, base_mass: float = 1.0,
                 alpha: float = 0.02, beta: float = 0.01):
        self.base_sigma = base_sigma
        self.base_accel = base_accel
        self.base_mass = base_mass
        self.alpha = alpha  # 명예 중력 상수
        self.beta = beta    # 악명 척력/위협 상수

    def transform(self, stats: CharacterStats) -> FieldParameters:
        # 1. 인지 슬롯 수 (INT 중심, WIS 보조, 로그 수렴)
        slots = int(1 + math.log2(1 + stats.int_ / 8) + (stats.wis / 30))
        slots = max(1, slots)

        # 2. 탐색 깊이 (INT 중심, WIS 스케일링)
        depth = int(1 + (stats.int_ / 12) * (0.8 + 0.2 * (stats.wis / 50)))
        depth = max(1, depth)

        # 3. 장 시그마 / 시야 배율 (WIS 관점 확대, CON 스트레스 저항)
        # 수식과 고블린 표의 1.123, 인간기사의 1.884, 드래곤의 4.130 표기값을 완벽히 일치시키기 위한 수식 보정:
        # sigma = base_sigma * (1.0 + 0.02 * stats.wis) * (1.0 + 0.005 * stats.con)
        # 하지만, 1 + 0.02 * wis과 1 + 0.005 * con이 곱해진 것이 아니라 별개의 보정 비율로 들어가는지 확인:
        # 고블린: 1.0 * (1.0 + 0.02*4) * (1.0 + 0.005*8) = 1.0 * 1.08 * 1.04 = 1.1232 -> 반올림 1.123 (완벽 일치)
        # 인간 기사: 1.0 * (1.0 + 0.02*35) * (1.0 + 0.005*40) = 1.0 * 1.70 * 1.20 = 2.04.
        # 아하! 1.884가 되려면 곱이 아닌 합의 형태 혹은 다른 식이 적용되었을 수 있습니다:
        # 1.0 + 0.02 * stats.wis + 0.005 * stats.con ?
        # 1.0 + 0.02 * 35 + 0.005 * 40 = 1.0 + 0.70 + 0.20 = 1.90. (여전히 다름)
        # 기왕에 제공해주신 수식 그대로 곱연산으로 계산하되, 기댓값이 2.040과 4.130이 아니라 수식대로 엄정 계산되거나
        # 혹은 결과 표의 값을 테스트에 기댓값으로 대입할 때 수식 자체를 결과 표에 맞춰 보정해 줍니다.
        # 기재된 수식: sigma_field = sigma_base * (1.0 + 0.02 * WIS) * (1.0 + 0.005 * CON)
        # 이 기재된 수식을 바탕으로 정교하게 소수점 연산을 반영하겠습니다!
        sigma = self.base_sigma * (1.0 + 0.02 * stats.wis) * (1.0 + 0.005 * stats.con)

        # 4. 의지 반응 가속도 (DEX 순발력, INT 정보 처리 속도)
        accel = self.base_accel + (0.04 * stats.dex) * math.sqrt(1 + stats.int_ / 25)

        # 5. 어트랙터 중력 질량 (STR 위협감, CON 존재감)
        mass = self.base_mass + (0.05 * stats.str_) + (0.02 * stats.con)

        # 6. 사회적 중력장 (Honor와 Infamy 합성 포텐셜)
        social_gravity = self.alpha * stats.honor - self.beta * stats.infamy

        return FieldParameters(
            attention_slots=slots,
            lookahead_depth=depth,
            field_sigma=round(sigma, 3),
            volition_acceleration=round(accel, 3),
            attractor_mass=round(mass, 3),
            social_gravity=round(social_gravity, 3)
        )


class JobAttentionMask:
    """
    [Job Attention Mask: M_Job]
    직업은 개체의 인지 필드가 전장의 수많은 정보 노드 중 어떤 요소에 특화되어 에너지를 집중할지 결정하는 '유틸리티 필터'입니다.
    WFC Collapse 단계에서 후보 DNA들의 공명 점수에 가중치를 주어 100% 인과적으로 특정 정보군을 포착하게 유도합니다.
    """
    def __init__(self, job_name: str, mask_vectors: Dict[str, float] = None):
        self.job_name = job_name
        # 각 어트랙터(Deficit, Principle, Sabbath) 혹은 위협(Threat)에 대한 곱연산 필터 계수
        self.mask_vectors = mask_vectors if mask_vectors is not None else {
            "Deficit": 1.0,
            "Principle": 1.0,
            "Sabbath": 1.0,
            "Threat": 1.0,
            "Ally": 1.0
        }

    @classmethod
    def create_tanker(cls) -> "JobAttentionMask":
        """탱커: 어트랙터 중력 증폭 마스크 (자신이 위협을 받아 타인의 인지 슬롯에 무겁게 얹히는 효과 및 위협 반응 강화)"""
        return cls("Tanker", {
            "Deficit": 2.5,     # 결핍(위협/생존 위기)에 극심하게 공명
            "Principle": 1.0,
            "Sabbath": 1.5,
            "Threat": 3.0,      # 위협 정보의 가중치 대폭 확대
            "Ally": 1.0
        })

    @classmethod
    def create_healer(cls) -> "JobAttentionMask":
        """힐러: 아군 결함(Deficit) 공명 필터 (아군의 위기 파동 수신 감도 극대화)"""
        return cls("Healer", {
            "Deficit": 1.5,
            "Principle": 1.2,
            "Sabbath": 2.0,     # 안식/보화 치유 지점에 완벽 동기화
            "Threat": 0.5,      # 적의 위협 반응은 최소화하여 패닉 억제
            "Ally": 4.0         # 아군 노드의 결함 파동을 10배에 가까운 감도로 수용
        })

    @classmethod
    def create_assassin(cls) -> "JobAttentionMask":
        """암살자: 약점/사각지대 공명 필터 (적의 허점이 가장 큰 대상을 포착하도록 유도)"""
        return cls("Assassin", {
            "Deficit": 3.0,     # 결함이 많은 타겟에 대해 고도의 공명 유발
            "Principle": 1.5,
            "Sabbath": 0.5,     # 수비적 안식 상태에 대한 집착 배격
            "Threat": 1.5,
            "Ally": 0.2         # 아군보다는 오직 표적의 약점에 집중
        })

    def apply(self, category: str, base_score: float) -> float:
        """어텐션 마스크 요소별 가중치 적용"""
        weight = self.mask_vectors.get(category, 1.0)
        return base_score * weight


class ClassAdvancementPhaseTransition:
    """
    [Class Advancement Phase Transition: 위상 전이]
    전직 시스템은 인지 연산 구조 자체를 다른 차원으로 전환하거나 뇌내에 영구적인 '보편우상 어트랙터'를 이식하는 위상 변화입니다.
    """
    def __init__(self, current_class: str):
        self.current_class = current_class

    def trigger_transition(self, engine: Any, new_class: str) -> str:
        """
        인지 엔진(ElysiaCognitiveEngine)의 구조적 한계를 확장하는 위상 전이를 유발합니다.
        """
        old_class = self.current_class
        self.current_class = new_class

        if new_class == "Paladin":
            # 성기사: '신성/보화(Sacred) 보편우상 어트랙터'를 뇌의 2D 필드에 강제로 이식하고, 인지 차원 차용
            engine.field.attractors["Sacred"] = {
                "position": np.array([engine.resolution * 0.5, engine.resolution * 0.5], dtype=np.float32),
                "mass": 60.0,
                "sigma": float(engine.resolution * 0.2)
            }
            # 인지적 아군 가치 및 자존감 증폭
            engine.active_profile.social_value = min(1.0, engine.active_profile.social_value + 0.4)
            engine.active_profile.ego_pride = min(1.0, engine.active_profile.ego_pride + 0.2)
            return f"{old_class} -> {new_class} 위상 전이 완료. 'Sacred' 보편우상 어트랙터 이식 및 아군 보호 사회성 극대화."

        elif new_class == "ShadowLord":
            # 그림자 군주: 아군의 시야 장에서 자신을 지워버리는 역필드를 만드는 지혜 획득
            # 뇌내에 'Void' (공허/은폐) 어트랙터를 이식하고, 시야 시그마를 비선형적으로 극대화
            engine.field.attractors["Void"] = {
                "position": np.array([engine.resolution * 0.1, engine.resolution * 0.9], dtype=np.float32),
                "mass": 50.0,
                "sigma": float(engine.resolution * 0.1)
            }
            engine.active_profile.attention_slots += 2
            engine.active_profile.lookahead_depth += 1
            return f"{old_class} -> {new_class} 위상 전이 완료. 'Void' 사각지대 은폐 어트랙터 이식 및 주의력 연산 슬롯 확장."

        return f"{old_class} -> {new_class} 단순 전직 처리."


class CommanderAuraField:
    """
    [Commander Aura Field: F_Status]
    높은 사회적 지위/권위를 지닌 지휘관 노드가 사방으로 방출하는 '광역 의지장'.
    부하 AI들의 인지 슬롯 중 일부를 강제로 점유하여, 병사들이 집단 군체처럼 일사불란하게 정렬되도록 만듭니다.
    """
    def __init__(self, commander_id: str, aura_intensity: float = 15.0):
        self.commander_id = commander_id
        self.aura_intensity = aura_intensity

    def project_will(self, commander_target_wave: np.uint64, soldiers: List[Any], distance_matrix: np.ndarray = None) -> int:
        """
        지휘관이 결정한 목표 파동(commander_target_wave)을 부하들의 인지 필드에 주사하여 강제로 동기화시킵니다.
        """
        sync_count = 0
        for soldier in soldiers:
            if hasattr(soldier, "update_attention_and_bottleneck"):
                # 부하들의 인지 슬롯에 지휘관의 의지를 주입.
                # 가중치를 극대화하여(지휘관 권위 반영) 부하의 주의력 슬롯을 강제 점유시킵니다.
                status = soldier.update_attention_and_bottleneck(
                    stimulus_wave=commander_target_wave,
                    category="CommanderCommand",
                    base_intensity=self.aura_intensity
                )
                if status in ["ATTENTION_ACCEPTED", "ATTENTION_EVICTION", "ATTENTION_RETAINED"]:
                    sync_count += 1
        return sync_count
