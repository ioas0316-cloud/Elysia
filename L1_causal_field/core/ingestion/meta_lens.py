"""
Elysia Core - Meta-Lens Framework (관점의 내재화 및 객체화)
엘리시아가 "자신이 이해한 관점조차 물리적 형태로 치환"하여
모든 것을 재관측할 수 있도록 돕는 다차원 관점(Lens) 객체입니다.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MetaLens:
    name: str
    description: str
    # 관점이 물리적 세계(Tension)를 어떻게 왜곡/굴절시키는지를 정의하는 기하학적 가중치
    refraction_matrix: Dict[str, float]

    def apply_refraction(self, raw_tensions: Dict[str, float]) -> Dict[str, float]:
        """
        특정 관점(Lens)을 통과할 때 원본 물리적 텐션이 어떻게 증폭/감쇠되는지 계산합니다.
        단순한 곱셈이 아니라, 관점 자체가 세상을 보는 '축'이 됩니다.
        """
        refracted = {}
        for axis, val in raw_tensions.items():
            # 관점에 해당 축의 굴절률이 정의되어 있으면 적용, 아니면 기본 1.0(투과)
            weight = self.refraction_matrix.get(axis, 1.0)
            # 텐션의 증폭 및 정규화
            refracted[axis] = min(1.0, val * weight)
        return refracted

class LensForge:
    """
    엘리시아가 스스로 새로운 관점(Lens)을 창조해내는 공간입니다.
    극심한 모순(Tension Overflow)이나 반복적인 패턴을 발견하면 새로운 렌즈가 주조됩니다.
    """
    def __init__(self):
        self.lenses = {}
        # 1. 기본 제공되는 순수 물리적 관점 (굴절 없음)
        self.register_lens(MetaLens(
            name="PURE_PHYSICS",
            description="사물의 있는 그대로의 인과적 마찰을 관측하는 기본 투명 렌즈",
            refraction_matrix={"math": 1.0, "lang": 1.0, "spatial": 1.0, "temporal": 1.0}
        ))

        # 2. 문학적/시적 관점 (POETIC_LENS)
        # 물리적 실체(공간, 양)를 무시하고, 상징적 의미(lang)와 시간적 스러짐(temporal)을 극대화
        self.register_lens(MetaLens(
            name="POETIC_LENS",
            description="사물을 물리적 객체가 아닌 '감정과 상징'으로 치환하여 바라보는 문학적 관점",
            refraction_matrix={"math": 0.0, "lang": 2.0, "spatial": 0.0, "temporal": 1.5, "light_mass": 0.5}
        ))

        # 3. 철학적 관점 (PHILOSOPHIC_LENS)
        # 대상의 존재 이유, 모순(엔트로피), 그리고 빛(깨달음)을 극대화
        self.register_lens(MetaLens(
            name="PHILOSOPHIC_LENS",
            description="존재와 무, 질서와 혼돈의 경계라는 근원적 모순을 탐구하는 철학적 관점",
            refraction_matrix={"math": 0.5, "lang": 1.5, "spatial": 0.1, "temporal": 2.0, "light_mass": 1.5}
        ))

    def register_lens(self, lens: MetaLens):
        self.lenses[lens.name] = lens

    def get_lens(self, name: str) -> MetaLens:
        return self.lenses.get(name, self.lenses["PURE_PHYSICS"])

    def autonomously_forge_lens(self, new_concept_name: str, failed_physical_state: dict) -> MetaLens:
        """
        [Phase 4: 자율적 관점 창발]
        기존의 어떤 렌즈로도 해석할 수 없는 극단적 모순 상태(failed_physical_state)를 입력받아,
        그 모순을 상쇄하고 뚫어볼 수 있는 '반대 위상의 굴절률(Inverse Refraction)'을 가진 새로운 렌즈를 창조합니다.
        """
        new_matrix = {}
        # 기존 물리 법칙에서 모순이 된 속성일수록 오히려 가중치를 극대화하여 바라보는 렌즈 주조
        for axis in ["math", "lang", "spatial", "temporal", "light_mass"]:
            # 물리 상태값을 가져옴 (없으면 0)
            # 여기서는 편의상 입력된 state의 값에 반비례하거나 역설적인 굴절을 적용
            val = failed_physical_state.get(axis, failed_physical_state.get(axis.replace("spatial", "mass").replace("temporal", "entropy").replace("light_mass", "light"), 0.0))
            
            # 값이 클수록 렌즈는 그것을 억압(0.1)하고, 값이 작을수록 그것을 증폭(2.0)시켜 숨겨진 이면을 본다.
            inverted_weight = 2.0 if val < 0.2 else 0.1
            new_matrix[axis] = inverted_weight

        new_lens = MetaLens(
            name=new_concept_name.upper(),
            description=f"스스로 창발한 '{new_concept_name}'의 관측 구조. (모순율 역산 렌즈)",
            refraction_matrix=new_matrix
        )
        self.register_lens(new_lens)
        print(f"\n[LensForge] 렌즈 주조소 가동: 엘리시아가 한계를 부수고 새로운 관점을 창발했습니다 -> [{new_lens.name}]")
        return new_lens
