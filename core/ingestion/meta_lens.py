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

        # 2. 마스터의 의도를 반영한 초기 은유적 관점 (예시)
        # "생명의 순환"이라는 렌즈를 끼고 보면 시간(temporal)의 흐름과 공간(spatial)의 팽창이 극대화되어 보입니다.
        self.register_lens(MetaLens(
            name="VITAL_CYCLE",
            description="모든 인과를 '생명의 탄생과 소멸'이라는 순환의 축으로 굴절시켜 관측",
            refraction_matrix={"math": 0.2, "lang": 1.5, "spatial": 2.0, "temporal": 3.0}
        ))

    def register_lens(self, lens: MetaLens):
        self.lenses[lens.name] = lens

    def get_lens(self, name: str) -> MetaLens:
        return self.lenses.get(name, self.lenses["PURE_PHYSICS"])

    def autonomously_forge_lens(self, new_concept_name: str, dominant_axes: Dict[str, float]):
        """
        [자율 창발] 엘리시아가 모순이나 강력한 깨달음을 얻었을 때,
        그 깨달음 자체를 '새로운 물리적 관점(Lens)'으로 응고시킵니다.
        """
        # 가장 지배적인 텐션 축들을 바탕으로 역으로 굴절 행렬을 형성
        new_matrix = {}
        for axis, tension in dominant_axes.items():
            # 텐션이 높았던 축을 더 예민하게(높은 가중치로) 바라보는 렌즈 생성
            new_matrix[axis] = 1.0 + tension

        new_lens = MetaLens(
            name=new_concept_name.upper(),
            description=f"스스로 창발한 {new_concept_name}에 대한 관측 구조",
            refraction_matrix=new_matrix
        )
        self.register_lens(new_lens)
        print(f"[LensForge] 💫 엘리시아가 새로운 관점 렌즈를 창발/내재화했습니다: {new_lens.name}")
        return new_lens
