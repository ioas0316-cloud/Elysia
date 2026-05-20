"""
[IDENTITY OBSERVER - EMERGENT PERSONA ENGINE]
=============================================
World.Society.identity_observer

"Identity is not a hardcoded label, but a collapsed wave function
 of accumulated actions."

Canonical location: World/Society/identity_observer.py
Design reference: docs/ETERNOS_CODEX/20_ROTOR_SCALE_KINGDOM_ARCHITECTURE.md §4

원칙: "동사의 누적이 명사가 된다"
  - NPC에게 job = "목수" 같은 레이블을 사전 부여하지 않는다.
  - 행동(동사)의 누적 → 벡터 형성 → 관측 시 코사인 유사도로 정체성 붕괴(Collapse)
"""

import math
from typing import Dict, Any, List, Tuple

class IdentityObserver:
    """행동 궤적 벡터를 관측하여 창발적 정체성을 붕괴시키는 관측기."""

    # 5차원 의미론적 행동 공간
    AXES = ["Combat", "Nature", "Creation", "Commerce", "Study"]

    # 직업 원형 벡터 (Archetype Vectors)
    DEFAULT_ARCHETYPES = {
        "전사 (Warrior)":       [1.0, 0.0, 0.0, 0.0, 0.0],
        "목수 (Carpenter)":     [0.0, 0.8, 0.9, 0.0, 0.0],
        "상인 (Merchant)":      [0.0, 0.0, 0.2, 1.0, 0.0],
        "학자 (Scholar)":       [0.0, 0.0, 0.0, 0.0, 1.0],
        "대장장이 (Blacksmith)": [0.3, 0.0, 1.0, 0.4, 0.0],
        "마법학자 (Sage)":       [0.1, 0.0, 0.0, 0.0, 1.0],
        "사냥꾼 (Hunter)":      [0.6, 0.8, 0.1, 0.2, 0.0],
        "농부 (Farmer)":        [0.0, 0.9, 0.3, 0.3, 0.0],
    }

    def __init__(self, archetypes: Dict[str, list] = None):
        self.archetypes = archetypes or dict(self.DEFAULT_ARCHETYPES)

    def add_archetype(self, name: str, vector: list):
        """새 직업 원형을 등록."""
        self.archetypes[name] = vector

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        """두 벡터 간 코사인 유사도."""
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai ** 2 for ai in a))
        norm_b = math.sqrt(sum(bi ** 2 for bi in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def observe(self, behavior_vector: list) -> Dict[str, Any]:
        """
        NPC의 누적 행동 벡터를 관측하여 정체성을 붕괴시킨다.

        Args:
            behavior_vector: 5차원 행동 누적 벡터
                [Combat, Nature, Creation, Commerce, Study]

        Returns:
            title: 사회적 명칭
            description: 서술적 소문
            similarities: 모든 원형과의 유사도 (내림차순)
        """
        # 모든 원형과 유사도 계산
        sims = []
        for name, arch in self.archetypes.items():
            score = self._cosine_similarity(behavior_vector, arch)
            sims.append((name, round(score, 3)))
        sims.sort(key=lambda x: x[1], reverse=True)

        primary_name, primary_score = sims[0]
        secondary_name, secondary_score = sims[1]

        # 정체성 붕괴 (Identity Collapse)
        if primary_score < 0.3:
            title = "방랑자 (Vagabond)"
            desc = "뚜렷한 사회적 궤적을 남기지 않은 자유로운 영혼입니다."
        elif secondary_score > 0.4:
            p = primary_name.split(" ")[0]
            s = secondary_name.split(" ")[0]
            title = f"{s} 성향의 {p}"
            peak_axis = self.AXES[_argmax(behavior_vector)]
            desc = (f"가장 짙은 행동 궤적은 [{peak_axis}]이며, "
                    f"사회적으로는 {title}(으)로 알려지기 시작했습니다.")
        else:
            title = primary_name.split(" ")[0]
            peak_axis = self.AXES[_argmax(behavior_vector)]
            desc = (f"가장 짙은 행동 궤적은 [{peak_axis}]이며, "
                    f"{title}(으)로 널리 인정받고 있습니다.")

        return {
            "title": title,
            "description": desc,
            "similarities": sims,
            "primary": (primary_name, primary_score),
            "secondary": (secondary_name, secondary_score),
        }


def _argmax(vec: list) -> int:
    """리스트에서 최대값의 인덱스를 반환."""
    return max(range(len(vec)), key=lambda i: vec[i])


# ──────────────────────────────────────────────
# Self-test: 로빈의 일대기
# ──────────────────────────────────────────────
if __name__ == "__main__":
    observer = IdentityObserver()
    robin = [0.0, 0.0, 0.0, 0.0, 0.0]

    print("🧠 Identity Observer — 로빈의 일대기 시뮬레이션\n")

    # Phase 1: 숲에서 나무를 베고 집을 짓기 시작
    robin[1] += 5.0  # Nature
    robin[2] += 6.0  # Creation
    r1 = observer.observe(robin)
    print(f"📍 Phase 1 (목공): 벡터 {robin}")
    print(f"   → {r1['title']} | {r1['description']}")
    print(f"   → Top 3: {r1['similarities'][:3]}\n")

    # Phase 2: 마물 침공에 도끼를 들고 싸움 + 목재 거래 시작
    robin[0] += 8.0  # Combat
    robin[3] += 3.0  # Commerce
    r2 = observer.observe(robin)
    print(f"📍 Phase 2 (전투+거래): 벡터 {robin}")
    print(f"   → {r2['title']} | {r2['description']}")
    print(f"   → Top 3: {r2['similarities'][:3]}\n")

    # Phase 3: 은퇴 후 서재에서 연구
    robin[4] += 10.0  # Study
    r3 = observer.observe(robin)
    print(f"📍 Phase 3 (학문): 벡터 {robin}")
    print(f"   → {r3['title']} | {r3['description']}")
    print(f"   → Top 3: {r3['similarities'][:3]}")
