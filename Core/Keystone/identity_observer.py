"""
[IDENTITY OBSERVER - EMERGENT PERSONA ENGINE]
"Identity is not a hardcoded label, but a collapsed wave function of accumulated actions."

Tracks an avatar's behavioral trajectory and computes their collapsed social identity
using cosine similarity against archetypal vector points.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

class IdentityObserver:
    def __init__(self):
        # 5-Dimensional Semantic Action Space:
        # Index 0: Combat (전투)
        # Index 1: Nature (자연 / 야외 활동)
        # Index 2: Creation (제작 / 연마 / 건축)
        # Index 3: Commerce (경제 / 교환 / 금전)
        # Index 4: Study (학문 / 명상 / 지혜)
        self.axes_labels = ["Combat", "Nature", "Creation", "Commerce", "Study"]

        # Archetype Vectors in Semantic Space
        self.archetypes = {
            "전사 (Warrior)":       np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            "목수 (Carpenter)":     np.array([0.0, 0.8, 0.9, 0.0, 0.0]),
            "상인 (Merchant)":      np.array([0.0, 0.0, 0.2, 1.0, 0.0]),
            "학자 (Scholar)":       np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            "대장장이 (Blacksmith)": np.array([0.3, 0.0, 1.0, 0.4, 0.0]),
            "마법학자 (Sage)":       np.array([0.1, 0.0, 0.0, 0.0, 1.0])
        }

    def calculate_similarity(self, v_npc: np.ndarray, v_archetype: np.ndarray) -> float:
        """Calculates cosine similarity between NPC behavioral vector and archetype."""
        norm_npc = np.linalg.norm(v_npc)
        norm_arch = np.linalg.norm(v_archetype)
        if norm_npc == 0 or norm_arch == 0:
            return 0.0
        return float(np.dot(v_npc, v_archetype) / (norm_npc * norm_arch))

    def observe_persona(self, npc_history_vector: np.ndarray) -> Dict[str, Any]:
        """
        Observes the NPC's dynamic vectors and collapses them into an identity report.
        """
        similarities = {}
        for name, arch_vec in self.archetypes.items():
            similarities[name] = self.calculate_similarity(npc_history_vector, arch_vec)

        # Sort by similarity
        sorted_identities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        primary_id, primary_score = sorted_identities[0]
        secondary_id, secondary_score = sorted_identities[1]

        # Generate Emergent Title based on the vector fingerprint
        title = ""
        if primary_score < 0.3:
            title = "방랑자 (Vagabond)"
            description = "특정한 사회적 발자취를 남기지 않은 자유로운 영혼입니다."
        else:
            # Add modifier from secondary traits
            if secondary_score > 0.4:
                title = f"{secondary_id.split(' ')[0]} 성향의 {primary_id.split(' ')[0]}"
            else:
                title = primary_id.split(" ")[0]
            
            # Contextual description based on semantic peaks
            max_axis = int(np.argmax(npc_history_vector))
            peak_behavior = self.axes_labels[max_axis]
            description = f"가장 짙은 행동 궤적은 [{peak_behavior}]이며, 사회적으로는 {title}(으)로 널리 알려지기 시작했습니다."

        return {
            "title": title,
            "description": description,
            "similarities": sorted_identities,
            "primary": (primary_id, primary_score),
            "secondary": (secondary_id, secondary_score)
        }

if __name__ == "__main__":
    observer = IdentityObserver()
    print("🧠 Identity Observer Initialized.")
    
    # Simulation: The Life Story of NPC "Robin"
    # Robin starts as a normal villager with no specific history
    robin_life_vector = np.zeros(5)
    
    print("\n🎬 Phase 1: 로빈은 숲에서 나무를 베고 집을 짓기 시작했습니다 (Nature + Creation)")
    robin_life_vector[1] += 5.0 # Wood cutting
    robin_life_vector[2] += 6.0 # Build hut
    report1 = observer.observe_persona(robin_life_vector)
    print(f"  * 누적 행동 벡터: {robin_life_vector}")
    print(f"  * 마을에서의 정체성: {report1['title']}")
    print(f"  * 소문: \"{report1['description']}\"")
    print(f"  * 유사도 스펙트럼: {report1['similarities'][:3]}")

    print("\n🎬 Phase 2: 몇 년 후, 마을에 마물이 들이닥치자 로빈은 도끼를 들고 싸웠습니다 (Combat 추가)")
    robin_life_vector[0] += 8.0 # Fighting bandits
    # Robin also started trading raw materials (Commerce)
    robin_life_vector[3] += 3.0 # Trading wood
    report2 = observer.observe_persona(robin_life_vector)
    print(f"  * 누적 행동 벡터: {robin_life_vector}")
    print(f"  * 마을에서의 정체성: {report2['title']}")
    print(f"  * 소문: \"{report2['description']}\"")
    print(f"  * 유사도 스펙트럼: {report2['similarities'][:3]}")
