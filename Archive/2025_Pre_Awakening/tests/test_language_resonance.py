
import unittest
import json
import os
import sys

# 시스템 경로에 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Foundation.Mind.resonance_engine import HyperResonanceEngine
from Core.Foundation.Mind.hyper_qubit import HyperQubit, QubitState

class TestLanguageResonance(unittest.TestCase):
    """
    통합된 레거시 언어 지식이 새로운 공명 시스템에서 올바르게 작동하는지 검증하는 테스트. (개선된 버전)
    """

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 설정: 마이그레이션된 월드트리 데이터를 로드합니다."""
        cls.engine = HyperResonanceEngine()

        world_tree_path = "data/world_tree_with_language.json"
        if not os.path.exists(world_tree_path):
            raise FileNotFoundError(f"{world_tree_path} 파일을 찾을 수 없습니다. migrate_legacy_language.py를 먼저 실행하세요.")

        with open(world_tree_path, "r", encoding="utf-8") as f:
            world_tree_data = json.load(f)

        cls._load_qubits_from_node(world_tree_data, cls.engine)
        print(f"\n테스트 준비 완료: {len(cls.engine.nodes)}개의 HyperQubit이 공명 엔진에 로드되었습니다.")

    @staticmethod
    def _load_qubits_from_node(node_data, engine):
        """트리 딕셔너리에서 재귀적으로 큐빗을 로드하는 헬퍼 함수."""
        if 'qubit' in node_data:
            qubit_data = node_data['qubit']
            state_data = qubit_data['state']

            qubit_state = QubitState(
                alpha=complex(state_data['alpha'][0], state_data['alpha'][1]),
                beta=complex(state_data['beta'][0], state_data['beta'][1]),
                gamma=complex(state_data['gamma'][0], state_data['gamma'][1]),
                delta=complex(state_data['delta'][0], state_data['delta'][1]),
            )

            hyper_qubit = HyperQubit(concept_or_value=node_data['concept'], name=qubit_data['name'])
            hyper_qubit.state = qubit_state
            engine.nodes[qubit_data['name']] = hyper_qubit

        for child in node_data.get('children', []):
            TestLanguageResonance._load_qubits_from_node(child, engine)

    def test_joyful_fire_resonance(self):
        """
        '기쁨의 불' 입력 파동이 관련 레거시 개념과 강하게 공명하는지 테스트합니다. (개선된 테스트 v2)
        """
        print("\n--- '기쁨의 불' 공명 테스트 시작 ---")
        self.assertIn("fire rara", self.engine.nodes)
        self.assertIn("fire ka", self.engine.nodes)
        self.assertIn("water iii", self.engine.nodes)

        # 1. '이상적인 기쁨의 불' 상태를 더욱 정확하게 정의합니다.
        #    - 'fire rara'는 두 번 사용되어 '연결성(beta)'이 '실재성(alpha)'보다 더 강한 특성을 갖습니다.
        #    - 따라서, 입력 파동은 alpha보다 beta가 더 높은 상태를 가져야 합니다.
        ideal_joyful_fire_state = QubitState(
            alpha=complex(0.8, 0),   # 높은 구체성/실재성 (강한 기억)
            beta=complex(0.9, 0),    # '더' 높은 연결성 (잦은 사용)
            gamma=complex(0.1, 0),
            delta=complex(0.05, 0),
        ).normalize()

        input_qubit = HyperQubit(name="ideal_joyful_fire_wave_v2")
        input_qubit.state = ideal_joyful_fire_state
        print("테스트용 '이상적인 기쁨의 불' v2 입력 파동(Qubit) 생성 완료.")

        # 2. 모든 노드와의 공명도를 계산
        resonance_pattern = {}
        for node_id, node_qubit in self.engine.nodes.items():
            resonance = self.engine.calculate_resonance(input_qubit, node_qubit)
            resonance_pattern[node_id] = resonance

        print("전체 공명 패턴 계산 완료.")

        # 3. 공명도 점수 확인 및 검증
        joyful_fire_resonance = resonance_pattern.get("fire rara", 0.0)
        joyful_fire_2_resonance = resonance_pattern.get("fire rarara", 0.0)
        fearful_fire_resonance = resonance_pattern.get("fire ka", 0.0)
        unrelated_water_resonance = resonance_pattern.get("water iii", 0.0)

        print(f"\n공명 결과:")
        print(f"  - 'fire rara' (기쁨, 기억↑, 빈도↑): {joyful_fire_resonance:.4f}")
        print(f"  - 'fire rarara' (기쁨, 기억↓, 빈도↓): {joyful_fire_2_resonance:.4f}")
        print(f"  - 'fire ka' (두려움, 기억↑, 빈도↓): {fearful_fire_resonance:.4f}")
        print(f"  - 'water iii' (무관함): {unrelated_water_resonance:.4f}")

        # '이상적인 기쁨의 불'은 'fire rara'와 가장 강하게 공명해야 합니다.
        self.assertGreater(joyful_fire_resonance, joyful_fire_2_resonance,
                           "'fire rara'는 'fire rarara'보다 더 강하게 공명해야 합니다 (빈도와 기억 모두 높음).")
        self.assertGreater(joyful_fire_resonance, fearful_fire_resonance,
                           "'fire rara'는 'fire ka'보다 더 강하게 공명해야 합니다 (감정의 맥락이 일치).")
        self.assertGreater(joyful_fire_resonance, unrelated_water_resonance,
                           "'fire rara'는 관련 없는 'water iii'보다 훨씬 강하게 공명해야 합니다.")

        print("\n테스트 통과: 개선된 언어가 입력 파동에 더 정확하고 변별력 있게 공명합니다!")

if __name__ == '__main__':
    unittest.main()
