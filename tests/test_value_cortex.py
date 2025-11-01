import unittest
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.value_cortex import ValueCortex
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

class TestValueCortexSemanticDepth(unittest.TestCase):

    def setUp(self):
        """테스트를 위한 ValueCortex 및 의존성 설정"""
        self.kg_manager = KGManager()
        self.kg_manager.kg = {
            "nodes": [
                {"id": "사랑"}, {"id": "친절"}, {"id": "도움"},
                {"id": "성장"}, {"id": "지식"},
                {"id": "계산"}, {"id": "숫자"}
            ],
            "edges": [
                {"source": "친절", "target": "사랑", "relation": "is_a"},
                {"source": "도움", "target": "친절", "relation": "is_a"},
                {"source": "지식", "target": "성장", "relation": "is_a"},
                {"source": "성장", "target": "사랑", "relation": "is_a"},
                {"source": "계산", "target": "숫자", "relation": "is_a"}
            ]
        }
        self.wave_mechanics = WaveMechanics(self.kg_manager)
        self.value_cortex = ValueCortex(self.wave_mechanics)

    def test_measure_semantic_depth(self):
        """'의미의 깊이'가 개념의 연결성 풍부성에 비례하는지 테스트합니다."""
        # '도움' -> '친절' -> '사랑' (풍부한 연결)
        action_help = "도움이 필요하신가요?"
        depth_help = self.value_cortex.measure_semantic_depth(action_help)

        # '계산' -> '숫자' (상대적으로 고립됨)
        action_calc = "계산해주세요."
        depth_calc = self.value_cortex.measure_semantic_depth(action_calc)

        # '도움'의 의미 깊이가 '계산'보다 훨씬 더 깊어야 함
        self.assertGreater(depth_help, depth_calc)

        # '도움'의 울림 총량이 '계산'의 울림 총량보다 크다는 것만 검증
        self.assertTrue(depth_help > depth_calc)

    def test_decide_chooses_deepest_action(self):
        """가장 '의미의 깊이'가 깊은 행동을 선택하는지 테스트합니다."""
        candidates = [
            "1+1 계산해주세요.",  # 얕은 의미
            "성장을 위해 새로운 지식을 알려드릴게요." # 깊은 의미
        ]

        best_action = self.value_cortex.decide(candidates)

        # '성장'과 '지식'이 포함된 두 번째 후보가 선택되어야 함
        self.assertEqual(best_action, candidates[1])

if __name__ == '__main__':
    unittest.main()
