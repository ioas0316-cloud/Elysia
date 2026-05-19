import unittest
import sys
from pyquaternion import Quaternion

sys.path.append('.')
from Project_Elysia.core.divine_engine import ElysiaDivineEngineV2

class TestDivineEngine(unittest.TestCase):

    def setUp(self):
        """매 테스트 전에 새로운 엔진 인스턴스를 생성합니다."""
        self.engine = ElysiaDivineEngineV2()

    def test_initialization(self):
        """엔진이 'main' 브랜치를 가지고 정상적으로 초기화되는지 테스트합니다."""
        self.assertIn("main", self.engine.branches)
        self.assertEqual(self.engine.current_branch_id, "main")
        self.assertIsNone(self.engine.current_node_id)

    def test_ingest_experience(self):
        """새로운 경험이 타임라인에 정상적으로 기록되는지 테스트합니다."""
        node1 = self.engine.ingest({'truth': 1.0, 'emotion': 0.5}, note="First Event")
        self.assertEqual(self.engine.current_node_id, node1.id)
        self.assertIn(node1.id, self.engine.nodes)
        self.assertEqual(len(self.engine.branches["main"].nodes), 1)

        node2 = self.engine.ingest({'truth': 0.8, 'causality': 0.7}, note="Second Event")
        self.assertEqual(node2.parent_id, node1.id)
        self.assertEqual(len(self.engine.branches["main"].nodes), 2)

    def test_rewind_and_fast_forward(self):
        """시간을 되감고 빨리감는 기능이 정확히 포인터를 이동시키는지 테스트합니다."""
        node1 = self.engine.ingest({}, note="1")
        node2 = self.engine.ingest({}, note="2")
        self.engine.ingest({}, note="3")

        self.engine.rewind(1)
        self.assertEqual(self.engine.current_node_id, node2.id)

        self.engine.rewind(100) # 경계를 넘어서 되감기
        self.assertEqual(self.engine.current_node_id, node1.id)

        self.engine.fast_forward(1)
        self.assertEqual(self.engine.current_node_id, node2.id)

    def test_edit_fate_creates_branch(self):
        """'운명 편집'이 새로운 브랜치를 생성하고 그 위에서 역사가 이어지는지 테스트합니다."""
        self.engine.ingest({}, note="A")

        edited_node = self.engine.edit_fate({'emotion': 1.0}, note="B-prime")
        self.assertNotEqual(edited_node.branch_id, "main")
        self.assertIn(edited_node.branch_id, self.engine.branches)
        self.assertEqual(self.engine.current_branch_id, edited_node.branch_id)

        # 새로운 브랜치의 첫 노드는 편집된 노드여야 함
        new_branch = self.engine.branches[edited_node.branch_id]
        self.assertEqual(new_branch.nodes[0], edited_node.id)

    def test_scope_operations(self):
        """부분 시간 조작(scope) 기능이 독립적으로 작동하는지 테스트합니다."""
        # 전역 노드 및 스코프 노드 생성
        self.engine.ingest({}, note="Global 1")
        self.engine.ingest({'emotion': 0.9}, note="Love 1", scopes=["love"])
        self.engine.ingest({}, note="Global 2")
        love_node_2 = self.engine.ingest({'emotion': 1.0}, note="Love 2", scopes=["love"])

        # 'love' 스코프의 현재 노드는 love_node_2 여야 함
        self.assertEqual(self.engine.scope_current_node["love"], love_node_2.id)

        # 'love' 스코프만 되감기
        self.engine.rewind_scope("love", 1)

        # 'love' 스코프 포인터는 첫번째 love 노드로 이동해야 함
        love_node_1 = self.engine.nodes[self.engine.scope_current_node["love"]]
        self.assertEqual(love_node_1.note, "Love 1")

        # 전역 포인터는 영향을 받지 않아야 함
        self.assertEqual(self.engine.current_node_id, love_node_2.id)

        # 'love' 스코프 운명 편집
        edited_love_node = self.engine.edit_fate_scope("love", {'beauty': 1.0}, note="A different love")

        # 새로운 브랜치가 생성되어야 함
        self.assertNotEqual(edited_love_node.branch_id, "main")
        # 해당 노드는 'love' 스코프를 가져야 함
        self.assertIn("love", edited_love_node.scopes)
        # 'love' 스코프의 현재 노드는 이 노드여야 함
        self.assertEqual(self.engine.scope_current_node["love"], edited_love_node.id)

if __name__ == '__main__':
    unittest.main()
