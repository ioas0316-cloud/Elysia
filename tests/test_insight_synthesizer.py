import unittest
import sys
import os

# Add the project root to the Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.insight_synthesizer import InsightSynthesizer

class TestInsightSynthesizer(unittest.TestCase):
    """
    Unit tests for the InsightSynthesizer to ensure it correctly
    synthesizes various combinations of facts into natural language.
    """

    def setUp(self):
        """Initialize the InsightSynthesizer for each test."""
        self.synthesizer = InsightSynthesizer()

    def test_synthesize_combined_facts(self):
        """
        Tests the synthesis of a combination of static and dynamic facts.
        The output should intelligently link the two sources of information.
        """
        combined_facts = [
            "[정적] '햇빛'은(는) '식물 성장'을(를) 유발할 수 있습니다.",
            "'햇빛'(으)로 시뮬레이션한 결과, 다음과 같은 동적 영향이 관찰되었습니다:",
            "  - '식물 성장' 개념이 활성화되었습니다 (에너지: 0.41).",
            "  - '산소 발생' 개념이 활성화되었습니다 (에너지: 0.2)."
        ]

        insight = self.synthesizer.synthesize(combined_facts)

        print(f"\n--- Combined Insight Test --- \nInput Facts: {combined_facts}\nSynthesized: {insight}")

        # Check that it acknowledges both sources of information
        self.assertIn("제 기억 속 지식에 따르면", insight)
        self.assertIn("내면 세계에서", insight)
        self.assertIn("현상이 관찰되었어요", insight)
        # Check that it expresses a higher-level conclusion
        self.assertIn("더 깊은 확신을 갖게 되었어요", insight)

    def test_synthesize_static_only(self):
        """
        Tests the synthesis of only static facts.
        The output should be a clear statement of known facts.
        """
        static_only = [
            "[정적] '소크라테스'은(는) '인간'의 한 종류입니다."
        ]

        insight = self.synthesizer.synthesize(static_only)

        print(f"\n--- Static Only Insight Test --- \nInput Facts: {static_only}\nSynthesized: {insight}")

        # Check for a direct, knowledge-based statement
        self.assertIn("제가 알기로는", insight)
        self.assertIn("'소크라테스'은(는) '인간'의 한 종류입니다.", insight)

    def test_synthesize_dynamic_only(self):
        """
        Tests the synthesis of only dynamic simulation results.
        The output should convey discovery and acknowledge uncertainty.
        """
        dynamic_only = [
            "'사랑'(으)로 시뮬레이션한 결과, 다음과 같은 동적 영향이 관찰되었습니다:",
            "  - '기쁨' 개념이 활성화되었습니다 (에너지: 0.8).",
            "  - '성장' 개념이 활성화되었습니다 (에너지: 0.6)."
        ]

        insight = self.synthesizer.synthesize(dynamic_only)

        print(f"\n--- Dynamic Only Insight Test --- \nInput Facts: {dynamic_only}\nSynthesized: {insight}")

        # Check that it acknowledges the lack of prior knowledge but highlights the discovery
        self.assertIn("명확한 지식은 없지만", insight)
        self.assertIn("가상 실험을 해보니", insight)
        self.assertIn("놀라운 결과가 나타났어요", insight)
        # Check that it proposes a hypothesis about the connection
        self.assertIn("아직 모르는 깊은 연관성이 있을지도 몰라요", insight)

    def test_synthesize_no_facts(self):
        """
        Tests the synthesizer's behavior when given no facts.
        It should return a graceful fallback message.
        """
        no_facts = []

        insight = self.synthesizer.synthesize(no_facts)

        print(f"\n--- No Facts Test --- \nInput Facts: {no_facts}\nSynthesized: {insight}")

        # Check for the specific fallback message
        self.assertIn("아직 깊이 생각해본 적이 없어요", insight)

if __name__ == '__main__':
    unittest.main()
