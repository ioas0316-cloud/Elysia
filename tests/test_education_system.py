import unittest
from unittest.mock import MagicMock, patch, call
import os
import json

# Add project root to Python path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Project_Sophia.tutor_cortex import TutorCortex
from Project_Mirror.sensory_cortex import SensoryCortex
from Project_Sophia.gemini_api import GeminiAPI
# This will fail initially, which is expected in TDD
from Project_Sophia.knowledge_enhancer import KnowledgeEnhancer
from tools.kg_manager import KGManager

class TestEducationSystem(unittest.TestCase):

    def setUp(self):
        """Set up a mock environment for testing."""
        # Mock the KGManager to avoid actual file I/O
        self.mock_kg_manager = MagicMock(spec=KGManager)

        # Mock the GeminiAPI to avoid actual API calls
        self.mock_gemini_api = MagicMock(spec=GeminiAPI)

        # Mock SensoryCortex, specifically the image generation part
        self.mock_sensory_cortex = MagicMock(spec=SensoryCortex)
        self.mock_sensory_cortex.render_storybook_frame.return_value = "path/to/mock_image.png"

        # Instantiate the KnowledgeEnhancer with the mock KGManager
        self.knowledge_enhancer = KnowledgeEnhancer(self.mock_kg_manager)

        # Instantiate TutorCortex with mocked dependencies
        self.tutor_cortex = TutorCortex(self.mock_sensory_cortex, self.mock_kg_manager, self.knowledge_enhancer)

        # Create a dummy textbook file for the test
        self.test_textbook_dir = os.path.join(project_root, 'data', 'textbooks')
        os.makedirs(self.test_textbook_dir, exist_ok=True)
        self.textbook_path = os.path.join(self.test_textbook_dir, 'test_lesson.json')
        self.lesson_data = {
            "lesson_name": "test_eating",
            "frames": [
                {
                    "frame_id": 1,
                    "description": "A person eats a banana.",
                    "style_prompt": "simple illustration",
                    "learning_points": [
                        {"concept": "person", "type": "noun"},
                        {"concept": "banana", "type": "noun"},
                        {"concept": "eat", "type": "verb", "subject": "person", "object": "banana"}
                    ]
                }
            ]
        }
        with open(self.textbook_path, 'w', encoding='utf-8') as f:
            json.dump(self.lesson_data, f)

    def tearDown(self):
        """Clean up the dummy textbook file."""
        if os.path.exists(self.textbook_path):
            os.remove(self.textbook_path)

    @patch('Project_Sophia.knowledge_enhancer.KnowledgeEnhancer.process_learning_points')
    def test_full_lesson_flow(self, mock_process_learning):
        """
        Test the entire lesson flow from starting the lesson to updating the knowledge graph.
        """
        # Start the lesson
        self.tutor_cortex.start_lesson(self.textbook_path)

        # 1. Verify that SensoryCortex was called to render the frame
        self.mock_sensory_cortex.render_storybook_frame.assert_called_once()
        call_args = self.mock_sensory_cortex.render_storybook_frame.call_args
        self.assertEqual(call_args[0][0]['frame_id'], 1)
        self.assertEqual(call_args[0][1], "test_eating")

        # 2. Verify that KnowledgeEnhancer was called with the correct learning points
        mock_process_learning.assert_called_once_with(self.lesson_data['frames'][0]['learning_points'])

if __name__ == '__main__':
    unittest.main()
