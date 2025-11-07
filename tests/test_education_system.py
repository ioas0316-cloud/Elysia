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
                        {"type": "concept", "label": "person"},
                        {"type": "concept", "label": "banana"},
                        {"type": "relation", "source": "person", "target": "banana", "label": "eats"}
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

    @patch('os.path.exists', return_value=True)
    def test_full_lesson_flow_with_knowledge_update(self, mock_path_exists):
        """
        Test the full flow, ensuring KGManager is called correctly.
        """
        # More realistic learning points
        learning_points = [
            {"type": "concept", "label": "person"},
            {"type": "concept", "label": "banana"},
            {"type": "relation", "source": "person", "target": "banana", "label": "eats"}
        ]
        self.lesson_data['frames'][0]['learning_points'] = learning_points

        # Re-write the dummy textbook with updated learning points
        with open(self.textbook_path, 'w', encoding='utf-8') as f:
            json.dump(self.lesson_data, f)

        # Mock get_node to simulate nodes not existing initially
        self.mock_kg_manager.get_node.return_value = None

        # Start the lesson
        self.tutor_cortex.start_lesson(self.textbook_path)

        # 1. Verify SensoryCortex was called
        self.mock_sensory_cortex.render_storybook_frame.assert_called_once()

        # 2. Verify image file existence check was made (part of a complete test)
        mock_path_exists.assert_called()

        # 3. Verify KGManager was called to add nodes with visual experience
        expected_calls = [
            call('person', properties={'description': 'Concept learned from visual experience: path/to/mock_image.png', 'category': 'learned_concept', 'experience_visual': ['path/to/mock_image.png']}),
            call('banana', properties={'description': 'Concept learned from visual experience: path/to/mock_image.png', 'category': 'learned_concept', 'experience_visual': ['path/to/mock_image.png']})
        ]
        self.mock_kg_manager.add_node.assert_has_calls(expected_calls, any_order=True)

        # 4. Verify KGManager was called to add the edge with visual experience
        self.mock_kg_manager.add_edge.assert_called_once_with(
            'person', 'banana', 'eats', properties={'experience_visual': 'path/to/mock_image.png'}
        )

        # 5. Verify that the KG was saved
        self.mock_kg_manager.save.assert_called_once()

if __name__ == '__main__':
    unittest.main()
