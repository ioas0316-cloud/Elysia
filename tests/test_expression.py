import unittest
from unittest.mock import MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.emotional_cortex import EmotionalCortex, Mood
from Project_Sophia.value_centered_decision import VCDResult, ValueMetrics
from Project_Sophia.response_styler import ResponseStyler
from Project_Mirror.sensory_cortex import SensoryCortex
from Project_Sophia.cognition_pipeline import CognitionPipeline

class TestExpression(unittest.TestCase):

    def test_emotional_cortex_updates_mood(self):
        """Test that EmotionalCortex correctly generates moods."""
        cortex = EmotionalCortex()

        # Test sense_of_accomplishment
        metrics = ValueMetrics(growth_score=25)
        vcd_result = VCDResult("test", 80, 0.8, 0.8, metrics, [])
        cortex.update_mood_from_vcd(vcd_result)
        self.assertEqual(cortex.get_current_mood().primary_mood, "sense_of_accomplishment")

        # Test connectedness
        metrics = ValueMetrics(love_score=70)
        vcd_result = VCDResult("test", 80, 0.8, 0.8, metrics, [])
        cortex.update_mood_from_vcd(vcd_result)
        self.assertEqual(cortex.get_current_mood().primary_mood, "connectedness")

    def test_response_styler(self):
        """Test that ResponseStyler modifies text based on mood."""
        styler = ResponseStyler()

        # Test accomplished style
        mood = Mood(primary_mood="sense_of_accomplishment", intensity=0.8)
        response = "My analysis is complete."
        styled = styler.style_response(response, mood)
        self.assertIn("해냈어요", styled)

        # Test curious style
        mood = Mood(primary_mood="curiosity", intensity=0.8)
        response = "That is a fact."
        styled = styler.style_response(response, mood)
        self.assertIn("흥미롭네요", styled)

        # Test neutral style
        mood = Mood(primary_mood="neutral")
        response = "Hello."
        styled = styler.style_response(response, mood)
        self.assertEqual(response, styled)

    def test_sensory_cortex_accepts_mood(self):
        """Test that SensoryCortex can receive a mood parameter without crashing."""
        cortex = SensoryCortex()
        mood = Mood(primary_mood="connectedness", intensity=0.9)

        try:
            # We are just checking if the call is successful, not the image content
            image_path = cortex.visualize_concept("love", mood=mood)
            self.assertTrue(os.path.exists(image_path))
            os.remove(image_path)
        except Exception as e:
            self.fail(f"SensoryCortex crashed when given a mood: {e}")

    def test_full_pipeline_integration_for_mood(self):
        """End-to-end test for mood generation and expression."""
        pipeline = CognitionPipeline()

        # Mock the VCD to return a result that should trigger a specific mood
        mock_metrics = ValueMetrics(growth_score=30) # Should trigger 'sense_of_accomplishment'
        mock_vcd_result = VCDResult(
            chosen_action="I have completed the task.",
            total_score=90, confidence_score=0.9, value_alignment_score=0.9,
            metrics=mock_metrics, reasoning=[]
        )

        # Replace the real VCD `suggest_action` with a mock that returns our controlled result
        pipeline.vcd.suggest_action = MagicMock(return_value=mock_vcd_result)

        # Process a message that will use this mocked result
        final_response, final_mood = pipeline.process_message("Execute the plan.")

        # 1. Check if the mood was updated correctly
        self.assertEqual(final_mood.primary_mood, "sense_of_accomplishment")

        # 2. Check if the response was styled according to the new mood
        self.assertIn("해냈어요", final_response)
        self.assertIn("성장한 것 같네요", final_response)


if __name__ == '__main__':
    unittest.main()
