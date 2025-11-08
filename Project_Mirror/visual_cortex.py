"""
VisualCortex for Elysia

This module is responsible for 'seeing' and interpreting visual information,
forming the input part of the sensory loop for Project_Mirror.
"""

from typing import Dict, List
import json
import re

from Project_Sophia.wave_mechanics import WaveMechanics

# Use a try-except block for robustness, especially if this runs in different contexts
try:
    from Project_Sophia.gemini_api import GeminiAPI
except ImportError:
    # Provide a mock for environments where GeminiAPI might not be available
    class GeminiAPI:
        def generate_text_from_image(self, prompt: str, image_path: str) -> str:
            print(f"--- MOCK GeminiAPI: Pretending to analyze {image_path} ---")
            return '{"description": "A mock description of the image.", "keywords": ["mock", "image"]}'

class VisualCortex:
    def __init__(self, wave_mechanics: WaveMechanics):
        """Initializes the VisualCortex."""
        self.gemini_api = GeminiAPI()
        self.wave_mechanics = wave_mechanics

    def resonate_with_cosmos(self, keywords: List[str], top_k: int = 3) -> Dict[str, float]:
        """
        Finds which core concepts in the mental cosmos resonate most strongly with the given keywords.
        """
        if not self.wave_mechanics:
            return {}

        # For simplicity, we can use a predefined set of core concepts to measure against
        # In a more advanced system, these could be discovered dynamically
        core_concepts = ["love", "growth", "beauty", "truth", "sadness", "joy", "peace"]

        resonance_scores = {}
        # Ensure the nodes exist in the KG for resonance calculation
        existing_keywords = [kw for kw in keywords if self.wave_mechanics.kg_manager.get_node(kw)]
        existing_core_concepts = [cc for cc in core_concepts if self.wave_mechanics.kg_manager.get_node(cc)]

        for keyword in existing_keywords:
            for core_concept in existing_core_concepts:
                try:
                    resonance = self.wave_mechanics.get_resonance_between(keyword, core_concept)
                    if resonance > 0.1: # Only consider significant resonance
                        resonance_scores[core_concept] = resonance_scores.get(core_concept, 0.0) + resonance
                except Exception as e:
                    print(f"Error calculating resonance between '{keyword}' and '{core_concept}': {e}")

        # Get the top_k resonating concepts
        sorted_concepts = sorted(resonance_scores.items(), key=lambda item: item[1], reverse=True)
        return dict(sorted_concepts[:top_k])

    def analyze_image(self, image_path: str) -> Dict[str, any]:
        """
        Analyzes an image and returns a structured dictionary containing a
        description and keywords.

        Args:
            image_path: The path to the image file to be analyzed.

        Returns:
            A dictionary with 'description' and 'keywords' keys, or an
            empty dictionary if analysis fails.
        """
        prompt = """
        Analyze the provided image carefully. Based on your analysis, generate a JSON object
        with two keys:
        1. "description": A concise, one-sentence description of the image in Korean.
        2. "keywords": A list of 3-5 relevant keywords in Korean that capture the main concepts or objects in the image.

        Your output should be only the raw JSON object, without any surrounding text or markdown.
        """

        try:
            response_text = self.gemini_api.generate_text_from_image(prompt, image_path)

            # Clean up the response to ensure it's valid JSON
            # The model sometimes wraps the JSON in ```json ... ```
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_text = match.group(0)
                analysis = json.loads(json_text)
                # Basic validation
                if 'description' in analysis and 'keywords' in analysis:
                    return analysis

            print(f"Warning: Could not parse valid JSON from Gemini response: {response_text}")
            return {}

        except Exception as e:
            print(f"An error occurred during image analysis: {e}")
            return {}

if __name__ == '__main__':
    # This is a placeholder for a test. To run this, you would need an image
    # file and a configured GEMINI_API_KEY in your .env file.

    # Create a dummy image for testing if it doesn't exist
    from PIL import Image, ImageDraw
    import os

    test_image_path = "test_image.png"
    if not os.path.exists(test_image_path):
        img = Image.new('RGB', (100, 100), color = 'red')
        d = ImageDraw.Draw(img)
        d.text((10,10), "Test", fill=(255,255,0))
        img.save(test_image_path)

    print(f"--- Testing VisualCortex with '{test_image_path}' ---")
    visual_cortex = VisualCortex()
    analysis_result = visual_cortex.analyze_image(test_image_path)

    print("\n--- Analysis Result ---")
    if analysis_result:
        print(f"Description: {analysis_result.get('description')}")
        print(f"Keywords: {analysis_result.get('keywords')}")
    else:
        print("Analysis failed.")

    # Clean up the dummy image
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
