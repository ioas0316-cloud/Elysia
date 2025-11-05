"""
TutorCortex for Elysia

This module acts as a "teacher" for Elysia, guiding her through
structured learning materials, such as the "picture books" defined in `data/textbooks/`.
It orchestrates the learning process by presenting visual information (via SensoryCortex)
and textual information (via CognitionPipeline) to Elysia.
"""
import os
import json
from Project_Mirror.sensory_cortex import SensoryCortex
from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Sophia.knowledge_enhancer import KnowledgeEnhancer
from tools.kg_manager import KGManager
import logging

logger = logging.getLogger(__name__)

class TutorCortex:
    def __init__(self, sensory_cortex: SensoryCortex, kg_manager: KGManager, knowledge_enhancer: KnowledgeEnhancer, cognition_pipeline: CognitionPipeline = None):
        """
        Initializes the TutorCortex.

        Args:
            sensory_cortex: The sensory cortex to request visualizations from.
            kg_manager: The knowledge graph manager.
            knowledge_enhancer: The module to process and add learned concepts to the KG.
            cognition_pipeline: The main cognition pipeline (optional).
        """
        self.sensory_cortex = sensory_cortex
        self.kg_manager = kg_manager
        self.knowledge_enhancer = knowledge_enhancer
        self.cognition_pipeline = cognition_pipeline

    def start_lesson(self, textbook_path: str):
        """
        Starts a lesson from a given textbook file.

        Args:
            textbook_path: The path to the JSON textbook file.
        """
        logger.info(f"Starting lesson from textbook: {textbook_path}")
        try:
            with open(textbook_path, 'r', encoding='utf-8') as f:
                textbook = json.load(f)
        except FileNotFoundError:
            logger.error(f"Textbook file not found at: {textbook_path}")
            return
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding textbook JSON from {textbook_path}: {e}")
            return

        lesson_name = textbook.get('lesson_name', 'untitled_lesson')

        logger.info(f"Teaching lesson: '{lesson_name}'")

        # Process each frame in the storybook
        for frame in textbook.get('frames', []):
            self._process_frame(frame, lesson_name)

        logger.info(f"Lesson '{lesson_name}' completed.")

    def _process_frame(self, frame: dict, lesson_name: str):
        """
        Processes a single frame of the textbook, presenting visual info and updating knowledge.
        """
        description = frame.get('description')
        learning_points = frame.get('learning_points')

        if not description or not learning_points:
            logger.warning(f"Skipping incomplete frame in lesson '{lesson_name}': {frame}")
            return

        # 1. Ask the SensoryCortex to draw the scene.
        logger.info(f"Requesting visualization for: '{description}'")
        image_path = self.sensory_cortex.render_storybook_frame(frame, lesson_name)

        if not image_path or not os.path.exists(image_path):
            logger.error(f"Failed to get a valid image for frame, skipping. Frame: {frame}")
            return

        # 2. Process the learning points with the KnowledgeEnhancer.
        logger.info(f"Processing learning points for frame {frame.get('frame_id')}")
        self.knowledge_enhancer.process_learning_points(learning_points, image_path)

        print(f"--- Frame {frame.get('frame_id')} Processed ---")
        print(f"  Visual: {image_path}")
        print(f"  Learned: {len(learning_points)} points")
        print(f"-----------------------------")
