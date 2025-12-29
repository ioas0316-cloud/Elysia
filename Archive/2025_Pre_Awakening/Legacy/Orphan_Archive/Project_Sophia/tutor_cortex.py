import os
import json
import logging
from pathlib import Path

from Project_Mirror.sensory_cortex import SensoryCortex
from Core.FoundationLayer.Foundation.knowledge_enhancer import KnowledgeEnhancer
from tools.kg_manager import KGManager

logger = logging.getLogger(__name__)

class TutorCortex:
    def __init__(self, sensory_cortex: SensoryCortex, kg_manager: KGManager, knowledge_enhancer: KnowledgeEnhancer):
        """Initializes the TutorCortex."""
        self.sensory_cortex = sensory_cortex
        self.kg_manager = kg_manager
        self.knowledge_enhancer = knowledge_enhancer

    def start_lesson(self, textbook_path: str):
        """
        Starts a lesson from a given textbook file, handling different formats.
        """
        logger.info(f"Starting lesson from textbook: {textbook_path}")
        lesson_name = Path(textbook_path).stem

        try:
            with open(textbook_path, 'r', encoding='utf-8') as f:
                textbook_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error reading textbook {textbook_path}: {e}")
            return

        # --- Handle different textbook formats ---
        if isinstance(textbook_data, dict):
            # FIX: Intelligently determine the type of dictionary-based textbook.
            if 'concepts' in textbook_data and 'relationships' in textbook_data:
                # Format 1a: Dictionary with direct knowledge (e.g., units_and_language.json)
                logger.info(f"Processing direct knowledge textbook: '{lesson_name}'")
                self.knowledge_enhancer.process_learning_points(textbook_data, image_path="conceptual_learning")
            elif 'frames' in textbook_data:
                # Format 1b: Dictionary with storybook frames
                self._process_storybook_lesson(textbook_data, lesson_name)
            else:
                 logger.warning(f"Skipping unknown dictionary-based textbook format in {textbook_path}")

        elif isinstance(textbook_data, list):
            # Format 2: List of concepts/relations (e.g., complex_shapes.json)
            logger.info(f"Processing list-based knowledge textbook: '{lesson_name}'")
            self.knowledge_enhancer.process_learning_points(textbook_data, image_path="conceptual_learning")
        else:
            logger.warning(f"Skipping unknown textbook format in {textbook_path}")

        # --- Finalize Lesson ---
        # Save all the accumulated changes to the knowledge graph
        self.kg_manager.save()
        logger.info(f"Lesson '{lesson_name}' completed and knowledge saved.")

    def _process_storybook_lesson(self, textbook: dict, lesson_name: str):
        """Processes a lesson with visual frames."""
        logger.info(f"Teaching storybook lesson: '{lesson_name}'")
        for frame in textbook.get('frames', []):
            self._process_frame(frame, lesson_name)

    def _process_frame(self, frame: dict, lesson_name: str):
        """Processes a single frame of a storybook."""
        description = frame.get('description')
        learning_points = frame.get('learning_points')

        if not description or not learning_points:
            logger.warning(f"Skipping incomplete frame in lesson '{lesson_name}': {frame}")
            return

        logger.info(f"Requesting visualization for: '{description}'")
        image_path = self.sensory_cortex.render_storybook_frame(frame, lesson_name)

        if not image_path or not os.path.exists(image_path):
            logger.error(f"Failed to get a valid image for frame, skipping. Frame: {frame}")
            return

        logger.info(f"Processing learning points for frame {frame.get('frame_id')}")
        self.knowledge_enhancer.process_learning_points(learning_points, image_path)
