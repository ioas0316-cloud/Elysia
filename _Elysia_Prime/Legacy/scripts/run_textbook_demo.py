# [Genesis: 2025-12-02] Purified by Elysia
"""
Run a TutorCortex demo using the sample textbooks.

Usage:
  python -m scripts.run_textbook_demo --book data/textbooks/nlp_basics.json
  python -m scripts.run_textbook_demo --book data/textbooks/math_basics.json
"""
from __future__ import annotations

import argparse
from Project_Mirror.sensory_cortex import SensoryCortex
from Project_Sophia.tutor_cortex import TutorCortex
from Project_Sophia.knowledge_enhancer import KnowledgeEnhancer
from Project_Sophia.value_cortex import ValueCortex
from tools.kg_manager import KGManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", required=True, help="Path to textbook JSON")
    args = parser.parse_args()

    # Mirror/Sophia wiring for the lesson
    value = ValueCortex()
    sensory = SensoryCortex(value_cortex=value, telemetry=None)
    kg = KGManager()
    ke = KnowledgeEnhancer(kg)
    tutor = TutorCortex(sensory_cortex=sensory, kg_manager=kg, knowledge_enhancer=ke, cognition_pipeline=None)

    tutor.start_lesson(args.book)
    print("Lesson complete. KG summary:", kg.get_summary())


if __name__ == "__main__":
    main()
