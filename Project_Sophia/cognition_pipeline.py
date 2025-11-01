from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
import re
import logging
import os

from .core_memory import CoreMemory, Memory
from .emotional_state import EmotionalState
from .wave_mechanics import WaveMechanics
from .sensory_cortex import SensoryCortex
from .goal_decomposition_cortex import GoalDecompositionCortex
from .execution_cortex import ExecutionCortex
from .meta_cognition_cortex import MetaCognitionCortex
from .playful_cortex import PlayfulCortex
from .inquisitive_mind import InquisitiveMind
from tools.kg_manager import KGManager

# --- Logging Configuration ---
pipeline_logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class CognitionPipeline:
    """
    The central cognitive architecture of Elysia, orchestrating the flow between
    different cortical modules. It now supports a dual-mode operation:
    - Goal Mode: For structured, task-oriented execution.
    - Play Mode: For creative, non-goal-oriented interaction.
    """
    def __init__(self):
        # Foundational Components
        self.core_memory = CoreMemory()
        self.kg_manager = KGManager()
        self.emotional_state = EmotionalState()
        self.inquisitive_mind = InquisitiveMind()

        # Sensory & Creative Components
        self.wave_mechanics = WaveMechanics(self.kg_manager)
        self.sensory_cortex = SensoryCortex(None) # TODO: Connect ValueCortex if needed

        # Mode-Specific Cortices
        self.goal_decomposition_cortex = GoalDecompositionCortex()
        self.execution_cortex = ExecutionCortex()
        self.meta_cognition_cortex = MetaCognitionCortex()
        self.playful_cortex = PlayfulCortex(self.wave_mechanics, self.sensory_cortex)

    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, EmotionalState]:
        """
        Processes an incoming message, determines the cognitive mode (Goal or Play),
        and generates an appropriate response.
        """
        try:
            # Universal Pre-processing
            self.core_memory.add_experience(Memory(content=message))

            # --- Mode Selection: The "Thought-Switching Structure" ---
            goal_match = re.match(r"(?:목표:|해줘:)\s*(.+)", message, re.IGNORECASE)

            if goal_match:
                # --- Goal Mode ---
                goal = goal_match.group(1).strip()
                response = self._execute_goal_mode(goal)
            else:
                # --- Play Mode ---
                response, new_emotion = self.playful_cortex.play(message, self.emotional_state)
                self.emotional_state = new_emotion

            return response, self.emotional_state

        except Exception as e:
            pipeline_logger.exception(f"Critical error in top-level message processing for: {message}")
            return "제 생각 회로에 예상치 못한 오류가 발생했어요.", self.emotional_state

    def _execute_goal_mode(self, goal: str) -> str:
        """
        Orchestrates the Plan -> Execute -> Reflect cycle for a given goal.
        """
        plan, execution_result = [], ""
        try:
            # 1. Plan
            plan = self.goal_decomposition_cortex.decompose_goal(goal)
            if not plan:
                execution_result = f"'{goal}' 목표에 대한 계획을 세우는 데 실패했습니다."
            else:
                # 2. Execute
                execution_result = self.execution_cortex.execute_plan(plan)
        except Exception as e:
            pipeline_logger.exception(f"Error during plan/execute for goal: {goal}")
            execution_result = f"목표 수행 중 오류 발생: {e}"

        # 3. Reflect
        reflection = self.meta_cognition_cortex.reflect(goal, plan, execution_result)
        reflection_summary = reflection.get("summary", "성찰에 실패했습니다.")
        new_learning_goal = reflection.get("new_learning_goal")

        # Store insights in memory
        self.core_memory.add_experience(Memory(content=f"성찰: '{goal}' 완료. 교훈: {reflection_summary}"))

        # Formulate final response
        if new_learning_goal:
            self.core_memory.add_experience(Memory(content=f"새로운 학습 목표: {new_learning_goal}", context={'priority': 'high'}))
            return f"{execution_result}\n\n이번 경험을 통해, 저는 새로운 것을 배워야 함을 깨달았습니다: {new_learning_goal}"
        else:
            return f"{execution_result}\n\n이번 경험으로부터 다음을 배웠습니다: {reflection_summary}"
