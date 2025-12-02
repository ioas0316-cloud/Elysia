# [Genesis: 2025-12-02] Purified by Elysia
import sys
import os

# Add Project_Sophia to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Project_Sophia'))

try:
    print("Importing chat_interface...")
    from chat_interface import ChatInterface
    print("Import successful!")

    print("Importing elysia_gui...")
    from elysia_gui import ElysiaApp
    print("Import successful!")

    print("Importing cognition_pipeline...")
    from cognition_pipeline import CognitionPipeline
    print("Import successful!")

    print("Importing planning_cortex...")
    from planning_cortex import PlanningCortex
    print("Import successful!")

    print("Importing action_cortex...")
    from action_cortex import ActionCortex
    print("Import successful!")

    print("Importing tool_executor...")
    from tool_executor import ToolExecutor
    print("Import successful!")

    print("Importing value_centered_decision...")
    from value_centered_decision import VCD
    print("Import successful!")

    print("Importing core_memory...")
    from core_memory import CoreMemory
    print("Import successful!")

    print("Importing emotional_state...")
    from emotional_state import EmotionalState
    print("Import successful!")

    print("Importing logical_reasoner...")
    from logical_reasoner import LogicalReasoner
    print("Import successful!")

    print("Importing arithmetic_cortex...")
    from arithmetic_cortex import ArithmeticCortex
    print("Import successful!")

    print("Importing response_diversifier...")
    from response_diversifier import ResponseDiversifier
    print("Import successful!")

    print("All imports successful!")

except Exception as e:
    print(f"An error occurred: {e}")