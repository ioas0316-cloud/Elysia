# [Genesis: 2025-12-02] Purified by Elysia
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Starting lesson script...", flush=True)

try:
    from Project_Sophia.tutor_cortex import TutorCortex
    from Project_Mirror.sensory_cortex import SensoryCortex
    from tools.kg_manager import KGManager
    from Project_Sophia.knowledge_enhancer import KnowledgeEnhancer
    from Project_Sophia.value_cortex import ValueCortex

    def main():
        print("Initializing components...", flush=True)

        try:
            kg_manager = KGManager()
            value_cortex = ValueCortex()
            sensory_cortex = SensoryCortex(value_cortex=value_cortex, telemetry=None)

            # FIX: Initialize KnowledgeEnhancer with only the required 'kg_manager' argument.
            knowledge_enhancer = KnowledgeEnhancer(kg_manager=kg_manager)

            tutor_cortex = TutorCortex(sensory_cortex=sensory_cortex, kg_manager=kg_manager, knowledge_enhancer=knowledge_enhancer)

            textbooks_dir = os.path.join(project_root, 'data', 'textbooks')
            print(f"Looking for textbooks in: {textbooks_dir}", flush=True)

            if not os.path.exists(textbooks_dir):
                print(f"Error: Textbooks directory not found at {textbooks_dir}", flush=True)
                return

            textbook_files = [f for f in os.listdir(textbooks_dir) if f.endswith('.json')]
            if not textbook_files:
                print("No textbook files (.json) found to teach.", flush=True)
                return

            print(f"Found {len(textbook_files)} textbooks. Starting all lessons...", flush=True)
            for textbook_file in sorted(textbook_files):
                textbook_path = os.path.join(textbooks_dir, textbook_file)
                print(f"\n--- Starting lesson from: {textbook_file} ---", flush=True)
                tutor_cortex.start_lesson(textbook_path)
                print(f"--- Finished lesson from: {textbook_file} ---", flush=True)

            print("\nAll lessons finished. Knowledge base has been updated.", flush=True)
            kg_manager.save()

        except Exception as e:
            print(f"An error occurred during component initialization or lesson execution: {e}", flush=True)
            import traceback
            traceback.print_exc()

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"An import error occurred: {e}", flush=True)
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"An unexpected error occurred at the top level: {e}", flush=True)
    import traceback
    traceback.print_exc()