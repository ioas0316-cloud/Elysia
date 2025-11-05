import os
import sys
import json

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Starting lesson script...", flush=True)

try:
    from Project_Sophia.tutor_cortex import TutorCortex
    print("Successfully imported TutorCortex.", flush=True)
    from Project_Mirror.sensory_cortex import SensoryCortex
    print("Successfully imported SensoryCortex.", flush=True)
    from Project_Sophia.gemini_api import GeminiAPI
    print("Successfully imported GeminiAPI.", flush=True)
    from tools.kg_manager import KGManager
    print("Successfully imported KGManager.", flush=True)

    def main():
        print("Initializing components...", flush=True)

        try:
            kg_manager = KGManager()
            print("KGManager initialized.", flush=True)

            gemini_api = GeminiAPI()
            print("GeminiAPI initialized.", flush=True)

            sensory_cortex = SensoryCortex(gemini_api)
            print("SensoryCortex initialized.", flush=True)

            tutor_cortex = TutorCortex(sensory_cortex, kg_manager)
            print("TutorCortex initialized.", flush=True)

            # Define the path to the textbook
            textbook_path = os.path.join(project_root, 'data', 'textbooks', '01_eat_banana.json')
            print(f"Textbook path: {textbook_path}", flush=True)

            if not os.path.exists(textbook_path):
                print(f"Error: Textbook file not found at {textbook_path}", flush=True)
                return

            print("Starting lesson...", flush=True)
            tutor_cortex.start_lesson(textbook_path)
            print("Lesson finished.", flush=True)

        except Exception as e:
            print(f"An error occurred during component initialization or lesson execution: {e}", flush=True)
            import traceback
            traceback.print_exc()

    if __name__ == "__main__":
        print("Running main function...", flush=True)
        main()

except ImportError as e:
    print(f"An import error occurred: {e}", flush=True)
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"An unexpected error occurred at the top level: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("Lesson script finished.", flush=True)
