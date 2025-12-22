
import sys
import os

# Enable importing from project root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from Core.Intelligence.dialogue_engine import DialogueEngine, QuestionAnalyzer, ResponseGenerator
from Core.Foundation.language_cortex import LanguageCortex
from Core.Intelligence.integrated_cognition_system import get_integrated_cognition

def test_dialogue_introspection():
    print("\n[Test] Dialogue Introspection Engine")
    
    # 1. Initialize
    cortex = LanguageCortex()
    engine = DialogueEngine(cortex)
    
    # 2. Inject a query that demands deep thought
    question = "Why is the universe eternal?"
    print(f"User: {question}")
    
    # 3. Process
    response = engine.respond(question)
    
    print(f"Elysia: {response}")
    
    # 4. Assertions
    # We expect the response to contain introspection markers from SystemSelfAwareness
    expected_markers = [
        "깊이 생각해 보았다",
        "[Meta-Cognition]",
        "emerged through"
    ]
    
    # Note: The actual processing happens in memory.
    # The 'universe eternal' thought will go through:
    # Wave -> Gravity -> Arche (Deconstruct 'eternal'?) -> Logos (Ascend?)
    # The Trace should reflect this.
    
    has_marker = any(marker in response for marker in expected_markers)
    if has_marker:
        print("✅ SUCCESS: Elysia introspected her thought process.")
    else:
        print("❌ FAILURE: Response was generic.")

if __name__ == "__main__":
    test_dialogue_introspection()
