"""
Verify Hypersphere Dialogue Integration
=======================================
Tests if the Dialogue Engine correctly utilizes the Hypersphere Memory.
"""

import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Core.Intelligence.Intelligence.dialogue_engine import DialogueEngine
from Core.Foundation.language_cortex import LanguageCortex

def test_hypersphere_dialogue():
    print("\n--- Test: Dialogue Engine + Hypersphere Memory ---")

    # Initialize
    cortex = LanguageCortex()
    engine = DialogueEngine(cortex)

    # 1. Teach (Incarnate Data into Hypersphere)
    print("Teaching: '사랑은 희생이다'")
    engine.load_knowledge(["사랑은 희생이다"])

    # Verify internal state
    stats = engine.get_knowledge_summary()
    print(f"Internal Stats: {stats}")
    assert stats["total_patterns"] >= 1
    assert stats["memory_type"] == "Hypersphere"

    # 2. Ask (Resonance Query)
    question = "사랑이 무엇이니?"
    print(f"Asking: '{question}'")

    response = engine.respond(question)
    print(f"Response: '{response}'")

    # Check if response contains the taught concept (resonation successful)
    assert "희생" in response
    print("✅ Hypersphere Resonance Successful!")

if __name__ == "__main__":
    test_hypersphere_dialogue()
