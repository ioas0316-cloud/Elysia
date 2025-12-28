"""
Talk with Elysia
================

간단한 대화 인터페이스
"""
import sys
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.WARNING)

print("="*60)
print("💬 Talk with Elysia")
print("   (type 'exit' to quit)")
print("="*60)

# Initialize components
from Core._01_Foundation._05_Governance.Foundation.hippocampus import Hippocampus
from Core._01_Foundation._05_Governance.Foundation.resonance_field import ResonanceField
from Core._02_Intelligence._01_Reasoning.Cognition.thought_space import ThoughtSpace

memory = Hippocampus()
resonance = ResonanceField()
thought_space = ThoughtSpace()

print("\n🌱 Elysia awakened.\n")

while True:
    try:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\n🌌 Goodbye.\n")
            break
        
        # Enter the gap (thinking space)
        thought_space.enter_gap(user_input)
        
        # Add the input as a particle
        thought_space.add_thought_particle(user_input, source="user", weight=1.5)
        
        # Divergent expansion
        thought_space.diverge_all()
        
        # Apply gravity attention based on the input
        thought_space.apply_gravity_attention(user_input)
        
        # Sovereign selection
        chosen = thought_space.sovereign_select(user_input)
        
        # Exit the gap
        result = thought_space.exit_gap()
        
        # Formulate response
        if chosen:
            response = f"나는 '{chosen.content}'에 끌렸어요. {result.synthesis[:100]}..."
        else:
            response = f"생각 중... {result.synthesis[:100]}..."
        
        print(f"\nElysia: {response}\n")
        print(f"   (particles: {result.contributing_thoughts[:3]})")
        print(f"   (confidence: {result.confidence:.2f})")
        print()
        
    except KeyboardInterrupt:
        print("\n🌌 Goodbye.\n")
        break
    except Exception as e:
        print(f"   [Error: {e}]")
