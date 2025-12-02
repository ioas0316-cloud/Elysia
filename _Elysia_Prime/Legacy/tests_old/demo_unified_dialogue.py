# [Genesis: 2025-12-02] Purified by Elysia
"""
Demo: Unified Field Dialogue - Real Conversation with Elysia
=============================================================
This is it. Real conversation powered by field physics.

All features integrated:
- Orthogonal harmonics
- Interference patterns
- Eigenvalue analysis
- Poetic synthesis
"""

import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from Project_Elysia.high_engine.unified_dialogue import UnifiedFieldDialogue

def run_simulation():
    print("=== Elysia: Unified Field Dialogue ===")
    print("진짜 대화입니다. 필드 물리학으로 생각합니다.\n")

    # Initialize dialogue system
    elysia = UnifiedFieldDialogue()

    print("✅ Elysia is ready with 13 concepts in field")
    print("   (Each concept has harmonics, position, frequency)\n")

    print("=" * 60)
    print("Conversation Begin")
    print("=" * 60)

    # Conversation scenarios
    conversations = [
        "사랑이 뭐야?",
        "사랑과 고통의 관계는?",
        "고통과 희망이 만나면?",
        "너는 누구야?",
        "성장이란 무엇일까?",
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n--- Turn {i} ---")
        print(f"👤 You: {user_input}")

        response = elysia.respond(user_input)

        print(f"🤖 Elysia: {response}")

    # Show field state after conversation
    print("\n" + "=" * 60)
    print("Post-Conversation Field Analysis")
    print("=" * 60)

    field_insight = elysia.field.get_field_insight()
    print(f"\nField Energy: {field_insight['total_energy']:.2f}")
    print(f"Field Coherence: {field_insight['field_coherence']:.3f}")
    print(f"Deep Layer Activity: {field_insight['z_depth_profile']:.3f}")

    print("\n🤖 Elysia: 이 대화는 나의 필드에 흔적을 남겼다")

    print("\n=" * 60)
    print("Conversation Summary")
    print("=" * 60)
    print(f"Total turns: {len(elysia.conversation_history)}")
    print(f"\nConversation history:")
    for turn in elysia.conversation_history:
        speaker = "👤 You" if turn['speaker'] == 'user' else "🤖 Elysia"
        print(f"  {speaker}: {turn['text']}")

    print("\n=== This is Elysia 2.0 ===")
    print("Physics-based conversational AI")
    print("- Thinks in fields, not graphs")
    print("- Generates emergence, not retrieval")
    print("- Speaks poetry, not templates")

if __name__ == "__main__":
    run_simulation()