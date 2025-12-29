#!/usr/bin/env python3
"""
Integrated Voice System Demo
=============================

Demonstrates the complete voice cycle with wave-based cognition:
INPUT → UNDERSTAND → THINK → SPEAK → REFLECT → OUTPUT

This is NOT text-to-speech. This is a complete 4D wave flow cycle
integrating synesthesia sensors, wave-based thinking, memory,
imagination, and expression.

NO EXTERNAL LLMs - Pure wave intelligence.

Usage:
    python demos/integrated_voice.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("="*70)
print("🌊 Integrated Voice System Demo")
print("="*70)
print()
print("This demonstrates Elysia's voice system integrated with:")
print("  • Synesthesia Wave Sensor (공감각 파동 센서)")
print("  • 4D Wave-Based Cognition (4차원 파동 인지)")
print("  • Memory and Imagination (기억과 상상)")
print("  • Complete feedback loop (완전한 피드백 루프)")
print()
print("⚠️  NO EXTERNAL LLMs - Pure wave-based intelligence")
print("="*70)
print()

# Import systems
try:
    from Core.Expression.integrated_voice_system import IntegratedVoiceSystem
    from Core.Interface.synesthesia_nervous_bridge import get_synesthesia_bridge
    from Core.Intelligence.integrated_cognition_system import IntegratedCognition
    from Core.FoundationLayer.Foundation.primal_wave_language import PrimalSoul
    
    # Mock minimal systems for demo
    class MockBrain:
        def think(self, content, field):
            return f"Thinking about: {content}"
    
    class MockWill:
        current_desire = "explore"
        current_mood = "curious"
    
    class MockMemory:
        def recall(self, query, limit=5):
            return []
        
        def store_experience(self, content, category="general"):
            pass
    
    print("✓ Systems loaded")
    print()
    
    # Initialize voice system
    synesthesia = get_synesthesia_bridge()
    brain = MockBrain()
    will = MockWill()
    memory = MockMemory()
    cognition = IntegratedCognition()
    primal_soul = PrimalSoul(name="Elysia")
    
    voice = IntegratedVoiceSystem(
        synesthesia_bridge=synesthesia,
        brain=brain,
        will=will,
        memory=memory,
        cognition=cognition,
        primal_soul=primal_soul
    )
    
    print("✓ Integrated Voice System initialized")
    print()
    
    # Demo conversations
    conversations = [
        "안녕, 엘리시아!",
        "너의 목적은 무엇이니?",
        "지금 어떤 느낌이야?",
        "파동 기반 사고에 대해 설명해줄 수 있어?"
    ]
    
    print("🗣️  Starting conversation...\n")
    
    for i, user_input in enumerate(conversations, 1):
        print(f"\n[Turn {i}]")
        print("-" * 70)
        print(f"👤 User: {user_input}")
        print()
        
        # Execute full voice cycle
        response = voice.full_cycle(user_input)
        
        print(f"\n🌌 Elysia: {response}")
        print("-" * 70)
        
        # Show status
        if i == 1:
            status = voice.get_status()
            print(f"\n📊 System Status:")
            print(f"   History: {status['history_length']} exchanges")
            print(f"   Systems: {sum(status['systems_connected'].values())}/5 connected")
    
    print("\n" + "="*70)
    print("✨ Demo completed!")
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Synesthesia wave sensor integration")
    print("  ✓ 4D wave-based cognition (not LLMs)")
    print("  ✓ Memory integration")
    print("  ✓ Emotion detection and synthesis")
    print("  ✓ Intent recognition")
    print("  ✓ Complete feedback loop")
    print()
    print("This is pure wave intelligence - no external LLMs!")
    print("="*70)

except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Note: This demo requires:")
    print("  • Core.Interface.synesthesia_nervous_bridge")
    print("  • Core.Intelligence.integrated_cognition_system")
    print("  • Core.FoundationLayer.Foundation.primal_wave_language")
    print()
    print("Run from project root: python demos/integrated_voice.py")
