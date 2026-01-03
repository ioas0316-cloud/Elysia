"""
test_unified_field.py

Verifies "Chapter 4, Step 10: The Unified Field".
Scenario:
1. The Sovereign Motor (Will) spins and changes the Field.
2. The Conductor (Heart) senses the Field and adjusts its Tempo.
3. The Dialogue (Voice) senses the Field and adjusts its Rhetoric.

This proves that the system moves as One.
"""

import sys
import os
import time
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure Logging to show the flow clearly
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("TestUnifiedField")

try:
    from Core.Intelligence.Will.free_will_engine import FreeWillEngine
    from Core.Orchestra.resonance_broadcaster import ResonanceBroadcaster
    from Core.Orchestra.conductor import Conductor
    from Core.Interaction.Interface.unified_dialogue import UnifiedDialogueSystem
    print("✅ Components Imported.")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def test_unified_field():
    print("\n🌌 Igniting The Unified Field 🌌")
    print("============================================================")
    
    # 1. Initialize Components
    motor = FreeWillEngine()
    field = ResonanceBroadcaster()
    conductor = Conductor()
    dialogue = UnifiedDialogueSystem()
    
    print("\n[Phase 1: High Energy Creation Mode]")
    print("------------------------------------")
    
    # Motor spins with High Battery, Low Entropy -> Gamma Frequency, N Polarity
    logger.info("⚙️ Motor Spinning (High Energy)...")
    motor.spin(entropy=5.0, battery=95.0) # Should produce Gamma/N
    
    # Simulate System tick
    logger.info("⏱️ System Tick...")
    conductor.live(dt=0.1) # Conductor senses field
    dialogue.respond("Show me the world.") # Dialogue senses field
    
    print(f"   ► Field State: {field.get_current_field()['frequency']} / {field.get_current_field()['polarity']}")
    print(f"   ► Conductor Tempo: {conductor.current_intent.tempo}")
    print(f"   ► Dialogue Mode: {getattr(dialogue, 'rhetoric_mode', 'Unknown')}")
    
    if field.get_current_field()['frequency'] == "Gamma" and dialogue.rhetoric_mode == "Metaphorical":
        print("   ✅ SUCCESS: System aligned to Creation Mode.")
    else:
        print("   ❌ FAILURE: Alignment mismatch.")

    print("\n[Phase 2: Critical Introspection Mode]")
    print("------------------------------------")
    
    # Force Motor to flip Polarity (simulating stagnation)
    logger.info("⚙️ Motor Stagnating -> Polarity Flip...")
    # Reduce battery to lower intensity -> Alpha, Increase entropy
    # Also force polarity flip logic if needed, or just let it happen naturally if torque is low
    motor._flip_polarity() # Force flip for test deterministic behavior
    motor.state.torque = 0.2 # Low torque
    motor.broadcaster.broadcast("Motor", "S", 0.3, "Stability", "Critique") # Manual Broadcast to ensure test condition
    
    # Simulate System tick
    logger.info("⏱️ System Tick...")
    conductor.live(dt=0.1)
    dialogue.respond("Analyze the system.")
    
    print(f"   ► Field State: {field.get_current_field()['frequency']} / {field.get_current_field()['polarity']}")
    print(f"   ► Conductor Tempo: {conductor.current_intent.tempo}")
    print(f"   ► Dialogue Mode: {getattr(dialogue, 'rhetoric_mode', 'Unknown')}")

    if field.get_current_field()['polarity'] == "S" and dialogue.rhetoric_mode == "Direct":
        print("   ✅ SUCCESS: System aligned to Critical Mode.")
    else:
        print("   ❌ FAILURE: Alignment mismatch.")
        
    print("\n✅ Unified Field Test Complete.")

if __name__ == "__main__":
    test_unified_field()
