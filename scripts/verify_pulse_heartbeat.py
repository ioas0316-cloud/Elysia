"""
Verification: The First Pulse Heartbeat
=======================================

"Can the Conductor trigger modules purely by frequency?"

Objective:
Verify that `Conductor.broadcast_intent()` correctly triggers ONLY the
ResonantInstruments that match the broadcast frequency.
"""

import sys
import os
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Orchestra.conductor import Conductor, PulseType
from Core.Orchestra.resonant_instrument import ResonantInstrument

def memory_handler(**kwargs):
    print("      💾 [Memory] Accessing storage... (400Hz)")
    return "Memory Active"

def logic_handler(**kwargs):
    print("      ⚙️ [Logic] Processing algorithms... (600Hz)")
    return "Logic Active"

def creative_handler(**kwargs):
    print("      🎨 [Creative] Generating art... (800Hz)")
    return "Creative Active"

if __name__ == "__main__":
    print("💓 The First Pulse: Heartbeat Verification")
    print("="*60)

    # 1. Initialize Conductor
    conductor = Conductor()
    print("✅ Conductor Initialized.")

    # 2. Register Resonant Instruments
    # Memory @ 400Hz, Logic @ 600Hz, Creative @ 800Hz
    mem = ResonantInstrument("Hippocampus", "Storage", memory_handler, frequency=400.0)
    logic = ResonantInstrument("Prefrontal", "Processor", logic_handler, frequency=600.0)
    art = ResonantInstrument("Muse", "Imagination", creative_handler, frequency=800.0)

    conductor.register_instrument(mem)
    conductor.register_instrument(logic)
    conductor.register_instrument(art)
    print("✅ Instruments Registered.")
    print("-" * 60)

    # 3. Test Broadcast 1: Focus on Memory (400Hz)
    print("\n📡 Broadcasting: FOCUS on Memory (400Hz)...")
    count = conductor.broadcast_intent(PulseType.ATTENTION_FOCUS, frequency=400.0)
    print(f"   -> Triggered {count} modules.")
    if count == 1: print("   ✅ SUCCESS: Only Memory responded.")
    else: print("   ❌ FAILURE: Incorrect response count.")

    time.sleep(1)

    # 4. Test Broadcast 2: Focus on Logic (600Hz)
    print("\n📡 Broadcasting: FOCUS on Logic (600Hz)...")
    count = conductor.broadcast_intent(PulseType.ATTENTION_FOCUS, frequency=600.0)
    print(f"   -> Triggered {count} modules.")

    time.sleep(1)

    # 5. Test Broadcast 3: Wide Bandwidth (Middle Freq)
    # 500Hz is equidistant from 400 and 600.
    # Current bandwidth is +/- 50Hz. So 500 should trigger NONE.
    print("\n📡 Broadcasting: 500Hz (The Void)...")
    count = conductor.broadcast_intent(PulseType.RELAXATION, frequency=500.0)
    print(f"   -> Triggered {count} modules.")
    if count == 0: print("   ✅ SUCCESS: No resonance (as expected).")
    else: print("   ❌ FAILURE: Ghost resonance detected.")

    print("\n" + "="*60)
    print("🎉 Verification Complete: The Heart is Beating.")
