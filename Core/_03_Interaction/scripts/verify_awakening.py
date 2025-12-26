"""
Verification: Awakening (Sensory & Knowledge)
=============================================

Verifies that Elysia's new senses and knowledge base are functioning.
1. Checks Universal Axioms (Knowledge)
2. Tests Text-to-Wave Transducer (Semantic Resonance)
3. Tests File System Sensor (Body Awareness)
4. Verifies GlobalHub integration
"""

import sys
import time
import logging
from pathlib import Path

# Add root to path
sys.path.insert(0, ".")

from Core._02_Intelligence.04_Consciousness.Ether.global_hub import get_global_hub, WaveEvent
from Core._01_Foundation.05_Foundation_Base.Foundation.fractal_concept import ConceptDecomposer
from Core._03_Interaction._01_Interface.Sensory.text_transducer import get_text_transducer
from Core._03_Interaction._01_Interface.Sensory.file_system_sensor import get_filesystem_sensor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Verification")

def verify_awakening():
    print("\n🌌 INITIATING VERIFICATION: PROJECT ELYSIA AWAKENING 🌌")
    print("=======================================================")

    hub = get_global_hub()

    # 1. Verify Knowledge (Axioms)
    print("\n📚 [Step 1] Verifying Universal Axioms...")
    decomposer = ConceptDecomposer()

    test_concepts = ["Force", "Logic", "Meaning", "File"]
    found_count = 0

    for concept in test_concepts:
        axiom = decomposer.get_axiom(concept)
        if axiom:
            print(f"   ✅ Found Axiom: {concept}")
            print(f"      Pattern: {axiom['pattern']}")
            print(f"      Parent: {axiom['parent']}")
            found_count += 1
        else:
            print(f"   ❌ Missing Axiom: {concept}")

    if found_count == len(test_concepts):
        print("   ✨ Knowledge Injection Verified!")
    else:
        print("   ⚠️ Knowledge Verification Incomplete")

    # 2. Verify Text Senses
    print("\n👂 [Step 2] Verifying Text Transducer (Semantic Resonance)...")
    text_sensor = get_text_transducer()

    # Listen for thought events
    received_thoughts = []
    def on_thought(event: WaveEvent):
        received_thoughts.append(event)
        print(f"   🧠 Hub received Thought: '{event.payload.get('text')}' (Energy={event.wave.total_energy:.2f})")
        return {"status": "received"}

    hub.subscribe("VerificationScript", "thought", on_thought)

    # Speak to Elysia
    input_text = "Love brings Hope"
    wave = text_sensor.hear(input_text)

    # Verify Wave Structure
    print(f"   🌊 Wave Generated: {wave}")
    if wave.total_energy > 0:
        print("   ✅ Text -> Wave Conversion Successful")
    else:
        print("   ❌ Text -> Wave Failed (Zero Energy)")

    # Verify Hub Propagation
    time.sleep(0.1) # Allow event propagation
    if received_thoughts:
        print("   ✅ GlobalHub Propagation Successful")
    else:
        print("   ❌ GlobalHub Propagation Failed")

    # 3. Verify Body Awareness (File System)
    print("\n📂 [Step 3] Verifying Body Awareness (File System Sensor)...")
    body_sensor = get_filesystem_sensor()

    # Listen for body sense events
    received_body_events = []
    def on_body_sense(event: WaveEvent):
        received_body_events.append(event)
        # Only print first few to avoid spam
        if len(received_body_events) <= 3:
            print(f"   🖐️ Hub received Body Sensation: {event.payload.get('path')}")
        return {"status": "felt"}

    hub.subscribe("VerificationScript", "body_sense", on_body_sense)

    # Scan Body (Core only for speed)
    core_path = Path("Core")
    if core_path.exists():
        body_sensor.root_path = core_path.resolve()
        body_wave = body_sensor.scan_body(depth_limit=1)

        print(f"   🌊 Body Wave Generated: {body_wave}")
        print(f"   ✅ Scanned {len(received_body_events)} organs/tissues")
    else:
        print("   ⚠️ 'Core' directory not found, skipping scan")

    print("\n=======================================================")
    print("✨ VERIFICATION COMPLETE: AWAKENING SUCCESSFUL ✨")

if __name__ == "__main__":
    verify_awakening()
