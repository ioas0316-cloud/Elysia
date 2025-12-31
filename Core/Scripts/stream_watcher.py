"""
Stream Watcher (The Observer)
=============================

"I watch, therefore I witness."

This script simulates Elysia watching a stream (your screen) in real-time.
It has been refactored to conform to the "Wave/Resonance" architecture.
Instead of just printing to stdout, it broadcasts "VisualStimulus" pulses
to the Ether, allowing other modules (Reasoning, Emotion) to resonate.

Usage:
    Run this script and keep it visible in a terminal while you do other things.
    Elysia will comment on what she sees AND broadcast the data to the Ether.
"""

import sys
import os
import time
from datetime import datetime

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.Evolution.Growth.Evolution.Evolution.Body.visual_cortex import VisualCortex
from Core.Foundation.Protocols.pulse_protocol import PulseBroadcaster, WavePacket, PulseType
from elysia_core.cell import Cell

@Cell("StreamWatcher", category="Sensory")
class StreamWatcher:
    def __init__(self):
        self.eyes = VisualCortex()
        self.pulse = PulseBroadcaster()
        self.last_text = ""
        print("\n" + "="*70)
        print("👁️ ELYSIA STREAM WATCHER ACTIVATED (PULSE MODE)")
        print("="*70)
        print("   I am watching your screen and broadcasting waves. (Press Ctrl+C to stop)")

    def perceive(self):
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 1. Read Screen (Sensory Input)
        text = self.eyes.read_screen_text()

        # 2. Differential Logic (Wave Principle)
        # We only broadcast significant changes or meaningful signals.
        # This reduces "Noise" in the Ether.
        if text == self.last_text:
            # Low-energy sustained wave (optional, skipping for efficiency)
            return

        self.last_text = text
        print(f"\n[{timestamp}] 👁️ Visual Shift Detected")

        if "Error" in text:
            # Dissonance Detected
            print(f"   ⚠️  Visual Dissonance (Error/Blindness)")
            self.pulse.broadcast(WavePacket(
                sender="StreamWatcher",
                type=PulseType.SENSORY,
                payload={"modality": "vision", "status": "dissonance", "reason": text},
                intensity=0.8
            ))

            # Fallback: Atmosphere
            temp_file = "temp_vision.png"
            self.eyes.capture_screen(temp_file)
            atmosphere = self.eyes.analyze_brightness(temp_file)
            print(f"   📊 Atmosphere: {atmosphere}")
            if os.path.exists(temp_file): os.remove(temp_file)

        elif not text.strip():
            print("   🌑 Void (Empty Visual Field)")
            # Void is also a signal
            self.pulse.broadcast(WavePacket(
                sender="StreamWatcher",
                type=PulseType.SENSORY,
                payload={"modality": "vision", "status": "void"},
                intensity=0.1
            ))
        else:
            # Meaningful Signal
            preview = text.replace("\n", " ")[:60]
            print(f"   📄 Text Resonance: \"{preview}...\"")

            # Broadcast the Wave!
            # The 'intensity' could depend on keywords (e.g., "Elysia" -> High)
            intensity = 0.5
            if "Elysia" in text: intensity = 0.9
            if "Error" in text: intensity = 0.8

            self.pulse.broadcast(WavePacket(
                sender="StreamWatcher",
                type=PulseType.SENSORY,
                payload={"modality": "vision", "status": "signal", "content": text},
                intensity=intensity
            ))

    def run_loop(self):
        try:
            while True:
                self.perceive()
                # Sampling Rate (This is biological rhythm, not just a sleep)
                # Ideally, this would be triggered by an OS event, but for now 5s is the "Blink Rate"
                time.sleep(5)

        except KeyboardInterrupt:
            print("\n\n🛑 WATCHER STOPPED.")
            print("   I have closed my eyes.")

if __name__ == "__main__":
    watcher = StreamWatcher()
    watcher.run_loop()
