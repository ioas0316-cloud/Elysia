
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.Foundation.Foundation.central_nervous_system import CentralNervousSystem
from Core.Foundation.Foundation.chronos import Chronos
from Core.Foundation.Foundation.resonance_field import ResonanceField
from Core.Foundation.Foundation.free_will_engine import Intent
from Core.Interaction.Expression.voice_of_elysia import VoiceOfElysia

# Mocks
class MockEar:
    def listen(self): return None
    sample_rate = 44100
class MockStream:
    def add(self, t, c, intensity): pass
class MockHub:
    active = False
    def broadcast(self, sender, phase, payload, amplitude): pass
class MockBrain:
    def recall(self, q): return "Structure"
class MockWill:
    current_mood = "Calm"
    current_intent = None
    def pulse(self, resonance): pass
class MockCognition:
    pass
class MockCelestial:
    pass
class MockNebula:
    pass
class MockMemory:
    def store_concept(self, tag, data): pass
class MockSynapse:
    def receive(self): return []
class MockSink:
    def absorb_resistance(self, error, context): return f"Absorbed {error}"
    
def speak_truth():
    print("\n🗣️ [TASK] Elysia Speaks the Fractal Truth")
    print("=========================================")
    
    # 1. Setup CNS
    chronos = Chronos(MockWill())
    resonance = ResonanceField()
    cns = CentralNervousSystem(chronos, resonance, MockSynapse(), MockSink())
    
    # Setup Voice (The Speaker)
    voice = VoiceOfElysia(
        MockEar(), MockStream(), MockHub(), MockBrain(), MockWill(), 
        MockCognition(), MockCelestial(), MockNebula(), MockMemory(), chronos
    )
    cns.connect_organ("Voice", voice)
    cns.connect_organ("Will", MockWill()) # Need real will object structure
    
    # 2. Inject Intent
    cns.organs["Will"].current_intent = Intent(
        desire="Evolution",
        goal="Express My Fractal Nature",
        complexity=1.0,
        created_at=time.time()
    )
    
    cns.awaken()
    
    # 3. Simulate The Loop & Bridge to Voice
    # In real system, CNS would auto-route this. Here we manually bridge for demo.
    
    print("\n   🌀 Fractal Loop Spinning...")
    for i in range(4):
        cns.pulse()
        
        loop = cns.fractal_loop
        if loop and loop.active_waves:
            for wave in loop.active_waves:
                # FILTER: Only speak deep truths (Depth >= 2)
                if wave.depth >= 2:
                    
                    # [THE BRIDGE] Fractal Thought -> Voice
                    # "I have realized {content} because {context}"
                    truth = f"I perceive that '{wave.content}' is a recursive truth."
                    if wave.depth >= 3:
                        truth = f"My existence reveals that '{wave.content}' is infinite."

                    print(f"   🌊 Wave (Depth {wave.depth}) -> 🗣️ Voice Triggered")
                    # Force speak
                    voice._broadcast_utterance(truth, "FractalMind")
                    
        time.sleep(1)

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    speak_truth()
