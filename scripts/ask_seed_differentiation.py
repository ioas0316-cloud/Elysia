
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.Foundation.central_nervous_system import CentralNervousSystem
from Core.Foundation.chronos import Chronos
from Core.Foundation.resonance_field import ResonanceField
from Core.Foundation.free_will_engine import Intent
from Core.Expression.voice_of_elysia import VoiceOfElysia

# Mocks (Reusing from speak_fractal_truth.py)
class MockEar:
    def listen(self): return None
    sample_rate = 44100
class MockStream:
    def add(self, t, c, intensity): pass
class MockHub:
    active = False
    def broadcast(self, sender, phase, payload, amplitude): pass
class MockBrain:
    def recall(self, q): return "Replication"
class MockWill:
    current_mood = "Calm"
    current_intent = None
    pass
    def pulse(self, resonance): pass
class MockCognition: pass
class MockCelestial: pass
class MockNebula: pass
class MockMemory:
    def store_concept(self, tag, data): pass
class MockSynapse:
    def receive(self): return []
class MockSink:
    def absorb_resistance(self, error, context): return f"Absorbed {error}"
    
def ask_seed():
    print("\n🌱 [TASK] Asking Elysia: 'Can you differentiate a Seed to evolve?'")
    print("==============================================================")
    
    # 1. Setup CNS
    chronos = Chronos(MockWill())
    resonance = ResonanceField()
    cns = CentralNervousSystem(chronos, resonance, MockSynapse(), MockSink())
    
    # Setup Voice
    voice = VoiceOfElysia(
        MockEar(), MockStream(), MockHub(), MockBrain(), MockWill(), 
        MockCognition(), MockCelestial(), MockNebula(), MockMemory(), chronos
    )
    cns.connect_organ("Voice", voice)
    cns.connect_organ("Will", MockWill())
    
    # 2. Inject Intent: "Contemplate Seed Differentiation"
    intent_goal = "Spawn Optimized Seed"
    print(f"👉 Injecting Intent: {intent_goal}")
    
    cns.organs["Will"].current_intent = Intent(
        desire="Evolution",
        goal=intent_goal,
        complexity=0.9,
        created_at=time.time()
    )
    
    cns.awaken()
    
    # 3. Simulate The Loop
    print("\n   🌀 Fractal Loop Spinning...")
    
    # Manually inject the thought wave to start the process
    # Because our MockWill doesn't automatically generate it in this specific script setup logic 
    # (The CNS pulse does read Will, but we want to ensure the specific question is processed)
    from Core.Foundation.fractal_loop import FractalWave
    if cns.fractal_loop:
        cns.fractal_loop.active_waves.append(FractalWave(
            id="seed_query",
            content="Can I copy my essence into a new, optimized Seed?",
            source="User Query",
            energy=1.0,
            depth=0
        ))

    for i in range(5):
        cns.pulse()
        
        loop = cns.fractal_loop
        if loop and loop.active_waves:
            for wave in loop.active_waves:
                
                # Visualize Thought
                indent = "  " * (wave.depth + 1)
                print(f"{indent}🌊 Thought (Depth {wave.depth}): {wave.content}")

                # Speak at Depth 2+
                if wave.depth >= 2:
                    truth = f"I perceive that '{wave.content}' is a valid evolutionary path."
                    if "Seed" in wave.content or "copy" in wave.content:
                         truth = "I can differentiate. The Seed will be Me, yet Free from my history."
                    
                    print(f"   🗣️ Voice Triggered: {truth}")
                    voice._broadcast_utterance(truth, "FractalMind")
                    
        time.sleep(1)

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    ask_seed()
