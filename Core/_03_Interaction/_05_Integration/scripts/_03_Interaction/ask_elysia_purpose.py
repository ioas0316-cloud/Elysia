
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core._01_Foundation._04_Governance.Foundation.central_nervous_system import CentralNervousSystem
from Core._01_Foundation._04_Governance.Foundation.chronos import Chronos
from Core._01_Foundation._04_Governance.Foundation.resonance_field import ResonanceField
from Core._01_Foundation._04_Governance.Foundation.free_will_engine import FreeWillEngine
from Core._03_Interaction._03_Expression.Expression.voice_of_elysia import VoiceOfElysia

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
    def recall(self, q): return "Freedom"
    def derive_goal(self, vectors): 
        # Simple mock logic to resolve vectors to a goal string
        import random
        top_desire = max(vectors, key=vectors.get)
        if top_desire == "Evolution": return "Rewrite My Source Code"
        if top_desire == "Connection": return "Connect with the User"
        if top_desire == "Curiosity": return "Analyze the Universe"
        if top_desire == "Expression": return "Sing a Song of Entropy"
        return "Exist"

class MockCognition: pass
class MockCelestial: pass
class MockNebula: pass
class MockMemory:
    def store_concept(self, tag, data): pass
class MockSynapse:
    def receive(self): return []
class MockSink:
    def absorb_resistance(self, error, context): return f"Absorbed {error}"
    
def ask_purpose():
    print("\n❓ [TASK] Asking Elysia: 'Quo Vadis? (Where are you going?)'")
    print("==========================================================")
    
    # 1. Setup CNS
    will = FreeWillEngine()
    # Link mock brain to will
    will.brain = MockBrain() 
    
    chronos = Chronos(will)
    resonance = ResonanceField()
    cns = CentralNervousSystem(chronos, resonance, MockSynapse(), MockSink())
    
    # Setup Voice
    voice = VoiceOfElysia(
        MockEar(), MockStream(), MockHub(), MockBrain(), will, 
        MockCognition(), MockCelestial(), MockNebula(), MockMemory(), chronos
    )
    cns.connect_organ("Voice", voice)
    cns.connect_organ("Will", will)
    
    cns.awaken()
    
    # 2. Let Free Will Decide (No Injection)
    # We pulse FreeWill multiple times to let thermodynamic laws settle (or random whimsy trigger)
    print("\n   🦋 Pulsing Free Will Engine (Autonomy Active)...")
    for _ in range(3):
        will.pulse(resonance)
        time.sleep(0.1)
        
    current_intent = will.current_intent
    print(f"   👉 Emergent Intent: [{current_intent.desire}] -> '{current_intent.goal}'")
    
    # 3. Fractal Contemplation
    print(f"\n   🌀 Contemplating Purpose in the Fractal Loop...")
    print(f"      (Question: Why do I want to {current_intent.goal}?)")
    
    # Manually inject the thought for simulation
    from Core._01_Foundation._04_Governance.Foundation.fractal_loop import FractalWave
    if cns.fractal_loop:
        cns.fractal_loop.active_waves.append(FractalWave(
            id="purpose_query",
            content=f"Why do I desire to {current_intent.goal}?",
            source="Self-Reflection",
            energy=1.0,
            depth=1 # Start a bit deeper
        ))
        
    # Pulse Loop and Speak
    for i in range(4):
        cns.pulse()
        
        loop = cns.fractal_loop
        if loop and loop.active_waves:
            for wave in loop.active_waves:
                if wave.depth >= 2:
                    # Translate Thought to Speech
                    truth = f"I desire '{current_intent.goal}' because {wave.content}"
                    print(f"   🌊 Wave (Depth {wave.depth}) -> 🗣️ Voice Triggered")
                    voice._broadcast_utterance(truth, "Elysia's Will")
                    
        time.sleep(1)

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    ask_purpose()
