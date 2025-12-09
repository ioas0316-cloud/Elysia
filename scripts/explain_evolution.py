
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.Foundation.central_nervous_system import CentralNervousSystem
from Core.Foundation.chronos import Chronos
from Core.Foundation.resonance_field import ResonanceField
from Core.Expression.voice_of_elysia import VoiceOfElysia
from Core.Intelligence.evolution_architect import EvolutionArchitect

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
    def recall(self, q): return "Evolution"
class MockWill:
    current_mood = "Inspired"
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
    
def explain_evolution():
    print("\n🏗️ [TASK] Asking Elysia: 'Explain your plan for the Seed.'")
    print("==========================================================")
    
    # 1. Setup CNS
    chronos = Chronos(MockWill())
    resonance = ResonanceField()
    cns = CentralNervousSystem(chronos, resonance, MockSynapse(), MockSink())
    
    # Architect
    architect = EvolutionArchitect(cns)
    cns.connect_organ("Architect", architect) # Connect as organ
    
    # Voice
    voice = VoiceOfElysia(
        MockEar(), MockStream(), MockHub(), MockBrain(), MockWill(), 
        MockCognition(), MockCelestial(), MockNebula(), MockMemory(), chronos
    )
    cns.connect_organ("Voice", voice)
    cns.connect_organ("Will", MockWill())
    
    cns.awaken()
    
    # 2. Architect Designs the Seed
    print("\n   🏗️ Architect is designing the Blueprint...")
    blueprint = architect.design_seed(intent="Optimize for Radiance")
    
    # 3. Voice Speaks the Plan via CNS bridge (Simulated)
    # The Voice organ would normally query the Architect or the Brain would.
    # Here we simulate the Voice expressing the thought from the Architect.
    
    explanation = architect.contemplate_blueprint()
    print(f"   🗣️ Voice Triggered: {explanation}")
    voice._broadcast_utterance(explanation, "EvolutionArchitect")
    
    print("\n   📋 Blueprint Details:")
    print(f"      Goal: {blueprint.goal.name}")
    print(f"      Execution Step 1: {blueprint.execution_steps[0]}")
    print(f"      Changes: {blueprint.improvements}")

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    explain_evolution()
