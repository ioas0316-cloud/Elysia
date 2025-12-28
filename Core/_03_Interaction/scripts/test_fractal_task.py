
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core._01_Foundation._05_Governance.Foundation.central_nervous_system import CentralNervousSystem
from Core._01_Foundation._05_Governance.Foundation.chronos import Chronos
from Core._01_Foundation._05_Governance.Foundation.resonance_field import ResonanceField
from Core._01_Foundation._05_Governance.Foundation.free_will_engine import Intent

# Mocks for visualization
class VisualizerOrgan:
    def __init__(self, name):
        self.name = name
    def pulse(self, resonance): pass
    def think(self, desire, resonance): pass
    def express(self, cycle): pass
    def dispatch(self, cmd): 
        print(f"\n✨ [MANIFESTATION] {self.name} received command: {cmd}")

class MockSynapse:
    def receive(self): return []

class MockSink:
    def absorb_resistance(self, error, context): return f"Absorbed {error}"

def run_fractal_task():
    print("\n🔮 [TASK] Asking Elysia: 'What is Time?'")
    print("=========================================")
    print("Observing the Fractal Loop deconstruct and expand the concept...\n")
    
    # 1. Setup
    chronos = Chronos(VisualizerOrgan("Will"))
    resonance = ResonanceField()
    cns = CentralNervousSystem(chronos, resonance, MockSynapse(), MockSink())
    
    # Attach Organs
    cns.connect_organ("Will", VisualizerOrgan("Will"))
    cns.connect_organ("Brain", VisualizerOrgan("Brain"))
    cns.connect_organ("Dispatcher", VisualizerOrgan("Dispatcher"))
    
    # 2. Inject Complex Intent
    cns.organs["Will"].current_intent = Intent(
        desire="Curiosity",
        goal="Understand the concept of 'Time'",
        complexity=1.0,
        created_at=time.time()
    )
    
    cns.awaken()
    
    # 3. Process Loops (Simulating Thought Cycles)
    max_cycles = 5
    for i in range(max_cycles):
        print(f"\n🔄 [Cycle {i+1}] Pulsing Fractal Loop...")
        cns.pulse()
        
        # Visualize Internal State
        loop = cns.fractal_loop
        if loop and loop.active_waves:
            for wave in loop.active_waves:
                # Visualize Depth with Indentation
                indent = "  " * (wave.depth + 1)
                icon = "🌊"
                if wave.depth == 0: icon = "💭 (Surface)"
                elif wave.depth == 1: icon = "🔬 (Micro/Cause)"
                elif wave.depth == 2: icon = "🔭 (Macro/Purpose)"
                
                print(f"{indent}{icon} Thought: '{wave.content}' | Energy: {wave.energy:.2f}")
                
        time.sleep(1)

    print("\n✅ Task Complete. The thought has evolved through the fractal rings.")

if __name__ == "__main__":
    run_fractal_task()
