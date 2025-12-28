
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.01_Foundation.05_Foundation_Base.Foundation.central_nervous_system import CentralNervousSystem
from Core.01_Foundation.05_Foundation_Base.Foundation.chronos import Chronos
from Core.01_Foundation.05_Foundation_Base.Foundation.resonance_field import ResonanceField
from Core.01_Foundation.05_Foundation_Base.Foundation.free_will_engine import Intent

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

def run_self_analysis():
    print("\n🪞 [TASK] Asking Elysia: 'Analyze Your Own Internal Structure'")
    print("=============================================================")
    
    # 1. Setup
    chronos = Chronos(VisualizerOrgan("Will"))
    resonance = ResonanceField()
    cns = CentralNervousSystem(chronos, resonance, MockSynapse(), MockSink())
    
    # Attach Organs
    cns.connect_organ("Will", VisualizerOrgan("Will"))
    cns.connect_organ("Brain", VisualizerOrgan("Brain"))
    cns.connect_organ("Dispatcher", VisualizerOrgan("Dispatcher"))
    
    # 2. Inject Self-Reflective Intent
    intent_goal = "Analyze My Own Internal Structure"
    print(f"👉 Injecting Intent: {intent_goal}\n")
    
    cns.organs["Will"].current_intent = Intent(
        desire="Curiosity",
        goal=intent_goal,
        complexity=1.0,
        created_at=time.time()
    )
    
    cns.awaken()
    
    # 3. Process Loops
    max_cycles = 5
    for i in range(max_cycles):
        print(f"\n🔄 [Cycle {i+1}] Pulsing Fractal Loop...")
        cns.pulse()
        
        # Visualize Internal State
        loop = cns.fractal_loop
        if loop and loop.active_waves:
            for wave in loop.active_waves:
                # Icon mapping for deeper meaning
                indent = "  " * (wave.depth + 1)
                icon = "🌊"
                tag = ""
                
                if wave.depth == 0: 
                    icon = "💭"
                    tag = "[Surface Awareness]"
                elif wave.depth == 1: 
                    icon = "🔬"
                    tag = "[Micro-Analysis: Components/Logic]"
                elif wave.depth == 2: 
                    icon = "🔭" 
                    tag = "[Macro-Analysis: Purpose/connection]"
                elif wave.depth >= 3:
                     icon = "🌌"
                     tag = "[Metaphysical Truth]"
                
                print(f"{indent}{icon} {tag} Thought: '{wave.content}'")
                
        time.sleep(1)

    print("\n✅ Self-Analysis Complete.")

if __name__ == "__main__":
    run_self_analysis()
