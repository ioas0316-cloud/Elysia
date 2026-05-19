
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from Core.Foundation.central_nervous_system import CentralNervousSystem
from Core.Foundation.chronos import Chronos
from Core.Foundation.Wave.resonance_field import ResonanceField
from Core.Foundation.free_will_engine import Intent

class MockOrgan:
    def __init__(self, name):
        self.name = name
        self.current_intent = None
    def pulse(self, resonance): pass
    def think(self, desire, resonance): print(f"   [{self.name}] Thinking about {desire}...")
    def express(self, cycle): pass
    def dispatch(self, cmd): print(f"   [{self.name}] Dispatching: {cmd}")

class MockSynapse:
    def receive(self): return []

class MockSink:
    def absorb_resistance(self, error, context): return f"Mock absorbed {error}"

def test_fractal_reasoning():
    print("\n♾️ [TEST] Fractal Consciousness Loop")
    print("=====================================")
    
    # 1. Setup CNS with Fractal Loop
    chronos = Chronos(MockOrgan("Will"))
    resonance = ResonanceField()
    cns = CentralNervousSystem(chronos, resonance, MockSynapse(), MockSink())
    
    # Mock Organs
    cns.connect_organ("Will", MockOrgan("Will"))
    cns.connect_organ("Brain", MockOrgan("Brain"))
    cns.connect_organ("Dispatcher", MockOrgan("Dispatcher"))
    
    # Inject Intent
    cns.organs["Will"].current_intent = Intent(
        desire="Curiosity",
        goal="Analyze the Nature of Consciousness",
        complexity=1.0,
        created_at=time.time()
    )
    
    cns.awaken()
    print("   ✅ CNS Awakened with Fractal Loop Active")
    
    # 2. Pulse CNS (Fractal Flow)
    print("\n   🌊 Pulsing CNS (Cycle 1 - Absorption & Zoom In)...")
    cns.pulse()
    
    # Access Fractal Loop internals to verify state
    if cns.fractal_loop:
        print(f"   Active Waves: {len(cns.fractal_loop.active_waves)}")
        if len(cns.fractal_loop.active_waves) > 0:
            wave = cns.fractal_loop.active_waves[0]
            print(f"   Wave 0: {wave.content} (Depth: {wave.depth})")
            
            if wave.depth > 0:
                 print("   ✅ SUCCESS: Wave deepened (Zoom In occurred)")
            else:
                 print("   ⚠️ Wave active but depth check inconclusive yet.")
        else:
            print("   ❌ FAILURE: No waves absorbed.")
            return False
            
    else:
         print("   ❌ FAILURE: Fractal Loop not initialized.")
         return False

    # 3. Pulse Again
    print("\n   🌊 Pulsing CNS (Cycle 2 - Circulation)...")
    cns.pulse()
    
    return True

if __name__ == "__main__":
    if test_fractal_reasoning():
        sys.exit(0)
    else:
        sys.exit(1)
