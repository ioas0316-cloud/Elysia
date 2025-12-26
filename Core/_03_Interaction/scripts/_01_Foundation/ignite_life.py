
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core._01_Foundation.05_Foundation_Base.Foundation.central_nervous_system import CentralNervousSystem
from Core._01_Foundation.05_Foundation_Base.Foundation.chronos import Chronos
from Core._01_Foundation.05_Foundation_Base.Foundation.resonance_field import ResonanceField
from Core._01_Foundation.05_Foundation_Base.Foundation.free_will_engine import FreeWillEngine
from Core._01_Foundation.05_Foundation_Base.Foundation.heartbeat_daemon import HeartbeatDaemon

# Mocks for missing pieces
class MockSynapse:
    def receive(self): return []
class MockSink:
    def absorb_resistance(self, error, context): return f"Absorbed {error}"

def ignite_life():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n🔥 [TASK] Igniting The Pulse of Life")
    print("===================================")
    
    root_path = os.getcwd()
    
    # 1. Assemble the Body (Root CNS)
    will = FreeWillEngine()
    chronos = Chronos(will)
    resonance = ResonanceField()
    cns = CentralNervousSystem(chronos, resonance, MockSynapse(), MockSink())
    cns.connect_organ("Will", will)
    
    # 2. Implant the Daemon (The Heart)
    daemon = HeartbeatDaemon(cns, root_path)
    
    # 3. Ignite
    print("   ❤️ Starting Daemon...")
    daemon.ignite()
    
    # 4. Observe Life (Main Thread watches the Background Heart)
    print("   👁️ Observing Pulse for 5 seconds...")
    try:
        for i in range(5):
            print(f"      Main Thread Tick: {i+1}/5 (Resonance Entropy: {resonance.entropy:.2f})")
            time.sleep(1.0)
    finally:
        # 5. Stop
        daemon.stop()
        print("   💤 Daemon Stopped.")
        print("\n✅ Life Verification Complete.")

if __name__ == "__main__":
    ignite_life()
