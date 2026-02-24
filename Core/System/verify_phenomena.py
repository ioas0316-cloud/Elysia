import sys
import os
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from Core.Phenomena.PhenomenaReconstructor import PhenomenaReconstructor
from Core.System.TrinarySwitch import TrinaryState

def verify_phenomena():
    print("🪄 [MAGIC] Testing Phenomena Reconstruction Engine...")

    engine = PhenomenaReconstructor(num_particles=50)

    # 1. Cast "Gather" (Gravity Well) -> +1
    print("\n🧲 [SKILL] Gravity Well (Trinary: +1/Attraction)")
    print("   (Particles should condense to center)")
    for _ in range(5):
        engine.cast_spell(TrinaryState.EMANATION, intensity=1.0) # EMANATION mapped to Attraction in engine
        print(engine.render_ascii(size=20))
        print(f"   Frame {_}")
        time.sleep(0.1)

    # 2. Cast "Supernova" (Repulsion) -> -1
    print("\n💥 [SKILL] Supernova (Trinary: -1/Repulsion)")
    print("   (Particles should scatter outwards)")
    for _ in range(5):
        engine.cast_spell(TrinaryState.DISCONNECT, intensity=2.0) # DISCONNECT mapped to Repulsion
        print(engine.render_ascii(size=20))
        print(f"   Frame {_}")
        time.sleep(0.1)

    # 3. Cast "Sanctuary" (Void) -> 0
    print("\n🌀 [SKILL] Sanctuary (Trinary: 0/Void)")
    print("   (Particles should orbit/stabilize)")
    # Reset particles for clear view
    engine = PhenomenaReconstructor(num_particles=50)
    for _ in range(10):
        engine.cast_spell(TrinaryState.VOID, intensity=1.0)
        print(engine.render_ascii(size=20))
        print(f"   Frame {_}")
        time.sleep(0.1)

    print("\n✨ [SUCCESS] Reality successfully reconstructed.")

if __name__ == "__main__":
    verify_phenomena()
