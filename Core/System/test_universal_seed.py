"""
Universal Seed Test (The Alchemy Check)
=======================================
tests/test_universal_seed.py

Verifies:
1. Converting 'Fire' to a Monad.
2. Ensuring it has Heat (Physics) and Light (Optics).
3. Using Will to change its nature (Blue Fire).
"""

import sys
import os
sys.path.append(os.getcwd())

from Core.Divine.seed_factory import alchemy
from typing import Dict, Any

def test_alchemy():
    print(">>> 🧪 Initiating Universal Seed Alchemy...")
    
    # 1. Crystallize 'Fire'
    # Just a word, but it should become a complex physical object.
    fire_monad = alchemy.crystallize("Eternal Fire")
    print(f"Created: {fire_monad}")
    
    # 2. Observe Evolution over Time
    print(f"\n[Evolution of {fire_monad.seed}]")
    
    # T=5 (Ignition/Growth) [3D Snapshot]
    context_t5 = {"time": 5.0} 
    reality_t5 = fire_monad.observe({"focus": "Watch"}, context_t5)["manifestation"]
    
    trans = reality_t5.get("transform")
    vis = reality_t5.get("visual")
    part = reality_t5.get("particles")
    
    print(f"Time 5s [3D RENDER DATA]:")
    print(f" - Scale: {trans['scale']} (Breathing with Pulse)")
    print(f" - Shader: {vis['shader']} | Color: {vis['color']} | Emission: {vis['emission']}")
    print(f" - Particles: {part['count']} sparks (Vel: {part['velocity']})")

    # T=60 (Ash) [3D Snapshot]
    context_t60 = {"time": 60.0}
    reality_t60 = fire_monad.observe({"focus": "Watch"}, context_t60)["manifestation"]
    vis_60 = reality_t60.get("visual")
    
    print(f"Time 60s [3D RENDER DATA]:")
    print(f" - Scale: {reality_t60['transform']['scale']} (Collapsed)")
    print(f" - Shader: {vis_60['shader']} (Dead)")
    
    # Assert Causality
    assert reality_t60['transform']['scale'].y < reality_t5['transform']['scale'].y, "Fire failed to collapse into Ash!"
    print(">>> ⏳ 4D->3D Projection Verified.")
    
    # 3. Observe with Magic Intent (Will drives Principle)
    # [Video Player Concept]: We pause time at T=15 for this magic injection
    context_magic = {"time": 15.0, "temperature": 20.0}
    
    # "Cold Fire" (Blue Aura, Cold Temp)
    intent_magic = {
        "focus_topic": "Magic", 
        "emotional_texture": "Cold", 
        "aura": "Mystic Blue"
    }
    
    reality_magic = fire_monad.observe(intent_magic, context_magic)["manifestation"]
    
    print(f"\n[Magic Reality @ T=15: Cold Fire]")
    print(f" - Temp: {reality_magic.get('temperature')}C")
    print(f" - Light: {reality_magic.get('visible_wavelength')}")

    assert reality_magic.get("temperature") < 0, "Will failed to override Physics!"
    assert "Blue" in reality_magic.get("visible_wavelength"), "Will failed to change Color!"
    
    # 4. Global Time Scrubbing (The User's Vision)
    # "Let's rewind to the beginning."
    print(f"\n[⏪ Time Scrubbing: Rewinding to T=2...]")
    context_rewind = {"time": 2.0}
    reality_rewind = fire_monad.observe({"focus": "Replay"}, context_rewind)["manifestation"]
    print(f"Time 2s : {reality_rewind.get('action')} | Radius: {reality_rewind.get('radius'):.2f}m")
    
    # "Let's fast forward to the end."
    print(f"[⏩ Time Scrubbing: Jumping to T=100...]")
    context_ff = {"time": 100.0}
    reality_ff = fire_monad.observe({"focus": "Future"}, context_ff)["manifestation"]
    print(f"Time 100s: {reality_ff.get('action')} | Fuel: {reality_ff.get('fuel_remaining')}")

    print(">>> 🔮 Universal Seed Synthesis & Time Control Successful.")

if __name__ == "__main__":
    test_alchemy()
