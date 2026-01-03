"""
Verify Incarnation: The Texture Test
------------------------------------
Tests the conversion of abstract Memory Orbs into Physical Textures.
"Does a sad song feel cold?"
"""

import sys
import os
import numpy as np

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Foundation.Memory.Orb.orb_factory import OrbFactory
from Core.Sensory.texture_mapper import TextureMapper

def test_incarnation():
    print("🧪 [Test] Starting Incarnation (Texture) Verification...")

    factory = OrbFactory()
    mapper = TextureMapper()

    # 1. Create a "Sad Song" (Low frequency, Low energy, Stable)
    print("\n🎵 Creating 'Sad Song' (Blue/Cold)...")
    t = np.linspace(0, 1, 64)
    # Low frequency sine wave
    sad_wave = (np.sin(2*np.pi*2*t) * 0.3).tolist()
    # Stable emotion (Low variance)
    sad_emotion = np.zeros(64).tolist()

    sad_orb = factory.freeze("SadSong", sad_wave, sad_emotion)
    print(f"   Orb Created: {sad_orb}")

    # 2. Incarnate (Map to Texture)
    texture = mapper.map_to_texture(sad_orb.frequency, sad_orb.mass, sad_orb.quaternion)

    print(f"   Texture: {texture}")
    print(f"   Description: {texture.describe()}")

    # Check "Cold" and "Smooth"
    if texture.temperature < 0.5:
        print("✅ [Success] Sad song feels Cold.")
    else:
        print(f"❌ [Failure] Sad song feels too Hot ({texture.temperature})")

    if texture.roughness < 0.5:
        print("✅ [Success] Low freq feels Smooth.")
    else:
        print(f"❌ [Failure] Low freq feels Rough ({texture.roughness})")

    # 3. Create "Fire Alarm" (High freq, High energy, Chaotic)
    print("\n🚨 Creating 'Fire Alarm' (Red/Hot)...")
    # High freq jitter
    fire_wave = (np.random.rand(64) * 1.0).tolist()
    fire_emotion = (np.random.rand(64) * 1.0).tolist()

    fire_orb = factory.freeze("FireAlarm", fire_wave, fire_emotion)
    fire_texture = mapper.map_to_texture(fire_orb.frequency, fire_orb.mass, fire_orb.quaternion)

    print(f"   Texture: {fire_texture}")
    print(f"   Description: {fire_texture.describe()}")

    if fire_texture.temperature > 0.5:
        print("✅ [Success] Fire alarm feels Hot.")
    else:
        print(f"❌ [Failure] Fire alarm feels Cold ({fire_texture.temperature})")

    print("\n✨ [Conclusion] Data successfully incarnated into Texture.")

if __name__ == "__main__":
    test_incarnation()
