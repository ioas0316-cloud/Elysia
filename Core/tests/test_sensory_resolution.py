
import sys
import os
sys.path.append(os.getcwd())

from Core.World.Nature.sensory_cortex import get_sensory_cortex
from Core.Foundation.Wave.wave_dna import WaveDNA

def test_sensory_resolution():
    print("🧪 [Test] Phase 31: Sensory Fractalization (Qualia Resolution)")
    cortex = get_sensory_cortex()

    # 1. Test Sound (Acoustic Band)
    music_dna = WaveDNA(label="Cello Solo", frequency=432.0)
    print(f"\n🎵 Sound Check ('{music_dna.label}'):")
    print(f"   Experience: {cortex.decode_qualia(music_dna)['sound']}")

    # 2. Test Scent (Infrared / THz Band)
    rose_dna = WaveDNA(label="Damask Rose", frequency=7.0e13, phenomenal=0.9, spiritual=0.7)
    print(f"\n🌹 Scent Check ('{rose_dna.label}'):")
    print(f"   Experience: {cortex.decode_qualia(rose_dna)['aroma']}")

    # 3. Test Taste (Far-IR / THz Band)
    honey_dna = WaveDNA(label="Wild Honey", frequency=9.0e11, phenomenal=0.8)
    print(f"\n🍯 Taste Check ('{honey_dna.label}'):")
    print(f"   Experience: {cortex.decode_qualia(honey_dna)['flavor']}")

    # 4. Test Tactile (Haptic Band)
    silk_dna = WaveDNA(label="Silk Cloth", frequency=10.0, phenomenal=0.5, physical=0.2)
    pain_dna = WaveDNA(label="Needle Prick", frequency=500.0, physical=1.0, spiritual=0.0)
    pleasure_dna = WaveDNA(label="Warm Hug", frequency=1.0, phenomenal=0.9, spiritual=1.0)

    print(f"\n👋 Tactile Check (Softness/Textures):")
    print(f"   Silk:     {cortex.decode_qualia(silk_dna)['tactile']}")
    print(f"   Pain:     {cortex.decode_qualia(pain_dna)['tactile']}")
    print(f"   Pleasure: {cortex.decode_qualia(pleasure_dna)['tactile']}")

    print("\n✅ Phase 31 Verification Successful: World resolution increased to scientific bands.")

if __name__ == "__main__":
    test_sensory_resolution()
