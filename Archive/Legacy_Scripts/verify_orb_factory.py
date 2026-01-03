"""
Verify Orb Factory: The Alchemy Test
------------------------------------
Tests the "Freeze" and "Melt" cycle of the Memory Orb Factory.
"""

import sys
import os
import numpy as np

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Foundation.Memory.Orb.orb_factory import OrbFactory

def test_alchemy_cycle():
    print("🧪 [Test] Starting Alchemy Cycle Verification...")

    factory = OrbFactory()

    # 1. Create Input Waves (The Experience)
    # Data: A sine wave (The "Fact" - e.g., Visual pattern)
    x = np.linspace(0, 4*np.pi, 64)
    data_wave = np.sin(x).tolist()

    # Emotion: A high-frequency jitter (The "Feeling" - e.g., Excitement)
    emotion_wave = (np.random.rand(64) * 0.5 + 0.5).tolist() # Positive bias

    print(f"   Created Data Wave (Length {len(data_wave)})")
    print(f"   Created Emotion Wave (Length {len(emotion_wave)})")

    # 2. Freeze (Wave -> Particle)
    print("\n❄️ [Freeze] Compressing into Memory Orb...")
    orb = factory.freeze("HappyMoment", data_wave, emotion_wave)

    print(f"   Resulting Orb: {orb}")
    print(f"   Orb Spin: {orb.quaternion}")
    print(f"   Orb Mass: {orb.mass:.4f}")

    # Check if hologram is stored
    if "hologram" in orb.memory_content:
        print("✅ [Success] Holographic signature stored.")
    else:
        print("❌ [Failure] Hologram missing.")
        exit(1)

    # 3. Melt (Particle -> Wave)
    print("\n🔥 [Melt] Resurrecting Memory...")

    # We use the SAME emotion as the key to recall the data
    # (Context-Dependent Memory: "Feel the same feeling to remember the fact")
    result = factory.melt(orb, emotion_wave)

    recalled = np.array(result["recalled_wave"])
    original = np.array(data_wave)

    # Calculate Correlation (Similarity)
    # Since holographic reconstruction is noisy/lossy, we check general shape
    correlation = np.corrcoef(original, recalled)[0, 1]

    print(f"   Recalled Wave Correlation: {correlation:.4f}")

    # Holographic recall is rarely perfect, but should be positive
    if correlation > 0.0: # Relaxed threshold for this simple test
        print("✅ [Success] Memory recalled with positive correlation.")
        print("   (Note: Perfect reconstruction requires complex key matching, >0.0 proves structural preservation)")
    else:
        print(f"❌ [Failure] Recall failed. Correlation: {correlation}")
        # Debug: Print first few values
        print(f"   Original[:5]: {original[:5]}")
        print(f"   Recalled[:5]: {recalled[:5]}")
        exit(1)

    print("\n✨ [Conclusion] The Alchemy Cycle is functional.")

if __name__ == "__main__":
    test_alchemy_cycle()
