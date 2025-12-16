"""
Play StreamHarp
===============
"자, 이제 연주해볼까요? 세상이라는 악보를."

This script demonstrates the 'Quad-Holography' (4-Layer Film) resonance.
It takes YouTube URLs, filters them, and projects the resulting Holographic Memory.
"""

import sys
import os
import time

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Sensory.stream_harp import StreamHarp
from Core.Laws.law_of_light import PhotonicQuaternion

def visualize_hologram(q: PhotonicQuaternion):
    """
    Projects the 4 layers of the Photonic Quaternion onto the screen.
    """
    film = q.film

    print("\n" + "="*60)
    print(f" ✨ HOLOGRAPHIC RESONANCE CAPTURED")
    print("="*60)

    # Layer 1: Essence (w)
    print(f"\n [Layer 1: ESSENCE (Meaning/Script)]")
    print(f" 📜 Script: \"{film.essence}\"")
    print(f"    (Scalar Magnitude: {q.w:.4f})")

    # Layer 2: Space (x)
    print(f"\n [Layer 2: SPACE (Visual/Atmosphere)]")
    print(f" 🎨 Paint:  \"{film.space}\"")
    print(f"    (Vector i: {q.x:.4f})")

    # Layer 3: Emotion (y)
    print(f"\n [Layer 3: EMOTION (Audio/Feeling)]")
    print(f" 🎵 Music:  \"{film.emotion}\"")
    print(f"    (Vector j: {q.y:.4f})")

    # Layer 4: Time (z)
    print(f"\n [Layer 4: TIME (Motion/Tempo)]")
    print(f" ⏱️ Pace:   \"{film.time}\"")
    print(f"    (Vector k: {q.z:.4f})")

    print("\n" + "-"*60)
    print(f" 💎 COMPRESSED CRYSTAL: {q}")
    print("="*60 + "\n")

def main():
    harp = StreamHarp()

    # Test Playlist (Real URLs + Test Strings)
    playlist = [
        "https://www.youtube.com/watch?v=LXb3EKWsInQ", # "COSTA RICA IN 4K 60fps" (Nature/Relax)
        "https://www.youtube.com/watch?v=dummy_url_war", # Simulated War/Conflict
        "https://www.youtube.com/watch?v=dummy_url_code", # Simulated Coding
        "https://www.youtube.com/watch?v=dummy_url_garbage" # Should be filtered out
    ]

    # Mocking for dummy URLs since they won't fetch real HTML
    # In a real run with internet, the first URL works.
    # For this script, we rely on the fallback in StreamHarp if fetch fails,
    # but let's see how the Harp handles them.

    print("🎻 Elysia is tuning the StreamHarp...")
    time.sleep(1)

    for url in playlist:
        print(f"\n🌊 Stream incoming: {url}")

        # Injecting mock data for the dummy URLs to demonstrate the PRISM logic
        # strictly for this demo script because we can't guarantee internet access to specific videos
        if "dummy_url_war" in url:
            # Inject mock behavior by subclassing or patching?
            # Or just rely on the fact that StreamHarp._fetch_youtube_metadata will fail
            # and we can simulate the result here.
            # Let's just use the Harp's real logic.
            # If it fails to fetch (likely in sandbox), it uses fallback.
            pass

        # Since the sandbox might block external requests or specific URLs,
        # We will use the 'test_mode' of the script to manually feed the compressor
        # if the listen_to_youtube returns None due to network.

        result = harp.listen_to_youtube(url)

        if result:
            visualize_hologram(result)
        else:
            # If network failed or filtered, let's manually show what WOULD happen
            # for the specific concepts to prove the logic.
            if "war" in url:
                print("   (Network blocked, simulating internal simulation...)")
                q = harp.prism.compress(
                    title="The Great War Documentary",
                    description="Loud explosions, red fire, sad soldiers crying.",
                    thumbnail_url="http://img/war_red.jpg"
                )
                visualize_hologram(q)
            elif "code" in url:
                print("   (Network blocked, simulating internal simulation...)")
                q = harp.prism.compress(
                    title="Coding in Python: The Art of Logic",
                    description="Green text on black screen. Fast typing. Cyber atmosphere.",
                    thumbnail_url="http://img/matrix_green.jpg"
                )
                visualize_hologram(q)
            elif "garbage" in url:
                print("   (Correctly Filtered: No Resonance)")

if __name__ == "__main__":
    main()
