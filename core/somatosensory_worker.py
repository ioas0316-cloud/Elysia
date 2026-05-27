import os
import time
import sys
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add root folder to sys.path to resolve imports correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.somatosensory_ingester import SomatosensoryIngester

def main():
    print("[Somatosensory Worker] Starting isolated hardware capture loop...")
    # Initialize in hardware mode (this process will import sounddevice/cv2/etc.)
    ingester = SomatosensoryIngester(hardware_mode=True)
    cache_path = r"c:\Elysia\data\somatosensory_cache.npz"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Pre-warm or cleanup previous cache to start clean
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except:
            pass
            
    while True:
        try:
            # Capture real audio and video
            audio = ingester.capture_audio(duration_sec=0.05, sample_rate=8000)
            video = ingester.capture_video()
            
            # Save to cache file atomically using a temp file to prevent partial reads on Windows
            temp_path = cache_path.replace(".npz", "_temp.npz")
            np.savez(temp_path, audio=audio, video=video, timestamp=time.time())
            
            # Safe replacement (Windows file lock safety loop)
            for _ in range(5):
                try:
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                    os.rename(temp_path, cache_path)
                    break
                except OSError:
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("[Somatosensory Worker] Stopping loop cleanly.")
            break
        except Exception as e:
            # Log error to stderr and sleep briefly before retry
            print(f"[Somatosensory Worker Error] {e}", file=sys.stderr)
            time.sleep(0.5)
            
        # Capture frequency: ~10 Hz (every 100ms)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
