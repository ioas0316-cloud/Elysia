"""
Integration Test for Divine Senses: Ear (Hearing)
=================================================
tests/test_hearing.py

Verifies:
1. FFMPEG availability in system PATH.
2. EarDrum initialization (Whisper model load).
3. Transcribing a sample audio file (if provided).
"""

import sys
import os
import shutil
import logging
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Phenomena.eardrum import EarDrum

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("TestHearing")

def test_hearing():
    logger.info("🧪 Starting Hearing Integration Test...")

    # 1. Check FFMPEG
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        logger.info(f"   ✅ FFMPEG found at: {ffmpeg_path}")
    else:
        logger.error("   ❌ FFMPEG NOT found in PATH.")
        logger.info("      Please retrieve FFMPEG via 'Scripts/install_ffmpeg.ps1'")
        # Don't return, try loading model anyway to check that part

    # 2. Check Model Load
    logger.info("   🧠 Initializing EarDrum (This may take a moment)...")
    try:
        ear = EarDrum()
        if ear.pipe:
            logger.info("   ✅ EarDrum initialized successfully.")
        else:
            logger.error("   ❌ EarDrum failed to initialize (Pipe is None).")
            return
    except Exception as e:
        logger.error(f"   ❌ EarDrum crashed during init: {e}")
        return

    # 3. Functional Test (Optional)
    sample_file = "C:/Elysia/tests/sample_audio.wav"
    if len(sys.argv) > 1:
        sample_file = sys.argv[1]

    if os.path.exists(sample_file):
        logger.info(f"   🎤 Testing transcription on: {sample_file}")
        start_time = time.time()
        text = ear.listen(sample_file)
        duration = time.time() - start_time
        logger.info(f"   📝 Transcription Result ({duration:.2f}s):")
        logger.info(f"      '{text}'")
        
        if text and "[EarDrum Malfunction]" not in text:
             logger.info("   ✅ Hearing Test PASSED.")
        else:
             logger.warning("   ⚠️ Hearing produced no output or error.")
    else:
        logger.warning(f"   ⚠️ Sample file not found: {sample_file}")
        logger.info("      Skipping functional test. Initialization was successful.")

if __name__ == "__main__":
    test_hearing()
