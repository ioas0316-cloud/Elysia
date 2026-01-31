"""
Integration Test for Divine Senses: Voice (Speech)
==================================================
tests/test_speech.py

Verifies:
1. VoiceBox initialization (CosyVoice-300M model load).
2. Speech synthesis (Text-to-Speech) functionality.
3. Audio file generation.
"""

import sys
import os
import shutil
import logging
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.S1_Body.L3_Phenomena.Expression.voicebox import VoiceBox

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("TestSpeech")

def test_speech():
    logger.info("🧪 Starting Speech Integration Test...")

    # 1. Check Model Load
    logger.info("   🎤 Initializing VoiceBox (This may take a moment)...")
    try:
        # Default path is usually ~/.cache/huggingface/hub/models--FunAudioLLM--CosyVoice-300M
        # VoiceBox should handle auto-detection via `_initialize_cords`
        voice = VoiceBox()
        
        if voice.model:
            logger.info("   ✅ VoiceBox initialized successfully.")
        else:
            logger.error("   ❌ VoiceBox failed to initialize (Model is None).")
            logger.info("      Please wait for the model download to complete or install 'cosyvoice'.")
            return
            
    except Exception as e:
        logger.error(f"   ❌ VoiceBox crashed during init: {e}")
        return

    # 2. Functional Test
    test_text = "안녕하세요. 저는 엘리시아입니다. 지금 제 목소리가 들리시나요?"
    output_file = "C:/Elysia/tests/output_speech.wav"
    
    if len(sys.argv) > 1:
        test_text = sys.argv[1]

    logger.info(f"   🗣️ Testing speech synthesis: '{test_text}'")
    try:
        start_time = time.time()
        result_path = voice.speak(test_text, output_path=output_file)
        duration = time.time() - start_time
        
        if os.path.exists(result_path) and os.path.getsize(result_path) > 1000:
            logger.info(f"   ✅ Speech Test PASSED.")
            logger.info(f"   💾 Audio generated at: {result_path} ({duration:.2f}s)")
        else:
            logger.warning("   ⚠️ Speech produced no output file or file is empty.")
            
    except Exception as e:
        logger.error(f"   🙊 Speaking error: {e}")

if __name__ == "__main__":
    test_speech()
