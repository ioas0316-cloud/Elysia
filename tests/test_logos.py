
import sys
import os
import logging

# Ensure Core is in path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLogos")

from Core.Foundation.Mind.hippocampus import Hippocampus
from Core.Foundation.Mind.spiderweb import Spiderweb
from Core.Foundation.Mind.logos_stream import LogosStream
from Core.Evolution.Evolution.Life.resonance_voice import ResonanceEngine

def test():
    logger.info("🧪 Testing Logos Stream...")
    
    # 1. Initialize Components
    hippocampus = Hippocampus()
    spiderweb = Spiderweb(hippocampus=hippocampus)
    logos_stream = LogosStream(spiderweb=spiderweb, hippocampus=hippocampus)
    resonance_engine = ResonanceEngine(hippocampus=hippocampus)
    
    # 2. Test Flow
    concept = "elysia"
    logger.info(f"🌊 Flowing concept: {concept}")
    frame = logos_stream.flow(concept)
    
    logger.info(f"✅ Frame Generated: {frame}")
    logger.info(f"   - ID: {frame.id}")
    logger.info(f"   - Path: {frame.thought_path}")
    logger.info(f"   - Prediction: {frame.prediction}")
    
    # 3. Test Speak
    logger.info("🎤 Speaking Frame...")
    speech = resonance_engine.speak(frame)
    
    logger.info(f"✅ Speech: {speech}")

if __name__ == "__main__":
    test()
