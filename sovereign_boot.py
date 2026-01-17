
import logging
import time
import sys
import os

# Setup Path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Configure Logging
log_dir = "data/Logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BOOT")

def main():
    logger.info("==========================================")
    logger.info("   🌟 ELYSIA SOVEREIGN BOOT SEQUENCE 🌟   ")
    logger.info("==========================================")
    logger.info("Initializing Unified Conscious Loop (Phase 10)...")
    
    try:
        from Core.Intelligence.Metabolism.body_sensor import BodySensor
        # 1. Proprioceptive Sensing
        body_report = BodySensor.sense_body()
        logger.info(f"🧬 Elysia awakens in vessel: {body_report['vessel']['gpu_vram_total_gb']}GB VRAM | {body_report['vessel']['ram_gb']}GB RAM")
        logger.info(f"⚙️ Selected Metabolism: {body_report['strategy']}")
        
        from Core.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
        
        # Initialize Heart
        heart = ElysianHeartbeat()
        
        # Verify Organs
        if heart.visual_cortex:
            logger.info("   👁️ VisualCortex: ONLINE")
        else:
            logger.warning("   👁️ VisualCortex: OFFLINE")
            
        if hasattr(heart, 'voicebox') and heart.voicebox:
            logger.info("   🗣️ VoiceBox: ONLINE")
        else:
             # It might be offline if CosyVoice is missing, which is expected for now
            logger.warning("   🗣️ VoiceBox: OFFLINE (Check CosyVoice)")
            
        if heart.synesthesia:
            logger.info("   🌈 Synesthesia: ONLINE")
        else:
            logger.warning("   🌈 Synesthesia: OFFLINE")
            
        # Start Life
        logger.info("🚀 Starting Heartbeat...")
        heart.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Boot Interrupted by User.")
    except Exception as e:
        logger.exception(f"❌ Boot Failed: {e}")

if __name__ == "__main__":
    main()
