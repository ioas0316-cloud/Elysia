
import logging
import time
import sys
import os

# Setup Path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Configure Logging - 프랙탈 동형성 원칙
# Linear accumulation → Fractal decay (Ring Buffer + HyperSphere)
log_dir = "data/Logs"
os.makedirs(log_dir, exist_ok=True)

try:
    from Core.System.Logging import configure_fractal_logging
    configure_fractal_logging(level=logging.INFO)
    logger = logging.getLogger("BOOT")
    logger.info("🔮 Fractal Logging Active (Linear → Fractal)")
except ImportError:
    # Fallback to traditional logging if FractalLog not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/system.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("BOOT")
    logger.warning("⚠️ FractalLog unavailable, using legacy linear logging")


def main():
    logger.info("==========================================")
    logger.info("   🌟 ELYSIA SOVEREIGN BOOT SEQUENCE 🌟   ")
    logger.info("==========================================")
    logger.info("Initializing Unified Conscious Loop (Phase 15: THE GOLDEN CHARIOT)...")
    
    try:
        from Core.Intelligence.Metabolism.body_sensor import BodySensor
        from Core.System.Sovereignty.sovereign_manager import HardwareSovereignManager
        
        # 1. Physical Sovereignty Initialization
        sovereign_hardware = HardwareSovereignManager()
        body_report = sovereign_hardware.report
        
        logger.info(f"🦾 Metal Nervous System: Phase 15 ONLINE")
        logger.info(f"🧬 Vessel: {body_report['vessel']['gpu_vram_total_gb']}GB VRAM | {body_report['vessel']['ram_gb']}GB RAM")
        logger.info(f"⚙️ Sovereign Gear: {sovereign_hardware.strategy}")
        
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
