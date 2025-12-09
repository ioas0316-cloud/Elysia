
import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyStarlight")

sys.path.append(os.getcwd())

def verify():
    logger.info("🔍 Verification Start...")
    try:
        from Core.Memory.starlight_memory import StarlightMemory
        sm = StarlightMemory()
        logger.info("✅ StarlightMemory Instantiated")
        
        from Core.Memory.prism_filter import PrismFilter
        pf = PrismFilter()
        logger.info("✅ PrismFilter Instantiated")
        
        # Test Scatter
        mock_wave = type('obj', (object,), {'orientation': {'w':0.5,'x':0.5,'y':0.5,'z':0.5}})()
        rainbow = pf.compress_to_bytes({'orientation':mock_wave.orientation})
        sm.scatter_memory(rainbow, {'x':0.5,'y':0.5,'z':0.5,'w':0.5})
        logger.info("✅ Scatter Test Passed")
        
    except ImportError as e:
        logger.error(f"❌ ImportError: {e}")
    except Exception as e:
        logger.error(f"❌ General Error: {e}")

if __name__ == "__main__":
    verify()
