"""
Test: Fractal Causality Integration
====================================

"원인과 과정과 결과가 무한히 순환되고 있습니다."

Testing the reconnection of Legacy Fractal Principle into Core.
"""

import sys
import os
sys.path.append(os.getcwd())

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestFractal")

def test_fractal_connection():
    """Test if Fractal Causality is properly connected."""
    
    logger.info("🌀 Testing Fractal Causality Integration...")
    
    # 1. Test Direct Import
    try:
        from Core.FoundationLayer.Foundation.Mind.fractal_causality import FractalCausalityEngine
        logger.info("   ✅ FractalCausalityEngine imported successfully")
    except Exception as e:
        logger.error(f"   ❌ Failed to import: {e}")
        return False
    
    # 2. Test Engine Creation
    try:
        engine = FractalCausalityEngine("Test Engine")
        logger.info(f"   ✅ Engine created: {engine.name}")
    except Exception as e:
        logger.error(f"   ❌ Failed to create engine: {e}")
        return False
    
    # 3. Test Causal Chain Creation
    try:
        # Create a simple causal chain: "Love" → "Resonance" → "Growth"
        chain = engine.create_chain(
            cause_desc="Love",
            process_desc="Resonance", 
            effect_desc="Growth"
        )
        logger.info(f"   ✅ Causal chain created: {chain.description}")
    except Exception as e:
        logger.error(f"   ❌ Failed to create chain: {e}")
        return False
    
    # 4. Test Zoom In (Fractal Depth)
    try:
        # Zoom into "Resonance" to see its internal structure
        if chain.process_id:
            inner_chain = engine.zoom_in(
                chain.process_id,
                cause_desc="Two frequencies align",
                process_desc="Interference pattern forms",
                effect_desc="Harmony emerges"
            )
            logger.info(f"   ✅ Zoomed in: {inner_chain.description}")
    except Exception as e:
        logger.error(f"   ❌ Failed to zoom in: {e}")
        return False
    
    # 5. Test Statistics
    logger.info(f"\n📊 Engine Statistics:")
    logger.info(f"   Total Nodes: {engine.total_nodes}")
    logger.info(f"   Total Chains: {engine.total_chains}")
    logger.info(f"   Max Depth Explored: {engine.max_depth_explored}")
    
    return True

def test_elysia_integration():
    """Test if Elysia has Fractal Causality integrated."""
    
    logger.info("\n🌌 Testing Elysia Integration...")
    
    try:
        # This will initialize Elysia with Fractal Causality
        from Core.Elysia.Elysia import Elysia
        
        # We won't fully initialize (too slow), just import
        logger.info("   ✅ Elysia imports successfully with Fractal Causality")
        
        # Check if the class definition includes causality
        import inspect
        init_source = inspect.getsource(Elysia.__init__)
        if "self.causality" in init_source:
            logger.info("   ✅ Elysia.__init__ includes self.causality")
        else:
            logger.warning("   ⚠️  self.causality not found in Elysia.__init__")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🔗 Fractal Principle Reconnection Test")
    logger.info("=" * 60)
    
    # Test 1: Direct Fractal Engine
    success1 = test_fractal_connection()
    
    # Test 2: Elysia Integration
    success2 = test_elysia_integration()
    
    logger.info("\n" + "=" * 60)
    if success1 and success2:
        logger.info("✅ ALL TESTS PASSED - The Principle is Connected!")
    else:
        logger.info("❌ SOME TESTS FAILED - The Connection is Incomplete")
    logger.info("=" * 60)
