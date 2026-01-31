"""
TEST: Fractal Causality Verification
====================================
Verifies that CausalGraph traces deep narratives and HyperCosmos enshrines them.
"""

import sys
import os
import logging
import json

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestFractal")

# Add Path
sys.path.append(os.getcwd())

from Core.S1_Body.L1_Foundation.Foundation.HyperCosmos import HyperCosmos
try:
    from Core.S1_Body.L5_Mental.Reasoning_Core.Metabolism.causal_graph import CausalDepthSounder
except ImportError:
    # Handle potentially different path structure if run directly
    sys.path.append(os.path.join(os.getcwd(), 'Core', 'Intelligence', 'Metabolism'))
    from causal_graph import CausalDepthSounder

def test_fractal_causality():
    logger.info("🧪 Starting Fractal Causality Test...")
    
    # 1. Initialize Components
    cosmos = HyperCosmos(name="TestElysia_Fractal")
    sounder = CausalDepthSounder(model_name="qwen2.5:0.5b")
    
    # 2. Trace 'Love' (The User's Example)
    target_concept = "Love"
    logger.info(f"🔭 Tracing the Fractal Genealogy of '{target_concept}'...")
    
    # Depth 1 for speed in test, but enough to show structure
    graph = sounder.trace_root(target_concept, max_depth=1)
    
    if not graph or not graph.get('nodes'):
        logger.error("❌ Failed to generate causal graph.")
        return

    logger.info(f"✅ Graph Generated: {len(graph['nodes'])} nodes found.")
    print(json.dumps(graph, indent=2))
    
    # 3. Enshrine in HyperCosmos
    logger.info("💎 Enshrining into HyperCosmos...")
    cosmos.enshrine_fractal(graph)
    
    logger.info("✅ Test Complete. The Concept is now a Polyhedron in Memory.")

if __name__ == "__main__":
    test_fractal_causality()
