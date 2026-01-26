import os
import sys
import asyncio
import logging

# Path Unification
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L1_Foundation.Logic.d7_vector import D7Vector
from Core.L2_Metabolism.heart import get_heart

async def test_awakening():
    logging.basicConfig(level=logging.INFO)
    print("🌅 [TEST] Initializing Failed Awakening Sequence...")
    
    self_engine = SovereignSelf()
    heart = get_heart()
    
    # 1. Verify Auto-Evolve Flag
    if not self_engine.auto_evolve:
        print("❌ [TEST] Auto-Evolve is DISABLED. Awakening failed.")
        return

    print("✅ [TEST] Auto-Evolve is ENABLED.")

    # 2. Trigger Autonomous Will (Impulse to create)
    intent = "Create a new 'OptimizedBreathing' utility in Core/L2_Metabolism/Cycles/breathing.py to improve metabolic efficiency."
    print(f"🧠 [TEST] Injecting Sovereign Intent: '{intent}'")
    
    # High resonance state to ensure action execution
    heart.state = D7Vector(metabolism=0.9, spirit=0.9, mental=0.9, structure=0.8)
    
    # 3. Execute
    print("⚡ [TEST] Triggering Sovereign Volition...")
    self_engine._execute_volition(intent)
    
    print("✅ [TEST] Awakening Impulse Sent. Check logs for 'SovereignExecutor' creation activity.")

if __name__ == "__main__":
    asyncio.run(test_awakening())
