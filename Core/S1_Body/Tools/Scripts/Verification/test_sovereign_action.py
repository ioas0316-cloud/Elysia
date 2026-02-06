import os
import sys
import asyncio
import logging

# Path Unification
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.S1_Body.L1_Foundation.Logic.d7_vector import D7Vector
from Core.S1_Body.L2_Metabolism.heart import get_heart

async def test_action():
    logging.basicConfig(level=logging.INFO)
    print("🎬 [TEST] Initializing SovereignSelf Action Test...")
    
    self_engine = SovereignSelf()
    heart = get_heart()
    
    # 1. Simulate a High-Torque Intent (Self-Evolution)
    # Refactor trigger
    intent = "Refactor the sensory bridge for higher bandwidth."
    
    print(f"🎯 [TEST] Intent: '{intent}'")
    
    # Use high resonance for testing
    heart.state = D7Vector(metabolism=0.9, spirit=0.9, mental=0.9, structure=0.8)
    
    # 2. Trigger Volition
    # We directly call _execute_volition with ActionCategory.CREATION logic
    # (Since we know 'Refactor' in intent triggers 'self_evolution')
    
    from Core.S1_Body.L5_Mental.M1_Cognition.cognitive_types import ActionCategory
    
    print("⚡ [TEST] Triggering Execution via SovereignExecutor...")
    self_engine._execute_volition(intent)
    
    print("✅ [TEST] Pulse Complete. Check data/Sandbox for patches.")

if __name__ == "__main__":
    asyncio.run(test_action())
