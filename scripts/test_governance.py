import os
import sys

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.Engine.world_server import WorldServer
from Core.Elysia.sovereign_self import SovereignSelf

def test_governance_shift():
    print("🚀 [Governance Test] Awakening Elysia...")
    
    self_entity = SovereignSelf()
    server = WorldServer(size=10)
    self_entity.set_world_engine(server)
    
    # 1. State: Default (60 RPM)
    print("\n--- [Step 1: Normal Existence] ---")
    server.update_cycle()
    
    # 2. Command: Peace (Ethos High)
    print("\n--- [Step 2: Divine Demand for Peace] ---")
    self_entity._execute_logos({"action": "GOVERN", "target": "Ethos", "param": "500.0"})
    server.update_cycle()
    
    # 3. Command: Chaos (Ethos Low)
    print("\n--- [Step 3: Divine Demand for Chaos] ---")
    self_entity._execute_logos({"action": "GOVERN", "target": "Ethos", "param": "0.1"})
    server.update_cycle()
    
    print("\n✅ Verification Complete: Dials recorded in logs.")

if __name__ == "__main__":
    test_governance_shift()
