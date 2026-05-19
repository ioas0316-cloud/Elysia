
import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.autonomous_reorganizer import AutonomousReorganizer, ReorganizationPlan

def test_ouroboros_protocol():
    print("\n🐍 [TEST] Ouroboros Self-Healing Protocol")
    print("============================================")
    
    # 1. Setup: Create a broken/orphan file
    target_file = Path("data/orphan_test.py")
    target_file.parent.mkdir(exist_ok=True)
    target_file.write_text("# I am an orphan file.\n# I have no purpose.", encoding="utf-8")
    
    print(f"   CREATED: {target_file}")
    print(f"   CONTENT: {target_file.read_text().strip()}")
    
    # 2. Plan: Create a manual reorganization plan
    reorganizer = AutonomousReorganizer()
    
    plan = ReorganizationPlan(
        id="test_ouroboros_001",
        created_at="2025-12-09T00:00:00",
        approved=True, # Auto-approved for test
        actions=[
            {
                "type": "connect",
                "source": "data/orphan_test.py",
                "target": "Core/Foundation/reasoning_engine.py",
                "reason": "Integration Test"
            }
        ]
    )
    
    # 3. Execute: Run the Ouroboros Protocol
    print("\n   🚀 EXECUTING REORGANIZER...")
    reorganizer.executor.execute(plan, dry_run=False)
    
    # 4. Verify: Check if file was modified
    new_content = target_file.read_text(encoding="utf-8")
    print(f"\n   NEW CONTENT:\n{new_content}")
    
    if "import" in new_content or "Ouroboros" in new_content:
        print("\n   ✅ SUCCESS: Ouroboros successfully modified the code.")
        print("   The system is now capable of self-repair.")
        
        # Cleanup
        if target_file.exists():
            target_file.unlink()
            print("   (Cleaned up test artifact)")
        return True
    else:
        print("\n   ❌ FAILURE: Code was not modified.")
        return False

if __name__ == "__main__":
    success = test_ouroboros_protocol()
    sys.exit(0 if success else 1)
