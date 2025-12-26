
import sys
import os
import time
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core._01_Foundation.05_Foundation_Base.Foundation.mycelium import Mycelium
from Core._01_Foundation.05_Foundation_Base.Foundation.patch_manager import PatchManager

def simulate_symbiosis():
    print("\n🌲 [TASK] Simulating Symbiotic Code Evolution")
    print("============================================")
    
    root_path = os.getcwd()
    
    # 1. Initialize Agents
    root_net = Mycelium("Root", root_path)
    nova_net = Mycelium("Nova", root_path)
    
    root_patch = PatchManager(root_path)
    nova_patch = PatchManager(root_path) # In reality, different paths, but sharing workspace context for script
    
    # Clean network
    if (Path(root_path) / "Network" / "HyperSpace").exists():
        for f in (Path(root_path) / "Network" / "HyperSpace").glob("*.spore"):
            os.remove(f)

    # 2. Nova Creates a Patch
    print("\n⚡ [Nova] Detecting Optimization Opportunity...")
    target_file = "Network/optimization_log.txt" 
    # (Using a safe dummy file for demonstration)
    
    new_code = "Log Entry: Nova has optimized the flow. Latency reduced by 40%."
    patch_data = nova_patch.create_patch(
        author="Nova", 
        target_file=target_file, 
        new_content=new_code,
        reason="Found redundant loop in logic."
    )
    
    print(f"   🔧 Patch Created: {patch_data['id']} (Checksum: {patch_data['checksum'][:8]})")
    
    # 3. Nova Transmits Patch
    print("\n⚡ [Nova] Transmitting Patch to Root...")
    nova_net.transmit("Root", "PATCH_PROPOSAL", patch_data)
    
    time.sleep(1)
    
    # 4. Root Receives and Verifies
    print("\n🌳 [Root] Checking Message Bus...")
    msgs = root_net.receive()
    
    if msgs:
        packet = msgs[0]
        if packet.type == "PATCH_PROPOSAL":
            print(f"   📦 Recieved Patch Proposal from {packet.sender}")
            received_patch = packet.payload
            
            # Verify
            is_valid = root_patch.verify_patch(received_patch)
            if is_valid:
                print("   ✅ Patch Integrity Verified.")
                
                # Apply (Dry Run)
                root_patch.apply_patch(received_patch, dry_run=False) # Actually write the log file
                
                # Acknowledge
                root_net.transmit("Nova", "PATCH_ACCEPTED", {"patch_id": received_patch['id']})
    else:
        print("   ❌ No patch received.")
        
    time.sleep(1)
    
    # 5. Nova Confirmation
    msgs = nova_net.receive()
    if msgs:
        print(f"\n⚡ [Nova] Received: {msgs[0].type} for {msgs[0].payload['patch_id']}")
        print("🎉 Symbiosis Cycle Complete.")

if __name__ == "__main__":
    simulate_symbiosis()
