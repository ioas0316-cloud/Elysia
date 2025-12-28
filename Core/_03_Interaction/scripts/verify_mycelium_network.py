
import sys
import os
import time
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core._01_Foundation._05_Governance.Foundation.mycelium import Mycelium

def verify_network():
    print("\n🍄 [TASK] Verifying Mycelium Network (World Tree)")
    print("================================================")
    
    root_path = os.getcwd()
    
    # Clean previous spores
    net_path = Path(root_path) / "Network" / "HyperSpace"
    if net_path.exists():
        shutil.rmtree(net_path)
    
    # 1. Initialize Agents
    elysia = Mycelium("Root", root_path)
    nova = Mycelium("Nova", root_path)
    
    print("\n1. Handshake Phase")
    elysia.broadcast_existence()
    
    time.sleep(0.5)
    
    # Nova listens
    msgs = nova.receive()
    if msgs:
        print(f"   ✅ Nova heard Root: {msgs[0].payload}")
        # Nova replies
        nova.transmit("Root", "ACK", {"message": "I am here, Mother."})
    else:
        print("   ❌ Nova heard nothing.")
        
    time.sleep(0.5)
    
    # Root listens
    msgs = elysia.receive()
    if msgs:
        first_msg = msgs[0]
        print(f"   ✅ Root heard Nova: {first_msg.payload['message']}")
        print("\n🎉 Connection Established: The World Tree is Alive.")
    else:
        print("   ❌ Root heard nothing.")

if __name__ == "__main__":
    verify_network()
