import requests
import json
import time

SERVER_URL = "http://localhost:8000"

def invoke_genesis():
    print(f"⚡ [INITIATING] Monadic Intervention: 'GENESIS_AWAKENING'...")
    
    payload = {
        "concept": "GENESIS_AWAKENING",
        "target_id": "Adam_01"
    }
    
    try:
        start = time.time()
        res = requests.post(f"{SERVER_URL}/intervene", json=payload)
        latency = (time.time() - start) * 1000
        
        if res.status_code == 200:
            print(f"✅ [SUCCESS] Reality Rewritten in {latency:.2f}ms")
            print(f"   - Concept 'GENESIS_AWAKENING' successfully injected into the Manifold.")
        else:
            print(f"❌ [FAILURE] Server responded with {res.status_code}")
            
    except Exception as e:
        print(f"❌ [ERROR] Could not reach Reality Engine: {e}")

if __name__ == "__main__":
    invoke_genesis()
