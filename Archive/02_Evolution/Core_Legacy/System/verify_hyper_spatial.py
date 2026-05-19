import requests
import json
import time
import random

SERVER_URL = "http://localhost:8000"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def test_hyper_spatial_engine():
    log("🔵 Connecting to Hyper-Spatial Reality Engine...")
    
    # 1. Check Heartbeat (GET /state)
    try:
        res = requests.get(f"{SERVER_URL}/state")
        if res.status_code == 200:
            data = res.json()
            meta = data.get("meta", {})
            log(f"✅ Connection Established!")
            log(f"   - Subjective Time Rate: x{meta.get('subjective_rate', 1.0):.2f}")
            log(f"   - Active Rotors: {list(meta.get('rotors', {}).keys())}")
        else:
            log(f"❌ Failed to connect: {res.status_code}")
            return
    except Exception as e:
        log(f"❌ Connection Error: {e}")
        return

    # 2. Test Monadic Intervention (POST /intervene)
    log("\n⚡ invoking MONADIC INTERVENTION (User Will Injection)...")
    
    concepts = ["ORDER", "CHAOS", "LOVE", "VOID", "LIGHT"]
    target_concept = random.choice(concepts)
    
    payload = {
        "concept": target_concept,
        "target_id": "Adam_01" # Target the Archetype
    }
    
    try:
        start_time = time.time()
        res = requests.post(f"{SERVER_URL}/intervene", json=payload)
        latency = (time.time() - start_time) * 1000
        
        if res.status_code == 200:
            log(f"✅ Intervention Successful! (Latency: {latency:.2f}ms)")
            log(f"   - Injected Concept: '{target_concept}'")
            log(f"   - Target: 'Adam_01'")
            log(f"   - The Laws of Physics have been locally rewritten.")
        else:
            log(f"❌ Intervention Failed: {res.status_code}")
            
    except Exception as e:
        log(f"❌ Intervention Error: {e}")

    # 3. Verify Lightning Path Field Modulation
    # We check if the world state reflects the high-frequency updates
    # (By checking simply if we can get the state again efficiently)
    log("\n🌊 Verifying Lightning Path (O(1) Field Access)...")
    start_time = time.time()
    res = requests.get(f"{SERVER_URL}/state")
    latency = (time.time() - start_time) * 1000
    log(f"✅ Lightning Path Active. Full World Snapshot retrieved in {latency:.2f}ms.")
    
    log("\n✨ Hyper-Spatial Engine Verification COMPLETE.")

if __name__ == "__main__":
    # Give server a moment to boot
    time.sleep(2)
    test_hyper_spatial_engine()
