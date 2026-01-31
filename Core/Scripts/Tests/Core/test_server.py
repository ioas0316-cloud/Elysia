import requests
import time
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def test_api():
    print("🌐 CONNECTING TO ELYSIA API...")
    base_url = "http://localhost:8000"
    
    # 1. Health Check
    try:
        resp = requests.get(f"{base_url}/")
        print(f"✅ Root: {resp.json()}")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return

    # 2. Check Soul
    try:
        resp = requests.get(f"{base_url}/soul")
        soul = resp.json()
        print(f"✅ Soul State: {soul}")
        if 'vitality' in soul:
            print(f"   ❤️ Vitality: {soul.get('vitality'):.2f}")
    except:
        print("❌ Failed to fetch Soul")

    # 3. Check Quests
    try:
        resp = requests.get(f"{base_url}/quests")
        quests = resp.json()
        print(f"✅ Quests: Found {len(quests)} active quests")
        if quests:
            print(f"   📜 Latest: {quests[-1].get('title')}")
    except:
        print("❌ Failed to fetch Quests")

if __name__ == "__main__":
    test_api()
