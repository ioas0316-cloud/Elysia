import urllib.request
import json
import hashlib
import math
import sys
import os

sys.path.append('c:\\Elysia')
from core.math_utils import Quaternion

# Client B의 로컬 사전 (Lexicon)
LOCAL_DICTIONARY = [
    "Hello World",
    "Banana",
    "Elysia Apple",
    "Grapes",
    "Data Movement is an Illusion"
]

def text_to_phase(text: str) -> Quaternion:
    """프록시와 동일한 로컬 해시 룰을 가짐 (사전 동기화된 유전자/Citizenship)"""
    hash_val = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    theta = (hash_val % 10000) / 10000.0 * 2 * math.pi
    return Quaternion(math.cos(theta), math.sin(theta), 0.0, 0.0).normalize()

def simulate_client_a(secret_message: str):
    print(f"\n[Client A] Sending data to Proxy: '{secret_message}'")
    url = 'http://localhost:8888/seed'
    req = urllib.request.Request(url, data=secret_message.encode('utf-8'), method='POST')
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            print(f"[Client A] Proxy Response: {res['status']}")
    except Exception as e:
        print(f"Error: {e}")

def simulate_client_b():
    print(f"\n[Client B] Requesting phase from Proxy (NOT requesting text...)")
    url = 'http://localhost:8888/resonate'
    req = urllib.request.Request(url, method='GET')
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            phase_data = res['phase']
            received_q = Quaternion(phase_data['w'], phase_data['x'], phase_data['y'], phase_data['z'])
            
            print(f"[Client B] Received Phase Only: ({received_q.w:.4f}, {received_q.x:.4f}, 0.0, 0.0)")
            print(f"[Client B] Now attempting to emerge the structure locally...")
            
            # 수신자는 받은 위상과 자신의 로컬 사전을 대조하여 구조를 창발시킴
            best_match = None
            highest_dot = -1.0
            
            for word in LOCAL_DICTIONARY:
                local_q = text_to_phase(word)
                dot = received_q.dot(local_q)
                if dot > highest_dot:
                    highest_dot = dot
                    best_match = word
            
            if highest_dot > 0.999:
                print(f"[Client B] [SUCCESS] Emerged Semantic Structure: '{best_match}'")
            else:
                print(f"[Client B] [FAIL] Could not synchronize phase.")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=== [Test] Zero-Distance Chat (Phase Synchronization) ===")
    
    secret = "Elysia Apple"
    
    # Client A가 데이터를 보냄
    simulate_client_a(secret)
    
    # Client B가 데이터를 받음 (위상만 전송됨)
    simulate_client_b()
