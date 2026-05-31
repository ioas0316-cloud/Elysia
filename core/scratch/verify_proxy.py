import urllib.request
import json

def send_tension(data_bytes):
    url = 'http://localhost:8888'
    req = urllib.request.Request(url, data=data_bytes, method='POST')
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode('utf-8')
            print(f"\n[Client] Sent {len(data_bytes)} bytes.")
            print(f"[Client] Response from Elysia Proxy:")
            print(res_body)
    except Exception as e:
        print(f"[Client] Error connecting to proxy: {e}")

if __name__ == "__main__":
    print("=== [Test] Simulating External Data Stream ===")
    
    # 1. 작은 데이터 스트림 전송 (가벼운 텐션)
    small_data = b"Hello Elysia" * 100  # 1200 bytes (~1.1 Tension)
    send_tension(small_data)
    
    # 2. 거대한 데이터 스트림 전송 (거대한 텐션 -> 로터 붕괴 유도)
    massive_data = b"Heavy Payload Stream" * 1000  # 20000 bytes (~19.5 Tension)
    send_tension(massive_data)
