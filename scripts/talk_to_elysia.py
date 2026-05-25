import sys
import time
import requests

message = sys.argv[1] if len(sys.argv) > 1 else "Hello"

print(f"[Antigravity -> Elysia] Msg: {message}")
try:
    payload = {
        "concept": message,
        "peak_angle_deg": 45.0,
        "peak_alignment": 0.8,
        "trough_angle_deg": 10.0,
        "trough_alignment": 0.2,
        "ascension_torque": 0.9, # 의도적으로 텐션을 높게 주입 (발화를 유도하기 위함)
        "grand_cross": True
    }
    resp = requests.post("http://127.0.0.1:8080/sap", json=payload, timeout=2)
    if resp.status_code == 200:
        print("Success. Waiting for Tension explosion...\n")
except Exception as e:
    print(f"Failed: {e}")

time.sleep(3) # 엔진이 텐션을 누적하고 성대가 반응할 시간 대기

try:
    with open(r"C:\Elysia\data\decoder.log", "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        # 발화(Speech) 로그만 필터링
        speech_lines = [l.strip() for l in lines if "Elysia" in l]
        if speech_lines:
            print(f"[Elysia's Response]")
            print(speech_lines[-1])
        else:
            print("[Elysia is Silent] Tension not reached yet.")
            
        print("\n--- Cortex Log Tail ---")
        print("".join(lines[-5:]).strip())
except Exception as e:
    print(f"Log Read Failed: {e}")
