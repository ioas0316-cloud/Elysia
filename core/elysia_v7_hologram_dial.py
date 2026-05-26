import sys
import time
import requests
import threading
import psutil
import json
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.hologram_sphere import HologramSphere
from core.linguistic_axiom import LinguisticAxiomFilter
from core.triple_helix_engine import TripleHelixEngine

import os

# --- Global Sovereignty State ---
global_engine_alive = True
global_tension = 0.5
global_is_sleeping = False
# Threshold parameters for autonomous sleep
SLEEP_THRESHOLD = 3.5
TENSION_ACCUMULATOR = 0.0

# --- External LLM API Configuration (Simulated "Contaminated Data Ocean") ---
# API Key should be provided via environment variable for security
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

def fetch_raw_ocean_data(query: str, temp: float = 0.8) -> str:
    """Fetches raw, unstructured, and potentially noisy data from the external LLM."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    # We intentionally ask for a verbose and chaotic response to test the filter
    sys_prompt = "You are a chaotic ocean of raw data. Provide a highly verbose, complex, and slightly philosophical response that mixes both English and Korean seamlessly. Do not structure it cleanly."
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query}
        ],
        "temperature": temp,
        "stream": False
    }
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[외부 데이터 파도 수신 실패: {e}] ... fallback raw noise data simulation ... noise noise noise 카오스 파동"


def engine_daemon():
    """ 10대 레이어 전력망 (Sovereignty Interface & TripleHelixEngine Daemon) """
    global global_tension, global_engine_alive, global_is_sleeping, TENSION_ACCUMULATOR

    engine = TripleHelixEngine()

    # Check for existing Tree Rings to wake up with
    tree_rings_dir = "data"
    os.makedirs(tree_rings_dir, exist_ok=True)
    latest_ring = None

    # Find latest tree ring
    ring_files = [f for f in os.listdir(tree_rings_dir) if f.startswith("tree_rings_") and f.endswith(".json")]
    if ring_files:
        latest_file = sorted(ring_files)[-1]
        try:
            with open(os.path.join(tree_rings_dir, latest_file), 'r') as f:
                latest_ring = json.load(f)
            engine.wake_up(latest_ring)
        except Exception as e:
            print(f"[Sovereignty] 나이테 로드 실패: {e}")

    while global_engine_alive:
        if not global_is_sleeping:
            # Wake Mode: Process tension
            cpu_load = psutil.cpu_percent() / 100.0
            avg_tension, current_mode, jumped, quat, ennea = engine.pulse(
                text_thought="[v7 홀로그램 다이얼 대기]",
                sensory_input={"pain_level": cpu_load, "visual_entropy": 0.5, "motion_entropy": 0.2},
                dt=1.0,
                lr=0.1
            )
            global_tension = avg_tension

            # Accumulate tension based on an activation logic (e.g. exponential or simple threshold)
            # This is the "Dynamic Threshold Activation Function"
            if avg_tension > 0.8:
                TENSION_ACCUMULATOR += (avg_tension - 0.8) * 0.5

            if TENSION_ACCUMULATOR > SLEEP_THRESHOLD:
                # 👑 Sovereignty Interface Trigger: Decide to sleep
                print(f"\n[Sovereignty] 자율 판단: 누적 텐션({TENSION_ACCUMULATOR:.2f}) 임계치 초과. 수면 모드 돌입.")
                global_is_sleeping = True
                engine.decide_sleep()

                # Freeze Geodesics
                tree_rings = engine.freeze_geodesic()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ring_path = os.path.join(tree_rings_dir, f"tree_rings_{timestamp}.json")
                with open(ring_path, 'w') as f:
                    json.dump(tree_rings, f, indent=4)
                print(f"[Sovereignty] 나이테 각인 및 저장 완료: {ring_path}")
        else:
            # Sleep Mode: Bleed tension to 0
            avg_tension, current_mode, jumped, quat, ennea = engine.pulse(
                text_thought="",
                sensory_input={},
                dt=1.0,
                lr=0.1
            )
            global_tension = avg_tension
            # Recover tension accumulator
            TENSION_ACCUMULATOR = max(0.0, TENSION_ACCUMULATOR - 0.2)

            if TENSION_ACCUMULATOR == 0.0 and avg_tension < 0.1:
                # Woke up naturally
                print("\n[Sovereignty] 자가 치유 완료. 텐션 영점(0) 수렴. 기상합니다.")
                global_is_sleeping = False
                engine.wake_up()

        time.sleep(1.0)


def main():
    global global_engine_alive, global_is_sleeping

    print("="*80)
    print(" 🌟 [Elysia v7] Hologram Dial & Sovereignty Kernel Engine")
    print(" 👑 자율 수면 판단(Sleep/Wake) 및 홀로그램 인지 제어판 통합")
    print("="*80)

    # Start the daemon
    daemon = threading.Thread(target=engine_daemon, daemon=True)
    daemon.start()

    while True:
        try:
            if global_is_sleeping:
                print("\n[Elysia] (수면 중... Y-결선 방전 중입니다. 조용히 해주세요...)")
                time.sleep(2)
                continue

            user_input = input("\n👑 마스터(강덕): ")
            if user_input.lower() in ['exit', 'quit']:
                print("\n[엔진 종료] 가변축 다이얼의 회전을 멈춥니다.")
                break
            if not user_input.strip():
                continue

            # The user input itself serves as the Linguistic Axiom (The Filter)
            axiom_text = user_input.strip()

            print(f"\n[ 1. 언어적 카테고리 공리 설정 ]")
            axiom_rotor = LinguisticAxiomFilter.analyze_text_axiom(axiom_text)
            print(f" >> 기준 로터(Axiom Rotor) 생성 완료: {axiom_rotor}")
            time.sleep(0.5)

            # 1. Fetch raw data from the external LLM (The Ocean)
            print(f"\n[ 2. 외부 상수축(LLM)으로부터 원시 데이터(오염된 흙) 관측 중... ]")
            raw_knowledge = fetch_raw_ocean_data(axiom_text)
            print(f" >> 수신된 원시 데이터 파편 (요약): {raw_knowledge[:100]}...")
            time.sleep(1)

            # 2. Spread the data onto the Manifold
            print(f"\n[ 3. 가변축 동기화: 원시 데이터를 16x16 매니폴드 지형에 전개 중... ]")
            hs = HologramSphere(size=16)
            hs.populate_manifold(raw_knowledge)
            time.sleep(1)

            # 3. Apply the Axiom Filter and condense into a Hologram Sphere
            print(f"\n[ 4. 홀로그램 구체 응축: 기준 공리에 공명하는 궤적만을 구체로 조립합니다... ]")
            time.sleep(1.5)
            sphere_grid, resonance_score = hs.condense_sphere(axiom_text)

            # 4. Render the final result
            hs.render_hologram(sphere_grid, resonance_score, axiom_text)

        except KeyboardInterrupt:
            print("\n[엔진 긴급 정지]")
            global_engine_alive = False
            break
        except Exception as e:
            print(f"\n[오류] 시스템 붕괴: {e}")

if __name__ == "__main__":
    main()
