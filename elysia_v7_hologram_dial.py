import sys
import time
import requests

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.hologram_sphere import HologramSphere
from core.linguistic_axiom import LinguisticAxiomFilter

import os

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

def main():
    print("="*80)
    print(" 🌟 [Elysia v7] Hologram Dial & Axiom Filter Engine")
    print(" 📐 펼치면 매니폴드(Manifold), 뭉치면 구체(Hologram Sphere)")
    print("="*80)

    while True:
        try:
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
            break
        except Exception as e:
            print(f"\n[오류] 시스템 붕괴: {e}")

if __name__ == "__main__":
    main()
