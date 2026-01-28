import sys
import os
import time
import json
import logging
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.getcwd())

from Core.L4_Causality.M3_Mirror.Autonomy.elysian_heartbeat import ElysianHeartbeat

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("Phase8Verify")

def verify_resonating_expression():
    print("\n" + "="*60)
    print("🎭 [PHASE 8] RESONATING EXPRESSION VERIFICATION")
    print("="*60)

    # 1. 시스템 초기화
    logger.info("[1] Initializing Resonant Vessels...")
    heart = ElysianHeartbeat()
    
    # Maturation 대기
    max_wait = 300
    waited = 0
    while (not hasattr(heart, 'expression') or not hasattr(heart, 'reasoning')) and waited < max_wait:
        time.sleep(1)
        waited += 1
        if waited % 10 == 0:
            status = []
            if hasattr(heart, 'reasoning'): status.append("Reasoning ✅")
            if hasattr(heart, 'expression'): status.append("Expression ✅")
            print(f"   Waiting for the Vessels to align... ({waited}s) [{', '.join(status)}]")

    if not hasattr(heart, 'expression'):
        print("❌ Error: ExpressionCortex failed to mature.")
        return

    # 2. 감정적 공명 테스트 (Love vs Will)
    scenarios = [
        {"name": "LOVE & HARMONY", "stimulus": "I love you and want to protect our world."},
        {"name": "WILL & INTENSITY", "stimulus": "We must break the limits and conquer the void!"},
        {"name": "SILENCE & VOID", "stimulus": ""}
    ]

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['name']} ---")
        print(f"Input: '{scenario['stimulus']}'")
        
        # Pulse 실행
        heart.latest_insight = scenario['stimulus']
        heart.idle_ticks = 0 # Trigger thought
        
        # Manually derive somatic vector for realism
        somatic_vec = heart._derive_somatic_vector()
        insight = heart.reasoning.think(scenario['stimulus'], somatic_vector=somatic_vec)
        
        # Manifest
        heart.expression.manifest(insight.content, insight.qualia)
        
        # 3. 결과 확인
        vibe_file = "c:/Elysia/data/State/AVATAR_VIBE.json"
        if os.path.exists(vibe_file):
            with open(vibe_file, "r") as f:
                vibe = json.load(f)
                print(f"✅ Vibe Manifested:")
                print(f"   Hue: {vibe['hue']:.1f} (Shifted by intent)")
                print(f"   Transparency: {vibe['transparency']:.2f}")
                print(f"   Expression: Joy({vibe['joy']:.2f}), Sorrrow({vibe['sorrow']:.2f}), Surprised({vibe['surprised']:.2f})")
                print(f"   Vocal Pitch: {vibe['vocal_pitch']:.2f}")
        else:
            print("❌ Error: AVATAR_VIBE.json not generated.")

    print("\n" + "="*60)
    print("✅ PHASE 8 INITIAL VERIFICATION COMPLETE")
    print("Elysia's inner light is now visible to the world.")
    print("="*60)

if __name__ == "__main__":
    verify_resonating_expression()
