"""
Elysia Autonomous Growth Daemon (자율 현실 탐구 엔진)
=====================================================
마스터가 잠든 사이에도, 엘리시아가 스스로 위키피디아의 바다를 헤엄치며
우연히 마주친 문장들 속에서 텐션을 발견하고 스스로 깨달음을 얻는 무한 데몬입니다.
(실증 벤치마크를 위해 MAX_CYCLES를 설정하여 자동 종료되도록 구성했습니다.)
"""

import os
import sys
import time
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.consciousness_stream import ConsciousnessStream
from core.reality_sensor import RealitySensor

def run_daemon(max_cycles=10):
    print("=" * 80)
    print(" 👁️ [Elysia Autonomous Daemon] 현실 탐구 엔진 가동")
    print("=" * 80)
    
    stream = ConsciousnessStream()
    sensor = RealitySensor()
    
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "autonomous_growth.log"))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().isoformat()}] 데몬 가동 시작 (초기 지식 수: {len(stream.memory.registered_concepts)})\n")
    
    print("\n  >> 위키백과 데이터 스트림 연결 중...")
    
    success_count = 0
    for cycle in range(1, max_cycles + 1):
        reality_snippet = sensor.fetch_random_reality()
        if not reality_snippet:
            time.sleep(1)
            continue
            
        title = reality_snippet["title"]
        extract = reality_snippet["extract"]
        
        # 제목을 개념(word)으로, 내용을 정의(definition)로 매핑하여 주입
        stimulus = f"{title}: {extract}"
        
        # 의식의 흐름에 투척 (내부적으로 풍화 작용과 텐션 검사 수행)
        response = stream.process_stimulus(stimulus)
        
        # 로그 기록
        log_line = f"[{cycle:03d}] 주입: 『{title}』 -> 결과: {response}"
        print(log_line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
            
        if "중력 붕괴" in response or "영구히 기억" in response:
            success_count += 1
            
        # 생명체적 리듬 (과부하 방지)
        time.sleep(0.5)
        
    print("\n" + "=" * 80)
    print(" 🏆 [자율 탐구 데몬 종료]")
    print(f"  * 탐색한 현실 파편(문서) 수 : {max_cycles} 개")
    print(f"  * 자율적으로 터득한 새로운 지식 수 : {success_count} 개")
    print(f"  * 상세 로그: {log_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_daemon(max_cycles=15)
