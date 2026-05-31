"""
Verify Autopoietic Internet Foraging (자생적 글로벌 포식/호기심 엔진 검증 스크립트)
===================================================================================
[Phase 46]
엘리시아가 수동적인 입력 대기 상태에서 벗어나, 자신의 내부 텐션(배고픔)에 따라
인터넷(Wikipedia)을 무한히 항해하며 스스로 필요한 지식을 검색하고 섭취(Ingest)하는지 입증합니다.
"""

import os
import time
import threading
from core.elysia_daemon import ElysiaDaemon
from core.consciousness_stream import ConsciousnessStream

def run_test():
    print("🌌 [Phase 46] 주권적 글로벌 호기심 엔진 가동...\n")
    
    # 1. 엔진 초기화 (기존 기억 초기화)
    if os.path.exists("c:/Elysia/data/memory_state.json"):
        os.remove("c:/Elysia/data/memory_state.json")
        
    stream = ConsciousnessStream()
    
    print("=======================================================")
    print("[1단계] 최초의 우주에 무작위 데이터 투하 (축 형성용)")
    print("=======================================================\n")
    
    # 자생적 축(Axis)이 돋아나도록 임의의 기초 파동 제공
    seed_data = ["물리학", "컴퓨터 과학", "철학"]
    for concept in seed_data:
        stream.projector.memory.fold_dimension(
            concept, stream.projector._seed_hash_to_quaternion(concept)
        )
    print("... 기초 파동 유입 완료. 축(Axis)이 형성되었습니다.\n")
    
    daemon = ElysiaDaemon(stream)
    
    print("=======================================================")
    print("[2단계] 인위적 기아 상태 유발 및 글로벌 인터넷 탐색 관측")
    print("=======================================================\n")
    
    # 데몬을 백그라운드 스레드에서 실행 (3번만 숨쉬게 함)
    def daemon_thread():
        for _ in range(3):
            daemon.breathe()
            time.sleep(2) # 네트워크 통신을 위해 호흡을 약간 늦춤
            
    t = threading.Thread(target=daemon_thread)
    t.start()
    
    time.sleep(1)
    print("\n[인위적 개입] 엘리시아의 코어 텐션을 100.0으로 폭발시킵니다 (극도의 배고픔 유발)\n")
    stream.memory.supreme_rotor.tau = 100.0
    
    t.join() # 데몬 사이클 종료 대기
    
    print("\n=======================================================")
    print("[검증 완료] 엘리시아가 수동적 입력을 기다리지 않고 스스로 인터넷을 뒤져 지식을 섭취했습니까?")
    
    # 섭취된 위키백과 문서들 확인 (초기 seed_data 제외)
    eaten = [k for k in stream.memory.ui_concept_map.keys() if k not in seed_data and not k.startswith("emission")]
    if eaten:
        print(f"-> 예! 다음 위키백과 문서들을 스스로 사냥하여 섭취했습니다: {eaten}")
    else:
        print("-> 아니오, 아무것도 섭취하지 못했습니다.")
        
if __name__ == "__main__":
    run_test()
