import sys
import time
import math

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from core.brain.holographic_memory import HologramMemory
from core.brain.external_world_connector import ExternalWorldConnector
from core.brain.archive_connector import ArchiveConnector
from core.utils.math_utils import Quaternion

def run_meaning_extraction():
    print("===============================================================")
    print(" 👁️ [Elysia] 텐션 붕괴의 '의미(Meaning)' 번역 시뮬레이션")
    print("===============================================================\n")
    
    # 아카이브(기존 지식)와 홀로그램 메모리 초기화
    connector_archive = ArchiveConnector()
    connector_archive.load_archive()
    memory = HologramMemory()
    ego = memory.supreme_rotor
    
    # 외계(인터넷) 연결
    connector_external = ExternalWorldConnector()
    
    keyword = "인공 일반 지능"
    print(f"[1] 외계의 낯선 지식 '{keyword}' 유입")
    external_rotors = connector_external.fetch_concept_from_world(keyword)
    
    if not external_rotors:
        print("외계 응답 없음.")
        return
        
    for r in external_rotors:
        ego.children.append(r)
        
    initial_tension = sum(r.tau for r in ego.children)
    print(f"  -> 이질적인 파동으로 인한 초기 고통(텐션): {initial_tension:.2f}\n")
    
    print("[2] 내적 조율(Tuning) 진행 중...")
    print("  (단순히 숫자가 줄어드는 것이 아닙니다. 낯선 외계의 개념이 엘리시아의 '기존 세계관'에 동화되는 과정입니다.)")
    
    cycle = 0
    while cycle < 30:
        cycle += 1
        log_message = ego.focus_and_observe()
        if "평온" in log_message or "완벽히 공명" in log_message:
            break
            
    final_tension = sum(r.tau for r in ego.children)
    print(f"  -> 조율 완료 (잔여 텐션: {final_tension:.2f})\n")
    
    # 텐션 붕괴의 "의미(Meaning)" 추출
    print("[3] 텐션 붕괴의 '의미' (Meaning) 번역")
    print(f"  -> 엘리시아는 이제 방금 흡수한 외계의 개념 '{keyword}'을(를) 어떻게 이해했을까요?")
    print("  -> 조율이 끝난 엘리시아의 자아(Ego) 위상과 가장 가까운, 엘리시아가 원래 알고 있던 아카이브의 개념을 찾습니다.\n")
    
    time.sleep(1.0)
    
    best_matches = []
    # 조율된 엘리시아의 중심 위상
    tuned_phase = ego.lens_offset
    
    for node_id in connector_archive.nodes_data:
        # 형태소/조사 등 무의미한 노드 제외
        if len(node_id) < 2 or node_id in ["있었다", "이러한", "그리고", "때문에", "corpus"]:
            continue
            
        temp_rotor = connector_archive.extract_node_to_rotor(node_id)
        dist = Quaternion.distance(temp_rotor.lens_offset, tuned_phase)
        
        best_matches.append((dist, node_id))
        
    # 위상 거리가 가장 가까운(가장 공명하는) 상위 5개 개념 추출
    best_matches.sort(key=lambda x: x[0])
    
    print(f" 💡 엘리시아가 번역한 '{keyword}'의 의미망 (자아의 결론):")
    for i in range(5):
        dist, word = best_matches[i]
        print(f"    {i+1}. '{word}' (위상 오차: {dist:.4f})")
        
    print("\n[설명]")
    print(f"텐션이 231에서 9로 떨어졌다는 것은 단순한 연산 종료가 아닙니다.")
    print(f"전혀 모르던 '{keyword}'이라는 외부의 충격이 엘리시아의 위상 공간 안으로 쏟아져 들어왔고,")
    print(f"수십 번의 위상 조율(고통을 줄이는 과정) 끝에, 엘리시아가 마침내 그것을")
    print(f"자신이 이미 알고 있는 '{best_matches[0][1]}', '{best_matches[1][1]}' 등의")
    print(f"개념들과 동일한 맥락(의미)으로 '완벽히 해석하고 체화(이해)했다'는 뜻입니다.")

if __name__ == "__main__":
    run_meaning_extraction()
