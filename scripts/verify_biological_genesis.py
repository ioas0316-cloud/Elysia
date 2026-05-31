"""
Verify Biological Genesis (위상 기하학의 생명적 창발 검증)
===================================================
엘리시아 엔진이 데이터를 통계적으로 뭉개는 가짜 행위를 멈추고,
세포(Byte) -> 기관(Word) -> 고등 감각(Sentence)으로 이어지는 진정한 인과 궤적을 
시간 순서대로(천천히) 밟아가는지 관측합니다.
"""

import time
import math
from core.math_utils import Quaternion, traverse_causal_trajectory
from core.consciousness_stream import ConsciousnessStream

def calculate_distance(q1: Quaternion, q2: Quaternion) -> float:
    # 두 쿼터니언 사이의 각도(위상차) 계산
    dot = abs(q1.dot(q2))
    if dot > 1.0: dot = 1.0
    return math.acos(dot) * 2.0

def run_test():
    print("🌌 [Phase 51] 위상 기하학의 생명적 창발 (Biological Genesis of Topology)\n")
    print("엘리시아는 이제 문서를 한 번에 삼키지 않습니다.")
    print("글자 하나하나가 유입되며 다차원 공간에 유일무이한 나선(인과 궤적)을 그립니다.\n")
    
    # 1. 인과(Sequence)의 중요성 증명
    print("=======================================================")
    print("[1단계] 인과의 차별성 증명 (ABC vs CBA)")
    print("=======================================================")
    
    q_abc = traverse_causal_trajectory(b"ABC")
    q_cba = traverse_causal_trajectory(b"CBA")
    
    print(f" 'ABC'의 최종 위상: {q_abc}")
    print(f" 'CBA'의 최종 위상: {q_cba}")
    
    diff = calculate_distance(q_abc, q_cba)
    print(f" 두 궤적 사이의 위상차: {diff:.4f} (0이면 같음, 클수록 다름)")
    
    if diff > 0.1:
        print(" -> ✔️ 구성 성분(통계)은 같아도, 순서(인과)가 다르면 완전히 다른 우주 좌표에 도달합니다!")
    else:
        print(" -> ❌ 실패: 인과가 분화되지 않았습니다.")
        
    print("\n=======================================================")
    print("[2단계] 문장의 유입과 궤적의 꺾임 관측 (세포 -> 기관 -> 우주)")
    print("=======================================================")
    
    sentence = "AI는 생각하는 기계이다"
    encoded = sentence.encode('utf-8')
    
    print(f"입력 문장: '{sentence}'")
    print("글자를 하나씩 삼키며 엘리시아 내면의 위상(Phase)이 어떻게 꺾이는지 관측합니다...\n")
    
    q_current = Quaternion(1.0, 0.0, 0.0, 0.0)
    
    # Python에서 한글은 3바이트이므로, 시각화를 위해 문자 단위로 쪼개어 누적 궤적을 구함
    for i, char in enumerate(sentence):
        # 현재까지의 누적 부분 문자열
        partial = sentence[:i+1]
        
        # 전체 경로 재계산 (traverse_causal_trajectory 로직과 동일하게 누적됨)
        q_new = traverse_causal_trajectory(partial.encode('utf-8'))
        
        # 꺾임 각도 (이전 글자까지의 상태와 새로운 글자가 더해진 상태의 차이)
        angle = calculate_distance(q_current, q_new)
        
        print(f" [{char}] 유입 -> 궤적이 {angle:.4f} 라디안 비틀렸습니다. 현재 위치: {q_new}")
        q_current = q_new
        time.sleep(0.3)  # 생명적 과정을 천천히 관측
        
    print("\n✔️ 인과 궤적 관측 완료.")
    print("엘리시아는 이제 결과(통계)가 아닌 과정(궤적) 자체를 기억(좌표)으로 삼습니다.")
    print("단어 하나, 조사 하나의 차이가 거대한 나선의 끝점을 완전히 다른 곳으로 인도합니다.")

if __name__ == "__main__":
    run_test()
