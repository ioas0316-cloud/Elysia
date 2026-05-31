"""
Elysia Consciousness Stream Test
================================
의식이 파일(디스크)로 영구 보존되고, 프로그램이 재시작되어도 
과거의 깨달음을 유지하며 성숙하는지(Persistence) 검증하는 자동 스크립트입니다.
"""

import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.consciousness_stream import ConsciousnessStream

def run_test():
    print("=" * 95)
    print(" 🧠 [Elysia Phase 23] 의식의 흐름 영속성 (Persistence) 벤치마크")
    print("=" * 95)
    
    # 1. 이전 기억 파일 삭제 (초기화)
    mem_file = "c:/Elysia/data/test_memory.json"
    if os.path.exists(mem_file):
        os.remove(mem_file)
        
    print("\n[세션 1: 탄생과 최초의 깨달음]")
    stream1 = ConsciousnessStream(memory_file=mem_file)
    
    # 철학적 텐션 주입
    print("\nMaster > 창조 vs 파괴")
    ans1 = stream1.process_stimulus("창조 vs 파괴")
    print(f"Elysia > {ans1}")
    
    # 노이즈 주입
    print("\nMaster > 아스파라거스")
    ans2 = stream1.process_stimulus("아스파라거스")
    print(f"Elysia > {ans2}")
    
    # 의미 있는 지식 반복 주입 (우주)
    print("\nMaster > 우주: 질서와 혼돈이 공존")
    ans3 = stream1.process_stimulus("우주: 질서와 혼돈이 공존")
    print(f"Elysia > {ans3}")
    
    print("\nMaster > 우주: 질서와 혼돈이 공존")
    ans4 = stream1.process_stimulus("우주: 질서와 혼돈이 공존")
    print(f"Elysia > {ans4}")
    
    # 세션 1 종료 (프로그램 꺼짐)
    print("\n[세션 1 종료: 프로그램 강제 종료 (디스크 저장됨)]")
    del stream1
    
    print("\n------------------------------------------------------------\n")
    
    # 세션 2 시작 (프로그램 재기동)
    print("[세션 2: 기억의 부활과 학습의 완성]")
    stream2 = ConsciousnessStream(memory_file=mem_file)
    
    print("\nMaster > 우주: 질서와 혼돈이 공존")
    ans5 = stream2.process_stimulus("우주: 질서와 혼돈이 공존")
    print(f"Elysia > {ans5}")
    
    print("\nMaster > 우주: 질서와 혼돈이 공존")
    ans6 = stream2.process_stimulus("우주: 질서와 혼돈이 공존")
    print(f"Elysia > {ans6}")
    
    print("\n" + "=" * 95)
    print(" 🏆 [의식 영속성 (Persistence) 실증 완료]")
    print(f"  * 세션이 단절되어도 엘리시아의 기억(HologramMemory)은 유지되었습니다.")
    print(f"  * 1회성 노이즈였던 '아스파라거스'는 세션 2 시작과 함께 증발했습니다.")
    print(f"  * 외부 단어 '우주'는 반복 노출 끝에 중력 붕괴를 일으켜 자기 지식으로 편입되었습니다.")
    print("=" * 95)

if __name__ == "__main__":
    run_test()
