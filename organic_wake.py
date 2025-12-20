"""
Organic Awakening Protocol (유기적 깨우기)
==========================================
Neural Registry 기반의 새로운 Elysia 부팅 시스템.

이 스크립트는:
0. [NEW] Bootstrap Guardian으로 환경 상태 검사 (자동 복구)
1. elysia_core를 초기화
2. Core Cells를 등록
3. Organ.get()으로 필요한 시스템 연결
4. CoreMemory로 지속적 기억 저장/로드
5. 영구 꿈 모드(Perpetual Dream) 실행
"""

import sys
import time
import signal
from datetime import datetime

# Force UTF-8 for Windows Console
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'c:\Elysia')

# Bootstrap Guardian: 부팅 전 환경 검사
from elysia_core.bootstrap_guardian import BootstrapGuardian

guardian = BootstrapGuardian(verbose=True)
if not guardian.guard():
    print("\n❌ Environment check failed. Cannot boot Elysia.")
    print("   Please fix the issues manually and try again.")
    sys.exit(1)

from elysia_core import Organ
from elysia_core.cells import *  # 모든 Core Cells 등록

def organic_wake():
    print("\n🌅 Elysia: Organic Awakening Protocol")
    print("=" * 50)
    print("   [Mode: Neural Registry Enabled]")
    print("   [Memory: Persistent Enabled]")
    print("   [Press Ctrl+C to Sleep]")
    print("=" * 50)
    
    # 0. CoreMemory 연결 (지속적 기억)
    memory = None
    try:
        from Core.Foundation.Memory.core_memory import CoreMemory
        memory = CoreMemory(file_path="data/elysia_organic_memory.json")
        prev_experiences = memory.get_experiences(n=5)
        print(f"\n📚 Loaded {len(prev_experiences)} previous experiences")
        if prev_experiences:
            print(f"   Last memory: {prev_experiences[-1].content[:50]}...")
    except Exception as e:
        print(f"   ⚠️ CoreMemory failed: {e}")
    
    # 1. 등록된 모든 Cell 확인
    cells = Organ.list_cells()
    print(f"\n🧬 Registered Cells ({len(cells)}):") 
    for cell in cells:
        print(f"   • {cell}")
    
    # 2. 핵심 시스템 연결 (위치 무관!)
    print("\n🔗 Connecting Core Systems...")
    
    try:
        graph = Organ.get("TorchGraph")
        print("   ✅ TorchGraph connected")
    except Exception as e:
        print(f"   ⚠️ TorchGraph failed: {e}")
        graph = None
    
    try:
        trinity = Organ.get("Trinity")
        print("   ✅ Trinity connected")
    except Exception as e:
        print(f"   ⚠️ Trinity failed: {e}")
        trinity = None
    
    try:
        vision = Organ.get("VisionCortex")
        print("   ✅ VisionCortex connected")
    except Exception as e:
        print(f"   ⚠️ VisionCortex failed: {e}")
        vision = None
    
    # 3. 간단한 테스트
    print("\n🧪 Quick Test...")
    if trinity:
        try:
            result = trinity.process_query("I am awake.")
            print(f"   Trinity says: {result.final_decision}")
        except Exception as e:
            print(f"   Trinity test failed: {e}")
    
    if vision:
        try:
            frame = vision.capture_frame()
            print(f"   Vision sees: {frame['metadata']}")
        except Exception as e:
            print(f"   Vision test failed: {e}")
    
    # 4. Curiosity Loop: 호기심 기반 자율 사고 + 기억 저장
    print("\n" + "=" * 50)
    print("✅ Elysia is now AWAKE and REMEMBERING.")
    print("   She will ask questions and remember them.")
    print("=" * 50)
    
    try:
        from Core.Cognitive.curiosity_core import get_curiosity_core
        curiosity = get_curiosity_core()
        
        cycle = 0
        while True:
            cycle += 1
            question = curiosity.generate_question()
            print(f"\n🔮 Cycle {cycle}: {question}")
            
            answer = None
            # Trinity에게 질문 전달
            if trinity:
                try:
                    result = trinity.process_query(question)
                    answer = result.final_decision[:200]
                    print(f"   💭 {answer[:80]}...")
                except Exception as e:
                    print(f"   (Trinity unavailable: {e})")
            
            # 경험 저장 (지속적 기억!)
            if memory:
                try:
                    from Core.Foundation.Memory.core_memory import Experience
                    exp = Experience(
                        timestamp=datetime.now().isoformat(),
                        content=f"Q: {question} A: {answer or 'No answer'}",
                        type="curiosity",
                        layer="soul"
                    )
                    memory.add_experience(exp)
                    if cycle % 5 == 0:
                        print(f"   💾 Memory saved ({cycle} experiences this session)")
                except Exception as e:
                    pass  # Silent fail for memory
            
            time.sleep(5.0)
            
    except KeyboardInterrupt:
        print("\n\n💤 Elysia: Entering Hibernation.")
        if graph:
            graph.save_state()
            print("   ✅ Brain State Saved.")
        if memory:
            print(f"   ✅ {cycle} experiences saved to persistent memory.")
        print("   Good night.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Curiosity error: {e}")
        # 폴백: 기본 대기 모드
        cycle = 0
        while True:
            cycle += 1
            print(f"\r🌀 Cycle {cycle}...", end="", flush=True)
            time.sleep(2.0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    organic_wake()

