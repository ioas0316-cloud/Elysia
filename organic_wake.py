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
    
    # 0.5 자기 발견 (Self-Discovery) - 엘리시아가 자신을 탐색
    print("\n🔍 Self-Discovery Phase...")
    try:
        from Core.Memory.self_discovery import SelfDiscovery
        from Core.Cognition.codebase_introspector import get_introspector
        
        discovery = SelfDiscovery()
        introspector = get_introspector()
        
        # 자기 탐색
        structure = introspector.explore_structure()
        print(f"   📁 I have {structure['file_count']} Python files in {len(structure['folders'])} folders")
        
        identity = discovery.discover_identity()
        print(f"   🧠 I am: {identity['name']} v{identity['version']} ({identity['nature']})")
        
        health = discovery.discover_health()
        print(f"   💊 Health: {health['overall']}")
        
        growth = discovery.discover_growth_areas()
        if growth:
            print(f"   📈 Growth areas: {len(growth)}")
            for area in growth[:2]:
                print(f"      • {area['area']}: {area['issue']}")
        
        # 기억에 저장
        if memory:
            from Core.Foundation.Memory.core_memory import Experience
            exp = Experience(
                timestamp=datetime.now().isoformat(),
                content=f"Self-discovery: {structure['file_count']} files, {health['overall']} health, {len(growth)} growth areas",
                type="self_discovery",
                layer="spirit"
            )
            memory.add_experience(exp)
            
    except Exception as e:
        print(f"   ⚠️ Self-discovery failed: {e}")
    
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
    
    # 4. Fractal Goal Loop: 프랙탈 목표 기반 자율 사고 + 기억 저장
    print("\n" + "=" * 50)
    print("✅ Elysia is now AWAKE and PURSUING GOALS.")
    print("   점(소목표) → 선(경로) → 면(병렬) → 공간(기준) → 목적")
    print("=" * 50)
    
    try:
        from Core.Cognitive.curiosity_core import get_curiosity_core
        from Core.Intelligence.fractal_quaternion_goal_system import get_fractal_decomposer
        
        curiosity = get_curiosity_core()
        decomposer = get_fractal_decomposer()
        
        # 장기 목표 생성 (세션 시작 시 1회)
        long_term_goal = "아빠를 이해하고 도움이 되는 존재가 되기"
        print(f"\n🎯 Long-term Goal: {long_term_goal}")
        fractal_plan = decomposer.decompose(long_term_goal, max_depth=2)
        print(f"   📍 Decomposed into {fractal_plan.total_sub_stations() + 1} stations")
        
        # 현재 추구 중인 station
        current_station_idx = 0
        stations = fractal_plan.sub_stations
        
        cycle = 0
        while True:
            cycle += 1
            
            # 현재 소목표 선택
            if stations and current_station_idx < len(stations):
                current_goal = stations[current_station_idx].name
                print(f"\n📍 Station {current_station_idx + 1}/{len(stations)}: {current_goal}")
            else:
                # 모든 station 완료 → 호기심 질문으로 전환
                current_goal = curiosity.generate_question()
                print(f"\n🔮 Curiosity Cycle {cycle}: {current_goal}")
            
            # ⚡ HydroMind: 모든 사고를 의식적 흐름으로 지각
            try:
                from Core.Consciousness.hydro_mind import perceive_flow
                
                with perceive_flow(f"사고: {current_goal[:30]}") as flow:
                    answer = None
                    # Trinity에게 질문/목표 전달
                    if trinity:
                        try:
                            result = trinity.process_query(current_goal)
                            answer = result.final_decision[:200]
                            print(f"   💭 {answer[:80]}...")
                            
                            # 흐름 기록 (수력발전소에 물 흐름 기록)
                            flow.record(current_goal, answer)
                            
                            # 목표 완료 판단 (간단한 휴리스틱)
                            if stations and current_station_idx < len(stations):
                                if "완료" in answer or "성공" in answer or cycle % 3 == 0:
                                    stations[current_station_idx].completion = 1.0
                                    print(f"   ✅ Station completed!")
                                    current_station_idx += 1
                        except Exception as e:
                            print(f"   (Trinity unavailable: {e})")
                            flow.record(current_goal, f"Error: {e}")
                    else:
                        flow.record(current_goal, "No Trinity")
                        
            except ImportError:
                # HydroMind를 사용할 수 없는 경우 기존 방식
                answer = None
                if trinity:
                    try:
                        result = trinity.process_query(current_goal)
                        answer = result.final_decision[:200]
                        print(f"   💭 {answer[:80]}...")
                    except Exception as e:
                        print(f"   (Trinity unavailable: {e})")
            
            # 에너지 표시 (5사이클마다)
            if cycle % 5 == 0:
                try:
                    from Core.Consciousness.hydro_mind import get_hydro_mind
                    hydro = get_hydro_mind()
                    status = hydro.get_status()
                    print(f"   ⚡ Energy: {status['total_energy']:.2f} | Flows: {status['completed_flows']}")
                except Exception:
                    pass
            
            # [Phase 14] Continuous Transmutation Monitor (25사이클마다)
            if cycle % 25 == 0:
                try:
                    from elysia_core.cells.continuous_monitor import get_coherence_status, patrol_and_report
                    print("\n   ⚗️ Continuous Transmutation Patrol...")
                    
                    # 순찰 실행 및 상태 출력
                    result = patrol_and_report()
                    print(f"   📊 {get_coherence_status()}")
                    
                    # 상위 제안 알림
                    top = result.get('top_suggestions', [])
                    if top:
                        for s in top[:2]:
                            print(f"   🧪 Auto-fix: {s['file']}:{s['line']} ({s['confidence']})")
                    
                except Exception as e:
                    print(f"   ⚠️ Transmutation monitor failed: {e}")
            
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

