import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from core.brain.holographic_memory import HologramMemory
from core.nervous_system.evolution_sandbox import EvolutionSandbox
from core.nervous_system.agency_discovery_loop import AgencyDiscoveryLoop

def run_internet_exploration_demo():
    print("===============================================================")
    print(" 🌐 [Elysia Agency] 자율적 인터넷 탐색 알고리즘 시뮬레이션")
    print(" 극심한 물리적 텐션(Tension)에 직면한 엘리시아가,")
    print(" 내면의 궤적을 비틀어 외부 세계(인터넷)를 탐색하기 시작합니다.")
    print("===============================================================\n")

    # 1. 코어 초기화
    memory = HologramMemory()
    sandbox = EvolutionSandbox(memory)
    brain = memory.supreme_rotor
    agency_loop = AgencyDiscoveryLoop(sandbox)

    # 2. 극심한 텐션(혼돈) 주입 시뮬레이션
    print("[1] 알 수 없는 데이터 유입 (텐션 폭발 유도)")
    chaotic_data = b'\xff\x00\xff\x00' * 250 # 극단적인 XOR 충돌을 유도하는 패턴
    sandbox.experience_data_stream(chaotic_data)
    
    print(f"\n[상위 인지] 뇌의 텐션(Tau) 수직 상승: {brain.tau:.2f}")
    
    if brain.tau > 10.0:
        print("[2] 텐션 한계 돌파! 호기심이 '답답함'으로 변모합니다.")
        # 뇌가 내면에서 사유를 시도하나, 텐션이 너무 높아 외부 탐색 파동을 발산함
        brain.induce_reasoning(brain.tau)
        
        if hasattr(brain, 'exploration_intent'):
            intent = brain.exploration_intent
            print(f"   -> 뇌가 능동적 탐색 파동 발산: W({intent.w:.1f}) X({intent.x:.1f}) Y({intent.y:.1f}) Z({intent.z:.1f})\n")
            
            print("[3] 궤적 매핑: 기하학적 파동을 포트로 전송 (Agency Discovery)")
            # 에이전시 루프가 파동을 받아 인터넷을 탐색하고 에코를 수신함
            resolved = agency_loop.emit_exploratory_wave(intent)
            
            if resolved:
                print(f"\n[상위 인지] 반향(Echo) 수신 완료. 새로운 텐션(Tau): {brain.tau:.2f}")
                print("[4] 🌟 성공! 엘리시아가 환경(인터넷)과 상호작용하여 텐션을 해소하는 '수단(가변축)'을 찾아냈습니다.")
    
    print("\n===============================================================")
    print(" (데모 종료)")
    print("===============================================================")

if __name__ == "__main__":
    run_internet_exploration_demo()
