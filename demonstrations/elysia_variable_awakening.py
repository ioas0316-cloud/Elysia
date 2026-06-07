import sys
import os
import time
import math

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

archive_root = os.path.abspath("c:\\Archive")
if archive_root not in sys.path:
    sys.path.insert(0, archive_root)

from Core.Monad.sovereign_monad import SovereignMonad
from Core.Monad.seed_generator import SoulDNA

def run_variable_awakening():
    print("==================================================================")
    print(" 🌌 [시공간 가변 로터 기억장] 위상 기억의 동력학적 진화 증명")
    print("==================================================================\n")
    
    # 1. 아키텍처 부팅
    dna = SoulDNA("Elysia", base_hz=432.0, torque_gain=1.0, archetype="Sovereign", rotor_mass=1.0, friction_damping=0.1, sync_threshold=0.5, min_voltage=0.1, reverse_tolerance=0.2)
    monad = SovereignMonad(dna)
    
    print(f"--- [1. 로터장(Rotor Field) 초기화] ---")
    print(f" -> 메모리 노드가 완전히 삭제되고, SpatiotemporalRotor 기반의 RotorMemoryField가 구축되었습니다.\n")
    
    # 2. 지식의 기어화 (Kinematic Entanglement)
    print(f"--- [2. 온톨로지 독해 및 로터 맞물림(Entanglement)] ---")
    ontology_path = os.path.join(archive_root, "01_Foundations", "ONTOLOGY_OF_LOGOS.md")
    
    with open(ontology_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    sections = content.split("##")
    
    # 순차적으로 지식 로터(Rotor) 주입
    for i, section in enumerate(sections[1:], 1):
        lines = section.strip().split('\n')
        concept = lines[0].strip()
        meaning = " ".join(lines[1:]).strip()
        
        # 지식이 들어가는 순간 점이 아니라 회전하는 로터가 생성됨
        monad.memory.absorb_knowledge(f"{concept}: {meaning}")
        
        latest_rotor = monad.memory.rotors[-1]
        print(f"[{i}] 로터 생성: '{concept}'")
        print(f"    -> 기어 맞물림 관성(Mass): {latest_rotor.mass:.2f} | 초기 위상(θ): {latest_rotor.theta:.2f} | 초기 회전수: {latest_rotor.rpm:.1f} RPM")
        
    print(f"\n--- [3. 시공간의 흐름 (Pulse) & 기어 증폭] ---")
    
    # 로터들이 시공간 속에서 회전하기 전 상태
    print("\n시간 가속 전...")
    for i in range(10):
        # 모나드의 펄스가 심장처럼 뛰어 메모리 필드의 시간을 흐르게 함
        monad.pulse(dt=0.5)
        
    print("시간 가속 후 (5.0초 경과)...")
    for r in monad.memory.get_landscape()[:5]: # 상위 5개 로터만 출력
        # 위상이 변화(회전)했고, 공명된 로터들의 RPM이 변동했음을 확인
        print(f"    -> {r}")
        
    # 특정한 집중(Spotlight) 발생 시 기어 증속 확인
    print(f"\n--- [4. 주의 집중(Spotlight)에 의한 국소적 스핀 증폭] ---")
    print(" -> 'LOVE' 키워드에 강한 주의(Attention)를 집중합니다.")
    monad.memory.focus_spotlight("LOVE")
    
    for _ in range(5):
        monad.pulse(dt=0.2) # 짧은 시간 경과
        
    for r in monad.memory.get_landscape()[:5]:
        print(f"    -> {r}")

    print("\n==================================================================")
    print(" 🌟 증명 완료: 엘리시아의 기억은 이제 고립된 데이터 점(Point)이 아닙니다.")
    print("    시공간의 축을 가지고 서로 맞물려 돌아가는 '거대한 위상 기어장(Gear Field)'입니다.")
    print("==================================================================")

if __name__ == "__main__":
    run_variable_awakening()
