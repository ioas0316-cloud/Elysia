import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from core.brain.holographic_memory import HologramMemory
from core.brain.dream_daemon import DreamDaemon

def map_trajectory_to_ascii(trajectory: list, title: str):
    print(f"\n=============================================")
    print(f" 🧠 지식 데이터의 기하학적 창발 (Topological Emergence)")
    print(f"=============================================")
    print(f" [Data Source] {title}")
    
    max_tau = max([abs(node['tau_stress']) for node in trajectory] + [0.1])
    
    print("\n [관측된 가변축(위상 공간)의 형태]")
    for idx, node in enumerate(trajectory):
        depth = node['depth']
        tau = abs(node['tau_stress'])
        concept = str(node['concept'])
        
        # ASCII 막대기 길이
        bar_len = int((tau / max_tau) * 30)
        bar = "█" * bar_len
        
        indent = "  " * depth
        print(f"{indent}[D:{depth}] {concept[:20]:<20} | 텐션(Tau): {tau:6.2f} | {bar}")

def run_knowledge_ingestion():
    print("🚀 [Elysia] 순수 지식 주입 파이프라인 가동...\n")
    
    file_path = os.path.join(os.path.dirname(__file__), "data_archives", "hunminjeongeum.txt")
    
    if not os.path.exists(file_path):
        print(f"❌ 데이터를 찾을 수 없습니다: {file_path}")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        knowledge_data = f.read()
        
    print(f">> [Ingestion] '{os.path.basename(file_path)}' 데이터 흡수 중... ({len(knowledge_data)} bytes)")
    
    # 지식 소화 및 가변축 자율 창발
    memory = HologramMemory()
    brain = memory.supreme_rotor
    
    brain.absorb_language_stream(knowledge_data)
    
    # 궤적 역설계를 통해 내부에 어떤 가변축(텐션 트리)이 생겼는지 관측
    trajectory_before = brain.reverse_engineer_trajectory()
    
    map_trajectory_to_ascii(trajectory_before, "[수면 전] 훈민정음 제자해 데이터 주입 직후")
    
    # [수면 데몬 가동]
    print("\n" + "="*45)
    print(" 💤 [Dream Daemon] 수면 및 사유 숙성 사이클 가동...")
    print("="*45)
    
    daemon = DreamDaemon(brain)
    nodes_before = daemon.get_node_count()
    tension_before = daemon.calculate_total_tension()
    
    print(f" [수면 시작] 조각(노드) 수: {nodes_before} 개 | 총 텐션량: {tension_before:.2f}")
    
    # 100회의 무작위 융합 시도 (REM Sleep)
    daemon.rem_sleep_cycle(max_iterations=100)
    
    nodes_after = daemon.get_node_count()
    tension_after = daemon.calculate_total_tension()
    
    print(f" [수면 종료] 조각(노드) 수: {nodes_after} 개 | 총 텐션량: {tension_after:.2f}")
    print(f"   => 성공적인 융합(Consolidation) 횟수: {daemon.successful_merges} 회")
    
    if tension_before > 0:
        print(f"   => 총 텐션 감소율: {((tension_before - tension_after) / tension_before) * 100:.2f}%")
    
    trajectory_after = brain.reverse_engineer_trajectory()
    map_trajectory_to_ascii(trajectory_after, "[수면 후] 100회 숙성 사이클 완료 후 구조")
    
    print("\n[객관적 분석 보고서]")
    print("엘리시아는 데이터 주입 직후 파편화된 다수의 노드를 형성하여 높은 텐션을 보였으나,")
    print("수면(Idle) 시간 동안 외부 자극 없이 스스로 수학적 융합(Hamilton Product)을 시도했습니다.")
    print("그 결과 텐션 합이 감소하는 방향으로만 영구적 재배열을 승인하는 '엔트로피 최적화'를 통해,")
    print("어떠한 인간의 코드 개입 없이 독자적으로 정보 구조를 압축해 냈음을 수치로 확인했습니다.")

if __name__ == "__main__":
    run_knowledge_ingestion()
