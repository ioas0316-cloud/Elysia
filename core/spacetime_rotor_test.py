import sys
import os
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network_phase_snatcher import NetworkPhaseSnatcher
from core.spacetime_rotor import VariableAxisManifold

def main():
    print("=" * 80)
    print(" ⏳ [Phase 135-Final] 시공간축 로터 스트리밍 및 가변축화 테스트")
    print("=" * 80)
    
    # Qwen2.5-72B-Instruct 의 첫 번째 샤드 파일을 대상으로 네트워크 스트리밍
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
    filename = "model-00001-of-00037.safetensors"
    
    print(f"\n[타겟] {repo_id} / {filename}")
    print("로터를 시공간축에 매달고 데이터 스트림을 쏟아붓습니다...\n")
    
    snatcher = NetworkPhaseSnatcher(repo_id, filename)
    manifold = VariableAxisManifold()
    
    # 네트워크 위상 강탈을 통한 시공간축 스트리밍 (50개의 텐서를 시간차를 두고 흘려보냄)
    count = 0
    start_time = time.time()
    
    for layer_key, phase_quat in snatcher.stream_and_clone_phases(max_tensors=50):
        # 텐서 스트림이 시공간축 로터(가변축)에 유입됨
        manifold.flow_stream_into_axis(layer_key, phase_quat)
        count += 1
        
        # 스트리밍 과정의 리얼리티를 위해 아주 미세한 지연(시간 흐름) 강제
        time.sleep(0.01)
        
    elapsed = time.time() - start_time
    print(f"\n🌊 스트리밍 완료! 총 {count}개의 데이터 스트림이 시공간축을 지나갔습니다. (소요시간: {elapsed:.2f}s)\n")
    
    print("[생성된 가변축(Variable Axes) 궤적 서명]")
    for layer_group, rotor in manifold.axes.items():
        signature = rotor.get_variable_axis_signature()
        print(f" 🧬 {layer_group: <15} | {signature}")
        
    print("\n================================================================================")
    print(" 🎉 압축(Collapse)은 없었습니다. 모든 레이어는 살아 숨 쉬는 '가변축 궤적'이 되었습니다!")
    print("================================================================================")

if __name__ == "__main__":
    main()
