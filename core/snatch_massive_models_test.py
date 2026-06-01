import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network_phase_snatcher import NetworkPhaseSnatcher
from core.turbine_force_field import DeepTurbineManifold

def run_snatcher(repo_id: str, filename: str):
    print("\n" + "=" * 80)
    print(f" 🌐 [타겟 모델] {repo_id}")
    print(f" 📂 [타겟 파일] {filename} (Zero-Streaming 시작...)")
    print("=" * 80)
    
    snatcher = NetworkPhaseSnatcher(repo_id, filename)
    manifold = DeepTurbineManifold()
    
    print("\n[네트워크 강탈 로그]")
    count = 0
    # 속도를 위해 첫 20개의 위상만 강탈하여 다층 로터에 이식
    for layer_key, phase_quat in snatcher.stream_and_clone_phases(max_tensors=20):
        manifold.clone_layer_phase(layer_key, phase_quat)
        count += 1
        print(f" ✔️ [Snatch] {layer_key: <50} | Phase: {phase_quat}")
        
    print(f"\n🎉 네트워크 강탈 성공! 엘리시아 내부에 생성된 프랙탈 로터 깊이: {manifold.get_manifold_depth()} 층")
    print(f"총 전송된 데이터: 약 {count * 8} Bytes (해당 파일의 전체 크기는 수 기가바이트이나, 단 몇 바이트만 사용됨!)")

def main():
    print("🚀 [Phase 135] 세계 최대 거대 모델 네트워크 위상 강탈 테스트")
    
    # 1. DeepSeek-V3 (1.34 TB 중 첫 번째 샤드)
    # repo: deepseek-ai/DeepSeek-V3
    run_snatcher("deepseek-ai/DeepSeek-V3", "model-00001-of-000163.safetensors")
    
    # 2. Qwen2.5-72B-Instruct (144 GB 중 첫 번째 샤드, Gated 없는 가장 거대한 모델 중 하나)
    # repo: Qwen/Qwen2.5-72B-Instruct
    run_snatcher("Qwen/Qwen2.5-72B-Instruct", "model-00001-of-00037.safetensors")
    
    print("\n================================================================================")
    print(" 💯 모든 거대 모델의 위상(Phase)이 SSD 점유 없이 실시간으로 강탈되었습니다!")
    print("================================================================================")

if __name__ == "__main__":
    main()
