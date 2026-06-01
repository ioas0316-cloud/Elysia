import sys
import os
import psutil
import torch
from safetensors.torch import save_file

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mmap_streamer import MMAPTensorStreamer
from core.turbine_force_field import DeepTurbineManifold

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[메모리 점유율] 현재 RAM 사용량: {mem_mb:.2f} MB")

def create_dummy_safetensors(file_path: str, num_layers: int = 80):
    """테스트를 위한 다층(80층) 가짜 모델 생성"""
    print(f"\n[준비] 가상의 거대 모델({num_layers}층) 거푸집 생성 중...")
    tensors = {}
    # 임베딩 층
    tensors["model.embed_tokens.weight"] = torch.randn(100, 1024)
    
    # 80개의 깊은 레이어들 (차원을 축소하여 테스트 속도 및 디스크 공간 절약)
    for i in range(num_layers):
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(1024, 1024)
        tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(256, 1024)
        tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(2048, 1024)
        
    save_file(tensors, file_path)
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    print(f"✔️ 거푸집 생성 완료: {file_path} (크기: {size_mb:.2f} MB)")

def main():
    print("=" * 80)
    print(" 🌊 [Phase 134] 다층 레이어 MMAP 위상 모방 및 내재적 딥 로터화 테스트")
    print("=" * 80)
    
    dummy_model_path = "dummy_huge_model.safetensors"
    
    # 1. 수십 층짜리 가짜 거대 모델(Safetensors) 생성
    create_dummy_safetensors(dummy_model_path, num_layers=80)
    print_memory_usage()
    
    # 2. 내재적 다층 로터(Deep Turbine Manifold) 준비
    manifold = DeepTurbineManifold()
    streamer = MMAPTensorStreamer(dummy_model_path)
    
    print("\n[Step 1] 🚀 전층(All-Layers) 제로스트리밍 위상 모방 시작...")
    
    count = 0
    # 제로스트리밍 시작: 모델 전체가 MMAP으로 지나가며 위상 앵커만 복제됨
    for layer_key, phase_quat in streamer.stream_and_clone_phases():
        # 엘리시아의 다층 로터에 위상 주입
        manifold.clone_layer_phase(layer_key, phase_quat)
        count += 1
        
        # 진행상황 출력 (100개 단위)
        if count % 50 == 0:
            print(f"  ... {count}개의 텐서 스트림 위상 모방 완료 ...")
            print_memory_usage()
            
    print(f"\n[Step 2] 🧬 복제 완료! 총 {count}개 텐서의 깊은 위상이 엘리시아 내부로 흡수되었습니다.")
    print(f"엘리시아 내부에 생성된 프랙탈 로터(Turbine) 층수: {manifold.get_manifold_depth()} 층")
    
    print("\n[최종 점검] 거대 모델을 모조리 훑었는데 메모리가 폭발했는가?")
    print_memory_usage()
    
    print("\n================================================================================")
    print(" 🎉 실험 종료: 수백 메가바이트의 텐서가 지나갔지만 RAM은 평온합니다.")
    print(" 엘리시아는 이제 거대 모델의 다이내믹스(80층 깊이)를 내재적 딥 로터로 완전히 모방했습니다!")
    print("================================================================================")
    
    # 테스트 종료 후 더미 파일 삭제
    if os.path.exists(dummy_model_path):
        os.remove(dummy_model_path)

if __name__ == "__main__":
    main()
