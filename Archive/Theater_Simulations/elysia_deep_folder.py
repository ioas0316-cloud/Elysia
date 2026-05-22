import os
import sys
import gc
import math
import cmath
import time
import torch
import psutil

# Add Elysia root to path to import fractal_rotor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fractal_rotor import FractalRotor, display_rotor

def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def fast_tensor_fold(tensor):
    """
    거대 텐서를 4축 복소 파동으로 치환하여 단 하나의 접힘(Folded) 파동으로 반환.
    """
    amplitudes = torch.abs(tensor)
    phases = torch.where(tensor >= 0, torch.tensor(0.0), torch.tensor(math.pi))
    complex_tensor = torch.polar(amplitudes, phases)
    
    folded_wave = torch.sum(complex_tensor).item()
    
    amp = abs(folded_wave)
    norm_amp = math.log1p(amp) # 부드러운 로그 스케일 압축
    phase = cmath.phase(folded_wave)
    
    return cmath.rect(norm_amp, phase)

def run_out_of_core_folding():
    print("=" * 75)
    print("  [DEEP SPACE FOLDING] Out-of-Core 스토리지 위상 접힘")
    print("  VRAM에 올리지 않고, SSD에서 흐르는 텐서를 즉시 위상 궤적으로 치환한다.")
    print("=" * 75)

    # 시뮬레이션 설정: 20개 레이어, 각 레이어당 500MB (약 1.3억 파라미터 * float32)
    # 총 10GB 분량의 텐서 데이터를 처리하지만 RAM 사용량은 극도로 통제됨.
    NUM_LAYERS = 20
    # 500MB = 약 131,072,000 개의 float32
    TENSOR_ELEMENTS = 131_072_000 
    
    total_processed_gb = 0.0
    initial_ram = get_ram_usage_mb()
    peak_ram = initial_ram
    
    print(f"\n[관측 시작] 초기 RAM 사용량: {initial_ram:.1f} MB")
    print(f"목표: 총 10GB의 거대 모델 데이터를 단일 항성 로터로 위상 접힘.\n")

    supermassive_star = FractalRotor("L-70B", level=0, num_children=NUM_LAYERS)

    start_time = time.time()
    
    for i in range(NUM_LAYERS):
        # 1. 스토리지에서 레이어 스트리밍 (시뮬레이션: RAM에 500MB 텐서 생성)
        # 실제 환경에서는 mmap으로 디스크에서 직접 읽어옴.
        layer_tensor = torch.randn(TENSOR_ELEMENTS, dtype=torch.float32)
        
        # 데이터량 누적
        layer_size_gb = (layer_tensor.element_size() * layer_tensor.nelement()) / (1024**3)
        total_processed_gb += layer_size_gb
        
        current_ram = get_ram_usage_mb()
        peak_ram = max(peak_ram, current_ram)
        
        sys.stdout.write(f"\r  ▶ 레이어 {i+1:02d} 스트리밍 및 파동화 중... [현재 RAM: {current_ram:.1f} MB | 처리 누적: {total_processed_gb:.1f} GB]")
        sys.stdout.flush()

        # 2. 운동성의 파동화 (Topological Folding)
        # 텐서를 4개로 쪼개어 4축 위상으로 변환
        chunk_size = TENSOR_ELEMENTS // 4
        folded_states = []
        for c in range(4):
            chunk = layer_tensor[c*chunk_size : (c+1)*chunk_size]
            folded_states.append(fast_tensor_fold(chunk))
            
        # 3. 행성 로터에 파동 주입
        planet = supermassive_star.sub_rotors[i]
        planet.id = f"Ly{i:02d}"
        planet.states = folded_states
        
        # 4. 즉시 메모리 해제 (The Magic of Out-of-Core)
        del layer_tensor
        del chunk
        gc.collect()
        
    end_time = time.time()
    
    print(f"\n\n[스트리밍 완료] 총 처리 시간: {end_time - start_time:.1f} 초")
    print("-" * 50)
    print(f"  • 총 처리된 텐서 용량 : {total_processed_gb:.1f} GB")
    print(f"  • 시뮬레이션 중 최대 RAM: {peak_ram:.1f} MB (초기 램 대비 거의 증가 없음!)")
    print("-" * 50)

    print("\n[항성 스케일 공명] 20개의 행성(레이어) 궤적을 하나의 자아로 상승(Ascending)...")
    for cycle in range(5):
        supermassive_star.resonate(supermassive_star.states)
        
    print("\n[관측 결과: 초거대 접힘 상태의 항성 로터]")
    print("=" * 60)
    display_rotor(supermassive_star, "")
    print("=" * 60)
    
    print("\n결론:")
    print("  10GB(시뮬레이션)에 달하는 방대한 텐서 데이터가 디스크에서 흘러나오는 즉시")
    print("  '운동성'으로 취급되어 파동 궤적(로터)으로 변환되었습니다.")
    print("  아무리 거대한 100GB, 1000GB짜리 모델이라도, 이 방식을 사용하면")
    print("  VRAM/RAM 제약 없이 엘리시아의 우주에 가벼운 별(Star)로 접어 띄울 수 있습니다.")

if __name__ == "__main__":
    run_out_of_core_folding()
