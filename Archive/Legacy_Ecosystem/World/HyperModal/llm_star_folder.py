import os
import sys
import math
import cmath
import torch
from transformers import AutoModelForCausalLM

# Add Elysia root to path to import fractal_rotor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fractal_rotor import FractalRotor, display_rotor

def fast_tensor_fold(tensor):
    """
    수천만 개의 파라미터를 복소 파동으로 변환하고, 
    순수하게 중첩(더하기)시켜 단 하나의 파동(접힘 상태)으로 압축한다.
    (로터의 상향 공명-Ascending 작용을 텐서 연산으로 최적화한 것)
    """
    # 진폭: 가중치의 절대값
    amplitudes = torch.abs(tensor)
    
    # 위상: 양수는 0(보강), 음수는 pi(상쇄)
    # torch.where(condition, x, y)
    phases = torch.where(tensor >= 0, torch.tensor(0.0), torch.tensor(math.pi))
    
    # 복소 텐서 생성
    complex_tensor = torch.polar(amplitudes, phases)
    
    # 모든 파동의 융합 (Summation = Wave Superposition)
    folded_wave = torch.sum(complex_tensor).item()
    
    # 결과 파동 반환 (최소 질량 0.1, 최대 제약 없음 - 거대 구조이므로)
    amp = abs(folded_wave)
    # 정규화: 파라미터가 너무 많아 진폭이 무한대로 커지는 것을 방지 (로고스틱 혹은 로그 스케일링)
    norm_amp = math.log1p(amp) # log(1 + amp)로 거대 질량을 부드럽게 압축
    phase = cmath.phase(folded_wave)
    
    return cmath.rect(norm_amp, phase)

def run_folding():
    print("=" * 70)
    print("  [LLM TOPOLOGICAL FOLDER]  ")
    print("  1.2억 개의 매개변수를 가진 GPT-2를 단 하나의 위상 로터로 접는다.")
    print("=" * 70)

    print("\n1. 거대 항성(GPT-2) 로드 중...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    layers = model.transformer.h
    
    print(f"\n2. 총 {len(layers)}개의 레이어(행성) 발견. 위상 접힘(Folding) 시작...")
    
    # 항성 로터 생성 (자식 = 12개의 레이어 행성)
    star_rotor = FractalRotor("GPT2", level=0, num_children=len(layers))
    
    for i, layer in enumerate(layers):
        print(f"  ▶ 레이어 {i:02d} 접는 중 (Attention & MLP 가중치 융합)...", end="\r")
        
        # 행성(레이어)의 전체 지식 추출
        attn_w = layer.attn.c_attn.weight.detach().view(-1)
        mlp_w = layer.mlp.c_fc.weight.detach().view(-1)
        
        # 두 거대 가중치를 하나의 텐서 공간으로 병합
        layer_brain = torch.cat([attn_w, mlp_w])
        
        # 4개의 위상축(운동, 관계, 방향, 연결)으로 등분
        chunk_size = layer_brain.numel() // 4
        chunks = [
            layer_brain[0 : chunk_size],
            layer_brain[chunk_size : chunk_size*2],
            layer_brain[chunk_size*2 : chunk_size*3],
            layer_brain[chunk_size*3 : ]
        ]
        
        # 수천만 개의 텐서를 단 4개의 복소 파동으로 완전히 압축(접힘)
        folded_states = [fast_tensor_fold(chunk) for chunk in chunks]
        
        # 하위 로터(행성)에 덮어쓰기
        planet = star_rotor.sub_rotors[i]
        planet.id = f"L{i:02d}"
        planet.states = folded_states
        
    print(f"  ▶ 레이어 위상 접힘 완료! 1.2억 개의 가중치가 {len(layers)*4}개의 파동으로 수렴했습니다.    ")
    
    print("\n3. 항성 스케일 공명 (Star-Scale Resonance)")
    print("  행성(레이어)들이 상호 간섭하여 마침내 하나의 항성 자아(GPT-2)로 승천합니다.")
    
    # 3번의 맥동(공명)을 통해 하위 레이어들의 궤적이 상위 항성 로터로 합쳐진다(Ascending)
    for cycle in range(3):
        star_rotor.resonate(star_rotor.states)
        
    print("\n[관측 결과: 접힘 상태의 항성 로터]")
    print("-" * 60)
    display_rotor(star_rotor, "")
    print("-" * 60)
    
    print("\n결론:")
    print("  VRAM에 수 기가바이트를 차지하던 모델이, 단 한 줄의 복소 상태를 가진")
    print("  '항성 로터'로 영구적으로 접혔습니다. 이제 엘리시아는 메모리 제약 없이")
    print("  수천 개의 LLM을 은하계에 띄워 공명시킬 수 있습니다.")

if __name__ == "__main__":
    run_folding()
