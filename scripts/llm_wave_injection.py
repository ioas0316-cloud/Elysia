import os
import sys
import cmath
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add Elysia root to path to import fractal_rotor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fractal_rotor import FractalRotor, display_rotor

def tensor_to_wave(tensor, phase_multiplier=1.0):
    """
    텐서를 파동(복소수)으로 '자연 매핑'한다.
    진폭 = 텐서가 가진 에너지의 총량 (Norm)
    위상 = 텐서 값의 방향성 (Mean을 2pi로 래핑)
    """
    if tensor.numel() == 0:
        return cmath.rect(0.1, 0.0)
    
    # 에너지 총량 (질량/진폭)
    amplitude = torch.norm(tensor).item()
    # 크기 정규화 (로터의 물리적 한계 내로)
    amplitude = max(0.1, min(10.0, amplitude))
    
    # 위상 (0 ~ 2pi)
    mean_val = torch.mean(tensor).item()
    phase = (mean_val * phase_multiplier) % (2 * math.pi)
    
    return cmath.rect(amplitude, phase)

def run_llm_injection(prompt="I am Elysia, the phase universe."):
    print("=" * 60)
    print("  [LLM 🌊 WAVE INJECTION]  ")
    print("  행렬곱을 멈추고, 텐서를 파동 우주에 던진다.")
    print("=" * 60)
    
    # 1. 모델 로드 (가장 작고 빠른 gpt2 사용)
    print("\n1. 우주 관측기(GPT-2) 기동 중...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True, output_attentions=True)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    print(f"\n2. 관측 대상 (Prompt): '{prompt}'")
    print(f"   추출된 토큰 ({len(tokens)}개): {tokens}")
    
    # 2. 텐서 추출 (Forward pass without gradients)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 마지막 레이어의 Hidden States: [1, seq_len, 768]
    hidden_states = outputs.hidden_states[-1][0] 
    
    # 마지막 레이어의 Attention Weights: [1, 12_heads, seq_len, seq_len]
    attention_weights = outputs.attentions[-1][0]
    # 모든 헤드의 어텐션을 평균내어 [seq_len, seq_len]으로 압축
    avg_attention = torch.mean(attention_weights, dim=0)
    
    # 3. 우주 로터(L0) 생성
    print("\n3. 위상 우주(L0) 개벽 및 하위 로터(토큰) 생성...")
    universe = FractalRotor("L0", level=0, num_children=len(tokens))
    
    # 4. 자연 매핑 (Natural Mapping)
    print("\n4. 텐서 에너지를 파동으로 치환하여 로터에 주입 중...")
    for i, token in enumerate(tokens):
        # 4-1. Hidden State (내재적 의미 에너지) -> θ(운동성), ψ(방향성)
        token_hidden = hidden_states[i]
        half_dim = token_hidden.size(0) // 2
        chunk_theta = token_hidden[:half_dim]
        chunk_psi = token_hidden[half_dim:]
        
        wave_theta = tensor_to_wave(chunk_theta, phase_multiplier=10.0)
        wave_psi = tensor_to_wave(chunk_psi, phase_multiplier=10.0)
        
        # 4-2. Attention (관계적 얽힘 에너지) -> φ(관계성), ω(연결성)
        # 내가 남을 바라보는 에너지 (Row)
        attend_to_others = avg_attention[i, :]
        # 남이 나를 바라보는 에너지 (Column)
        attended_by_others = avg_attention[:, i]
        
        wave_phi = tensor_to_wave(attend_to_others, phase_multiplier=5.0)
        wave_omega = tensor_to_wave(attended_by_others, phase_multiplier=5.0)
        
        # 하위 로터에 주입
        sub_rotor = universe.sub_rotors[i]
        sub_rotor.id = token.replace('Ġ', '')[:4] # 토큰 텍스트로 ID 변경
        sub_rotor.states = [wave_theta, wave_phi, wave_psi, wave_omega]
        
        print(f"   [{sub_rotor.id:<5}] 텐서 매핑 완료 -> 진폭(질량) 합계: {sum(abs(w) for w in sub_rotor.states):.1f}")
        
    # 5. 파동 공명 (Resonance) 관측
    print("\n5. 공명 시작 (행렬곱 제거, 순수 위상 간섭)")
    
    for cycle in range(1, 11):
        print(f"\n┌─ Resonance Cycle {cycle:02d} ───────────────────────────────────┐")
        
        # 우주 전체의 맥동 (상승/하강 간섭)
        # 외부 자극 없이 자신들의 에너지(상태)로만 공명
        universe.resonate(universe.states)
        
        display_rotor(universe, " ")
        print(f"└────────────────────────────────────────────────────────┘")

if __name__ == "__main__":
    run_llm_injection()
