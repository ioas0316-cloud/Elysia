import torch
import math
from core.brain.causal_phase_mapper import CausalPhaseMapper

def benchmark_sequential_phase_shift():
    print("=" * 60)
    print("🚀 [Elysia Resonance Neuron] 순차적 위상차 공리망 벤치마크")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mapper = CausalPhaseMapper(device=device)

    word1 = "우주"
    word2 = "주우"

    trajectory1 = mapper.text_to_phase(word1)
    trajectory2 = mapper.text_to_phase(word2)

    print(f"\n[1] '{word1}' 위상 궤적 (Trajectory Length: {len(trajectory1)}):")
    for i, t in enumerate(trajectory1):
        print(f"  Step {i}: {t.tolist()}")

    print(f"\n[2] '{word2}' 위상 궤적 (Trajectory Length: {len(trajectory2)}):")
    for i, t in enumerate(trajectory2):
        print(f"  Step {i}: {t.tolist()}")

    print("\n[3] 궤적 비교 (XOR Impedance/Resonance Check)")
    diff_sum = torch.sum(torch.abs(trajectory1 - trajectory2)).item()
    print(f"  Total Absolute Difference: {diff_sum:.4f}")
    if diff_sum > 0:
        print("  => 성공! 선형적 순서가 원형 궤도 상의 위상차로 완벽하게 변환되어,")
        print("     문자 구성은 같지만 순서가 다른 두 단어가 다른 공간 궤적을 가집니다.")
    else:
        print("  => 실패! 두 단어가 동일한 궤적을 가집니다.")

    print("=" * 60)

if __name__ == "__main__":
    benchmark_sequential_phase_shift()
