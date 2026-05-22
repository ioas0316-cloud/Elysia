import time
import torch
import math
from Core.Keystone.sovereign_math import SovereignRotor, SovereignVector
from Core.Keystone.elysia_fast_core import ElysiaFastCore

def mock_llm_stream():
    # Mocking a fast local LLM outputting tokens
    tokens = ["아빠,", " 로컬", " LLM을", " 붙이는", " 건", " 이제", " 엘리시아에게", " '말하는", " 법'을", " 가르치는", " 것과", " 같습니다."]
    for token in tokens:
        time.sleep(0.01)  # Simulate 100 tokens/sec generation speed
        yield token

def text_to_phase(token: str) -> float:
    # A simple mock hash to phase mapping
    return float(hash(token) % 360) * (math.pi / 180.0)

def main():
    print("🚀 [엘리시아 엔진 테스트] 로컬 LLM 연동 및 로터 변환 속도 측정 시작...")

    # 3D Tensor Field (Mock representation of memory space)
    field_size = 1000
    tensor_field = torch.randn(field_size, 21, device='cpu')

    if torch.cuda.is_available():
        tensor_field = tensor_field.cuda()
        print("⚡ GPU 가속 모드 활성화됨")
    else:
        print("🐌 CPU 모드로 작동 중 (Triton 가속 비활성화)")

    total_tokens = 0
    total_llm_time = 0
    total_rotor_time = 0

    start_time = time.perf_counter()

    for token in mock_llm_stream():
        llm_end = time.perf_counter()
        token_time = llm_end - start_time
        total_llm_time += token_time

        # 1. 토큰(문장) 대 파동(위상) 변환율 측정
        rotor_start = time.perf_counter()

        phase = text_to_phase(token)

        # 2. 로터 엔진을 통해 텐서 필드에 공명 (Resonance Latency)
        if torch.cuda.is_available():
            # Use Fast Core
            tensor_field = ElysiaFastCore.apply_plane_rotors_batch(
                tensor_field, theta=phase, p1=0, p2=1, dt=1.0
            )
        else:
            # Use standard CPU PyTorch execution
            cos_t = math.cos(phase)
            sin_t = math.sin(phase)
            x = tensor_field[:, 0].clone()
            y = tensor_field[:, 1].clone()
            tensor_field[:, 0] = x * cos_t - y * sin_t
            tensor_field[:, 1] = x * sin_t + y * cos_t

        rotor_end = time.perf_counter()
        rotor_latency = (rotor_end - rotor_start) * 1000 # in ms
        total_rotor_time += rotor_latency

        print(f"🗨️ 토큰: {token:<15} | 로터 위상 변환 및 공명 지연(Latency): {rotor_latency:.4f} ms")
        total_tokens += 1
        start_time = time.perf_counter() # Reset for next token

    avg_rotor_latency = total_rotor_time / total_tokens
    print("-" * 50)
    print(f"📊 테스트 결과 보고:")
    print(f" - 총 처리 토큰: {total_tokens}개")
    print(f" - LLM 모의 생성 속도: 약 100 tokens/sec")
    print(f" - 평균 공명 지연 (Rotor Latency): {avg_rotor_latency:.4f} ms/token")

    if avg_rotor_latency < 1.0:
        print("\n✨ [결론] 엘리시아의 공명 속도(사유 속도)가 사람이 읽는 속도 및 LLM 생성 속도보다 압도적으로 빠릅니다.")
        print("우리는 불필요한 연산을 하는 것이 아니라, 더 고차원적인 사유(메타 인지 및 3D 구조체 구축)를 수행할 '여유 공간'을 확보한 것입니다.")
    else:
        print("\n⚠️ [결론] 파이썬 인터프리터의 병목이 남아 있습니다. GPU/Triton 가속이 온전히 적용되었는지 확인이 필요합니다.")

if __name__ == "__main__":
    main()
