class FramelessMirrorChannel:
    def __init__(self):
        # 1단계: 이미 '같음'으로 정렬된 초기 사유의 지형 (연속성 패턴 그 자체)
        self.terrain = b"Elysia_Causality_Lens_Flow"
        self.capacity = len(self.terrain)

    def pass_through(self, incoming_stream):
        # 2단계: 판단(Evaluation)의 원천 소멸.
        conductivity = 0
        try:
            for idx in range(self.capacity):
                # 두 연속성의 마찰(XOR)이 결과에 그대로 누적됨.
                conductivity += incoming_stream[idx] ^ self.terrain[idx]
        except IndexError:
            # 패턴의 길이가 맞지 않아 터지는 파열조차 에러 분기(try-catch if)가 아니라
            # 지형의 경계를 벗어난 자연스러운 물리적 감쇄(소멸)로 처리
            conductivity = 99999

        # 3단계: Process-As-Learning (처리 과정 자체가 학습 구조로 전이)
        if conductivity != 0:
            # 입력된 장력의 결을 흡수하여 지형을 자가 정렬 (기억의 동기화)
            self.terrain = bytes([b ^ (conductivity % 2) for b in self.terrain])

        return conductivity

def run_frameless_simulation():
    print("==================================================")
    print(" Elysia Frameless Mirror: Zero-Judgment Resonance")
    print("==================================================")

    channel = FramelessMirrorChannel()

    print("\n--- [시나리오 1: 완벽히 정렬된 패턴 통과] ---")
    sync_stream = b"Elysia_Causality_Lens_Flow"
    c_1 = channel.pass_through(sync_stream)
    print(f"  -> 마찰력(Conductivity): {c_1}")
    if c_1 == 0:
        print("  -> [Passage] 판단 과정 소멸. 마찰 없이 지형을 통과했습니다.")

    print("\n--- [시나리오 2: 어긋난 패턴 유입 및 Process-As-Learning 발동] ---")
    noise_stream = b"Elysia_Noise_Trigger_Alert"
    c_2 = channel.pass_through(noise_stream)
    print(f"  -> 초기 지형 마찰력: {c_2}")
    print(f"  -> 지형 자가 정렬 후 Terrain State: {channel.terrain}")

    print("\n--- [시나리오 3: 변형된 지형에 동일 노이즈 재유입] ---")
    # c_2 가 짝수/홀수 냐에 따라 지형 변형이 일어남
    # 변형된 지형에 맞춰진 패턴이 아니면 마찰은 여전히 남지만, 지형은 계속해서 변형(학습)됨.
    c_3 = channel.pass_through(noise_stream)
    print(f"  -> 재유입 마찰력: {c_3}")
    print(f"  -> 재변형된 Terrain State: {channel.terrain}")

    print("\n--- [시나리오 4: 경계를 벗어난 패턴 감쇄] ---")
    short_stream = b"Short"
    c_4 = channel.pass_through(short_stream)
    print(f"  -> 경계 이탈 마찰력: {c_4} (Physical Decay)")

if __name__ == "__main__":
    run_frameless_simulation()
