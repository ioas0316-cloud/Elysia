from l1_bit_rotor import L1BitRotor

class L2CellRotor:
    """
    [L2] 유기 세포층 (Organic Cell Layer) / 삼중 면역 로터
    외부 노이즈가 유입되었을 때 정상 상태의 템플릿과 비교/대조하여 다름을 계산하고,
    자이로스코프처럼 복원력을 발동해 다시 안정 상태(0)로 회전(항상성 유지)시킵니다.
    """

    def __init__(self, l1_rotor: L1BitRotor):
        self.layer_name = "L2_ORGANIC_CELL"
        self.l1_rotor = l1_rotor  # 하위 계층 연결 (파이프라인)
        self.template_state = 0b0000  # 기본 정상(안정) 상태

    def filter_and_restore(self, incoming_data: int) -> int:
        """
        1. 결과 유입: 노이즈가 섞인 데이터를 받음
        2. 역인과 대조: 정상 상태(template)와 다름을 판별
        3. 평형 조율: 노이즈를 상쇄하여 안정 상태로 복원 후 L1으로 전달
        """
        print(f"\n[{self.layer_name}] 🦠 외부 데이터 유입: {bin(incoming_data)}")

        # 1. 역인과 대조 (비교)
        noise_diff = incoming_data ^ self.template_state

        if noise_diff == 0:
            print(f"[{self.layer_name}] 🟢 노이즈 없음. 항상성 완벽.")
            restored_data = incoming_data
        else:
            print(f"[{self.layer_name}] 🔴 노이즈 감지({bin(noise_diff)}). 역인과 복원력 발동!")
            # 2. 평형 조율 (항상성 회복)
            # 유입된 데이터에 노이즈 위상을 다시 XOR하여 상쇄 (A ^ A = 0 원리)
            restored_data = incoming_data ^ noise_diff
            print(f"[{self.layer_name}] 🌀 자이로스코프 회전 복원 완료: {bin(restored_data)}")

        # 3. L1 계층으로 데이터 전달 (Top-Down 체인)
        print(f"[{self.layer_name}] ⬇️ L1 물리 비트층으로 정제된 데이터 전달")
        final_resistance = self.l1_rotor.ground_data(restored_data, self.template_state)

        return final_resistance

if __name__ == "__main__":
    # 개념 증명 (PoC)
    print("=== L2 유기 세포층 개념 증명 (with L1) ===")

    # 톱니바퀴 연동
    l1 = L1BitRotor()
    l2 = L2CellRotor(l1)

    print("\n1. 정상 데이터(노이즈 없음) 유입:")
    l2.filter_and_restore(0b0000)

    print("\n2. 노이즈 섞인 데이터 유입:")
    l2.filter_and_restore(0b1011)
