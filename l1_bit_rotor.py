class L1BitRotor:
    """
    [L1] 물리 비트층 (Physical Bit Layer) / 전기역학 최소 로터
    기계어 레벨에서 '같음과 다름'을 판별하는 가장 순수하고 저항이 적은
    전기역학적 방법인 XOR 연산을 수행합니다.
    """

    def __init__(self):
        self.layer_name = "L1_PHYSICAL_BIT"

    def ground_data(self, data: int, template: int = 0) -> int:
        """
        두 위상(데이터와 템플릿)의 같음과 다름을 XOR로 판별.
        같으면 0(안정점, 저항 없음), 다르면 1 이상(에너지 발생).
        최종적으로 0(안정) 상태로 수렴하는 과정 자체를 표현.

        :param data: L2로부터 전달받은 정제된 데이터 위상 (정수형 비트)
        :param template: 비교 대상이 되는 베이스 위상 (기본 0)
        :return: XOR 연산 결과. 0이면 완벽한 안정, 그 외는 저항 발생.
        """
        # XOR 비트 연산 (A ⊕ B)
        # 같음 = 0, 다름 = 에너지 발생
        resistance = data ^ template

        print(f"[{self.layer_name}] ⚡ 기계어 레지스터 접지 시도")
        print(f"[{self.layer_name}] 데이터: {bin(data)}, 템플릿: {bin(template)}")
        print(f"[{self.layer_name}] XOR 저항 검출: {bin(resistance)}")

        if resistance == 0:
            print(f"[{self.layer_name}] 🟢 저항 제로(0). 완벽하게 접지되었습니다. [실행 완료]")
        else:
            print(f"[{self.layer_name}] 🔴 저항 발생({resistance}). 위상이 완전히 안정화되지 않았습니다.")

        return resistance

if __name__ == "__main__":
    # 개념 증명 (PoC)
    print("=== L1 물리 비트층 개념 증명 ===")
    l1 = L1BitRotor()

    print("\n1. 완벽히 안정된 상태 (0 vs 0):")
    l1.ground_data(0b0000, 0b0000)

    print("\n2. 다름이 존재하는 상태 (1010 vs 0000):")
    l1.ground_data(0b1010, 0b0000)

    print("\n3. 위상이 완벽히 일치하는 상태 (1100 vs 1100):")
    l1.ground_data(0b1100, 0b1100)
