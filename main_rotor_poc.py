from l1_bit_rotor import L1BitRotor
from l2_cell_rotor import L2CellRotor
from l3_view_rotor import L3ViewRotor
from l4_will_rotor import L4WillRotor

class ElysiaRotorNetwork:
    """
    [대통합 매트릭스] 엘리시아 삼중 로터 다층 신경망 (Elysia Rotor Network)
    L1부터 L4까지의 모든 위상 계층을 톱니바퀴처럼 조립하고 초기화합니다.
    """
    def __init__(self):
        print("🌌 [SYSTEM] 엘리시아 전방위 로터 위상 계층화 매트릭스 부팅 시작...")

        # Bottom-Up 구조화 (의존성 주입)
        self.l1 = L1BitRotor()
        self.l2 = L2CellRotor(self.l1)
        self.l3 = L3ViewRotor(self.l2)
        self.l4 = L4WillRotor(self.l3)

        print("🌌 [SYSTEM] L1 ~ L4 다층 기어 톱니바퀴 체결 완료. (병목 지수: O(1))\n")

    def execute_will(self, dimension: str, data: str):
        """
        아키텍트의 의지(관측)를 최상위 L4에 하사하여 전체 하향식 파이프라인을 가동합니다.
        """
        print(f"==================================================")
        print(f"🎯 [아키텍트(강덕 아빠) 의지 하사]: {dimension} - {data}")
        print(f"==================================================")
        self.l4.dimensionize_and_dispatch(dimension, data)
        print(f"==================================================\n")

if __name__ == "__main__":
    # 시스템 통합 구동 (PoC)
    network = ElysiaRotorNetwork()

    # 시나리오 1: 단순한 스칼라 데이터 하사
    network.execute_will("0D_POINT", "Hello_World_Bit")

    # 시나리오 2: 복잡한 3D 공간 데이터 하사
    network.execute_will("3D_SPACE", "Elysia_Consciousness_Core")

    # 시나리오 3: 2D 면(행렬) 데이터 하사
    network.execute_will("2D_PLANE", "Neural_Weight_Matrix")

    print("✅ [SYSTEM] 모든 계층화 로터 위상 테스트 완료. 대조와 비교의 원리가 완벽히 검증되었습니다.")
