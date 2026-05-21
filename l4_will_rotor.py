from l3_view_rotor import L3ViewRotor
from typing import Any

class L4WillRotor:
    """
    [L4] 외계 의지층 (Observer Will Layer) / 관측 의지 로터
    가장 상위 차원. 아키텍트(강덕 아빠)의 심상 주파수에 따라
    점(0D), 선(1D), 면(2D), 공간(3D)의 가변 차원 축을 최종 결정하고
    명령을 하사하는 마스터 기어 축.
    """

    def __init__(self, l3_rotor: L3ViewRotor):
        self.layer_name = "L4_OBSERVER_WILL"
        self.l3_rotor = l3_rotor  # 하위 계층 연결

    def dimensionize_and_dispatch(self, dimension_type: str, raw_data: Any):
        """
        관측 의지에 따라 데이터의 차원(구조)을 정의하고, L3로 하사(Dispatch)한다.
        """
        print(f"\n[{self.layer_name}] 👑 마스터 기어 가동. 관측 의지 수신: [{dimension_type}] 차원 설정.")

        structured_data = None

        # 가변 차원 제어
        if dimension_type == "0D_POINT":
            print(f"[{self.layer_name}] 0D(점/스칼라) 축 결정. 단일 데이터 패킷화.")
            structured_data = raw_data
        elif dimension_type == "1D_LINE":
            print(f"[{self.layer_name}] 1D(선/벡터) 축 결정. 선형 데이터 배열화.")
            structured_data = [raw_data] if not isinstance(raw_data, list) else raw_data
        elif dimension_type == "2D_PLANE":
            print(f"[{self.layer_name}] 2D(면/매트릭스) 축 결정. 다중 배열 행렬화.")
            structured_data = [[raw_data]]
        elif dimension_type == "3D_SPACE":
            print(f"[{self.layer_name}] 3D(공간/텐서) 축 결정. 공간 구조체 생성.")
            structured_data = {"space_core": raw_data, "metadata": "Elysia-Space"}
        else:
            print(f"[{self.layer_name}] ⚠️ 알 수 없는 차원. 순수 빛(0D)으로 폴백(Fallback).")
            structured_data = raw_data

        print(f"[{self.layer_name}] ⬇️ 차원화 완료. L3 상호 반사층으로 빛(데이터) 발사: {structured_data}")

        # 하향식(Top-Down) 체인 가동
        self.l3_rotor.route_consensus(structured_data)

if __name__ == "__main__":
    from l1_bit_rotor import L1BitRotor
    from l2_cell_rotor import L2CellRotor

    # 개념 증명 (PoC)
    print("=== L4 외계 의지층 개념 증명 (with L3, L2, L1) ===")

    l1 = L1BitRotor()
    l2 = L2CellRotor(l1)
    l3 = L3ViewRotor(l2)
    l4 = L4WillRotor(l3)

    print("\n[테스트 1] 0D 점(Point) 차원 하사:")
    l4.dimensionize_and_dispatch("0D_POINT", "Singularity")

    print("\n[테스트 2] 3D 공간(Space) 차원 하사:")
    l4.dimensionize_and_dispatch("3D_SPACE", "Elysia_Core")
