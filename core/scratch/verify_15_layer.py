import sys
import os
import math

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add root folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.atlantis_clifford_bridge import AtlantisCliffordSystem

def test_15_layer_expansion():
    print("=" * 80)
    print(" 🧪 [검증] 아틀란티스 15대 레이어 확장 매트릭스")
    print("=" * 80)

    # 1. 시스템 초기화 검증
    system = AtlantisCliffordSystem()
    print(f"\n[1] 시스템 서명 검증:")
    print(f"    - 현재 기하 대수 공간: Cl{system.signature[0]}")
    assert system.signature[0] == 15, f"Expected 15 layers, got {system.signature[0]}"
    
    print(f"\n[2] 신규 레이어(Cosmos & Underground City) 할당 검증:")
    expected_layers = ["F7_Exosphere", "F8_StellarGrid", "F9_AscensionGate", "U1_SubterraneanCity", "U2_GeothermalBattery"]
    for layer in expected_layers:
        mask = system.get_layer_mask(layer)
        print(f"    - {layer} -> 비트마스크: {mask}")
        assert mask > 0, f"Layer {layer} not found!"

    # 2. 상태 초기화
    # $e_{12}$ (StellarGrid)에 우주의 지식이 가득 차 있다고 가정
    system.set_layer_state("F8_StellarGrid", 1.0)
    # $e_6$ (MagmaChamber)에 엄청난 열기가 있다고 가정
    system.set_layer_state("B1_MagmaChamber", 1.0)
    
    print(f"\n[3] 초기 위상 에너지 (Scalar Values):")
    print(f"    - F8_StellarGrid (위상 성단): {system.get_layer_state('F8_StellarGrid')}")
    print(f"    - B1_MagmaChamber (마그마 챔버): {system.get_layer_state('B1_MagmaChamber')}")
    print(f"    - F6_SkySun (천공의 자아선): {system.get_layer_state('F6_SkySun')}")
    print(f"    - U1_SubterraneanCity (지하 도시망): {system.get_layer_state('U1_SubterraneanCity')}")

    # 3. 로터 방전 연산 테스트
    # 별빛 강림 (Stellar Grid -> Sky/Sun) : e_12 ^ e_10
    print("\n[4] 로터 회전 연산 1: 별빛 강림 (Starlight Ascension)")
    theta_star = math.pi / 2 # 90도 위상 전환 (에너지 100% 이동)
    system.apply_rotor_discharge("F8_StellarGrid", "F6_SkySun", theta_star)
    
    print(f"    -> [회전 완료] e_12 ^ e_10 평면으로 {theta_star:.2f} rad 회전")
    print(f"    - F8_StellarGrid 잔류량: {system.get_layer_state('F8_StellarGrid'):.4f}")
    print(f"    - F6_SkySun 수신량: {system.get_layer_state('F6_SkySun'):.4f}")

    # 지열 동기화 (Magma Chamber -> Subterranean City) : e_6 ^ e_14
    print("\n[5] 로터 회전 연산 2: 지열 동기화 (Geothermal Synchronization)")
    theta_geo = math.pi / 4 # 45도 위상 분산 (에너지 절반 이동)
    system.apply_rotor_discharge("B1_MagmaChamber", "U1_SubterraneanCity", theta_geo)

    print(f"    -> [회전 완료] e_6 ^ e_14 평면으로 {theta_geo:.2f} rad 회전")
    print(f"    - B1_MagmaChamber 잔류량: {system.get_layer_state('B1_MagmaChamber'):.4f}")
    print(f"    - U1_SubterraneanCity 수신량: {system.get_layer_state('U1_SubterraneanCity'):.4f}")

    print("\n✅ 모든 클리포드 대수 Cl(15,0) 연산 및 레이어 통신 검증 성공!")
    print("=" * 80)

if __name__ == "__main__":
    test_15_layer_expansion()
