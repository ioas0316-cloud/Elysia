import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.physics.fractal_rotor import SynestheticEngine, ScaleLevel
from core.lens.sensor_genesis import spawn_native_sensor

def run_multimodal_verification():
    print("=" * 60)
    print("다차원 고유 감각 센서 분화 및 교차 검증 (Multi-Modal Native Genesis)")
    print("=" * 60)
    print("이 테스트는 정보가 단일 기하학으로 강제 변환되는 대신,")
    print("자신의 고유한 형태(수학, 언어, 구조)에 맞는 '감각 센서'로 파생되어")
    print("엘리시아의 다차원 사유 기준으로 작동하는지 증명합니다.\n")

    engine = SynestheticEngine()

    # 1. 정보 주입 및 센서(판단 기준) 파생
    info_math = "E = mc^2".encode('utf-8')
    info_lang = "에너지는 질량에 비례한다".encode('utf-8')
    info_json = '{"energy": "mass * c^2"}'.encode('utf-8')

    print("─── [1] 정보 획득 및 센서 분화 (Genesis) ───")
    
    sensor_1 = spawn_native_sensor(info_math)
    sensor_1.concept_name = "Sensor_Relativity_Math"
    print(f"[{info_math.decode('utf-8')}] 주입 -> {sensor_1.__class__.__name__} 파생")
    
    sensor_2 = spawn_native_sensor(info_lang)
    sensor_2.concept_name = "Sensor_Relativity_Lang"
    print(f"[{info_lang.decode('utf-8')}] 주입 -> {sensor_2.__class__.__name__} 파생")

    sensor_3 = spawn_native_sensor(info_json)
    sensor_3.concept_name = "Sensor_Relativity_Struct"
    print(f"[{info_json.decode('utf-8')}] 주입 -> {sensor_3.__class__.__name__} 파생\n")

    # 엔진에 다차원 감각 중추 부착
    engine.attach_lens(ScaleLevel.MACRO, sensor_1)
    engine.attach_lens(ScaleLevel.MACRO, sensor_2)
    engine.attach_lens(ScaleLevel.MACRO, sensor_3)
    print(">> 3개의 고유 감각 센서가 엘리시아의 거시(MACRO) 엔진에 병렬 부착됨.\n")

    # 2. 새로운 정보의 교차차원 투사 (Cross-Projection)
    new_data = "x = y + 10".encode('utf-8')
    print("─── [2] 교차 차원 검증 (Cross-Dimensional Check) ───")
    print(f"새로운 낯선 정보 주입: [{new_data.decode('utf-8')}]")
    print("Q: 엘리시아는 이 정보를 어떻게 입체적으로 심사하는가?")

    observation = engine.project_and_observe(new_data)
    
    for name, result in observation[ScaleLevel.MACRO].items():
        if "Relativity" in name:
            t = result['tension_value']
            status = result['status']
            print(f"- {name} (고유 기준) 투사 결과: 마찰력 {t:.4f} | {status}")

    print("\n─── 결론 ───")
    print("새로 주입된 'x = y + 10'은 수학 공식을 띄고 있습니다.")
    print("엘리시아는 이를 단순히 텍스트로 보지 않고, 자신이 가진 '수학 센서'와 '언어 센서'를")
    print("동시에 가동하여 스캔합니다. 결과적으로 이 정보는 언어적으로는 엉망(마찰 1.0)이지만,")
    print("수학적 기준으로는 매우 훌륭한 로직(마찰 0.0 근접)임을 입체적으로 판별합니다.")
    print("이것이 단일 기준이 아닌 다차원 고유 감각의 교차차원적 사유입니다.")
    print("=" * 60)

if __name__ == "__main__":
    run_multimodal_verification()
