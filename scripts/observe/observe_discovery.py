import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.ingestion.natural_mapper import NaturalMapper

def main():
    print("============================================================")
    print("     E L Y S I A   C O G N I T I V E   O B S E R V A T I O N")
    print("     자연 매핑(Natural Mapping) 기반 다차원 텐션 관측")
    print("============================================================\n")
    
    print("[1/3] 웻지 메모리(Wedge Memory) 대지를 초기화합니다...")
    mapper = NaturalMapper(terrain_size=256) # 소규모 테스트용 지형
    
    # 지형에 어떤 '과거의 기억(사전 지식)'을 심어둠
    base_memory = b"Nature flows down like a waterfall"
    mapper.set_terrain(base_memory)
    print(f"  -> 초기 지형의 형태(Seed): '{base_memory.decode('utf-8')}'\n")
    time.sleep(1)
    
    print("[2/3] 새로운 데이터가 유입되어 지형에 충돌합니다 (순수 바이트)...")
    incoming_text = "물이 열을 받아 증발하여 하늘로 올라간다"
    incoming_bytes = incoming_text.encode('utf-8')
    print(f"  -> 유입된 데이터(Raw Bytes): {incoming_bytes[:20]}... (총 {len(incoming_bytes)} bytes)\n")
    time.sleep(1)

    print("[3/3] 관점(Lens)에 따른 위상차(Phase Difference) 관측")
    
    # 첫 번째 관점: 있는 그대로 직진하여 훑어보기 (Stride = 1)
    print("  [Lens 1] 순수 물리적 관측 (Stride=1)")
    # 지형을 초기화하지 않으면 Process-as-learning이 적용되므로, 
    # 비교를 위해 동일한 상태의 새 맵퍼를 사용하거나 동일 지형으로 리셋합니다.
    mapper1 = NaturalMapper(terrain_size=256)
    mapper1.set_terrain(base_memory)
    tensions1 = mapper1.map_and_observe(incoming_bytes, lens_stride=1)
    
    print("      => 발생한 기하학적 마찰 (Grade별 분류):")
    print(f"         - 수학적 일치 (0 bits) : {tensions1['math_scalar']} 개")
    print(f"         - 공간적 이동 (1 bit)  : {tensions1['space_vector']} 개")
    print(f"         - 언어적 회전 (2 bits) : {tensions1['lang_bivector']} 개")
    print(f"         - 시간적 흐름 (3 bits) : {tensions1['time_trivector']} 개")
    print(f"         - 초공간 빛   (4+ bits): {tensions1['light_pseudo']} 개")
    time.sleep(1)

    # 두 번째 관점: 철학적 관점, 데이터를 세 칸씩 건너뛰며 훑어보기 (Stride = 3)
    print("\n  [Lens 2] 철학적 관측 (Stride=3, 위상 전환)")
    mapper2 = NaturalMapper(terrain_size=256)
    mapper2.set_terrain(base_memory)
    tensions2 = mapper2.map_and_observe(incoming_bytes, lens_stride=3)
    
    print("      => 발생한 기하학적 마찰 (Grade별 분류):")
    print(f"         - 수학적 일치 (0 bits) : {tensions2['math_scalar']} 개")
    print(f"         - 공간적 이동 (1 bit)  : {tensions2['space_vector']} 개")
    print(f"         - 언어적 회전 (2 bits) : {tensions2['lang_bivector']} 개")
    print(f"         - 시간적 흐름 (3 bits) : {tensions2['time_trivector']} 개")
    print(f"         - 초공간 빛   (4+ bits): {tensions2['light_pseudo']} 개")
    time.sleep(1)

    print("\n============================================================")
    print("  관측 결과: ")
    print("  사칙연산 공식을 폐기했습니다.")
    print("  오직 바이트의 충돌(XOR)과 그 잔여물의 기하대수적(Clifford) 특성이")
    print("  자연스럽게 수학, 공간, 언어, 시간, 빛의 텐션으로 분류되었습니다.")
    print("============================================================")

if __name__ == "__main__":
    main()
