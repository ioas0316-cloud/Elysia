import sys
sys.path.append(r'c:\Elysia')
from core.utils.math_utils import Quaternion
from core.brain.wave_slicer import WaveSlicer
from core.brain.macro_axiom_rotor import MacroAxiomRotor
import math

def generate_wave(char: str) -> Quaternion:
    """글자의 유니코드 값을 바탕으로 연속적인 위상 궤적을 흉내냅니다."""
    code = ord(char) * 0.1  # 위상각의 차이를 명확히 벌려줌
    # 연속성을 위해 약간의 노이즈 추가
    return Quaternion(math.cos(code), math.sin(code), 0, 0).normalize()

def test():
    print("--- [시뮬레이션: 연속 파동 슬라이싱 및 원리적 놀이 (Tetris Puzzle Play)] ---")
    
    slicer = WaveSlicer()
    axiom_frame = MacroAxiomRotor("한글(Hangul)")
    
    print("\n1. 거대한 공리 뼈대(블랙홀) '한글(Hangul)'을 우주에 띄웁니다.")
    print("   이 뼈대는 [초성 위상] + [중성 위상] 결합일 때만 텐션이 0이 됩니다.")
    
    # 순차적으로 흘러가는 연속 파장
    continuous_stream = ['ㄱ', 'ㅏ', 'ㅇ', 'ㅇ', 'ㅏ', 'ㅈ', 'ㅣ']
    print(f"\n2. 연속된 파장이 흘러들어옵니다: {' -> '.join(continuous_stream)}")
    
    logs = []
    sliced_blocks = []
    
    # 엘리시아가 파장을 관측하며 스스로 단절점(변곡점)을 찾음
    for char in continuous_stream:
        wave = generate_wave(char)
        blocks = slicer.stream_wave(wave, char, logs)
        if blocks:
            sliced_blocks.extend(blocks)
            
    # 스트림이 끝나고 남은 버퍼도 잘라냄
    final_block = slicer.flush_buffer()
    if final_block:
        sliced_blocks.append(final_block)
        
    for log in logs:
        print(log)
        
    print(f"\n=> 엘리시아가 스스로 잘라낸 테트리스 조각들: {sliced_blocks}")
    
    print("\n3. 조각 맞춤 놀이 (Principled Play) 시작...")
    # 엘리시아가 무작위로 두 조각씩 집어서 뼈대에 맞춰봄
    play_logs = []
    
    # 의도적 오답 시도 (ㅇㅇ 블록과 ㅏ 블록의 조합)
    if len(sliced_blocks) >= 3:
        axiom_frame.try_fit_puzzle(sliced_blocks[1], sliced_blocks[2], play_logs)
    
    # 테트리스 블록 내부 검증 (잘라진 조각이 프레임에 맞는지)
    for block in sliced_blocks:
        if len(block) >= 2:
            axiom_frame.try_fit_puzzle(block[0], block[1], play_logs)
            
    for log in play_logs:
        print(log)
        
    print(f"\n4. [최종 사유 결과]")
    print(f"엘리시아가 원리적 놀이를 통해 스스로 맵핑해 낸 '{axiom_frame.axiom_name}' 범주 내 데이터:")
    print(f" -> {axiom_frame.categorized_blocks}")

if __name__ == "__main__":
    test()
