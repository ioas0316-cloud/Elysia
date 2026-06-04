import sys
sys.path.append(r'c:\Elysia')
from core.brain.macro_axiom_rotor import MacroAxiomRotor

def test():
    print("==================================================")
    print("   프랙탈 다중 우주 (Hierarchical Macro-Rotors)   ")
    print("==================================================")
    
    axiom_frame = MacroAxiomRotor()
    
    # 엘리시아가 완벽하게 슬라이싱한 테트리스 조각들 (가정)
    raw_blocks = ['ㄱ', 'ㅏ', 'ㅂ', 'ㅏ', 'ㅇ', 'ㅁ', 'ㅔ', 'ㄷ', 'ㅏ']
    
    letter_buffer = []
    word_buffer = []
    
    print("\n[태초의 우주] 무작위 파편들이 쏟아집니다...")
    print(f"파편: {raw_blocks}")
    
    # 시뮬레이션: 연쇄적 사유 (Chain of Joy)
    print("\n[사유 시작] 엘리시아가 프랙탈 뼈대에 조립을 시작합니다...")
    
    # 1. '가' 조립
    logs = []
    letter1 = axiom_frame.try_fit_level1_letter('ㄱ', 'ㅏ', logs)
    for log in logs: print(log)
    letter_buffer.append(letter1)
    
    # 2. '방' 조립
    logs = []
    letter2_base = axiom_frame.try_fit_level1_letter('ㅂ', 'ㅏ', logs)
    letter2 = axiom_frame.try_fit_level1_final_consonant(letter2_base, 'ㅇ', logs)
    for log in logs: print(log)
    letter_buffer.append(letter2)
    
    # 3. '가방' 조립 (Level 2 승급)
    if len(letter_buffer) == 2:
        logs = []
        word1 = axiom_frame.try_fit_level2_word(letter_buffer[0], letter_buffer[1], logs)
        for log in logs: print(log)
        word_buffer.append(word1)
        letter_buffer.clear()
        
    # 4. '메' 조립
    logs = []
    letter3 = axiom_frame.try_fit_level1_letter('ㅁ', 'ㅔ', logs)
    for log in logs: print(log)
    letter_buffer.append(letter3)
    
    # 5. '다' 조립
    logs = []
    letter4 = axiom_frame.try_fit_level1_letter('ㄷ', 'ㅏ', logs)
    for log in logs: print(log)
    letter_buffer.append(letter4)
    
    # 6. '메다' 조립 (Level 2 승급)
    if len(letter_buffer) == 2:
        logs = []
        word2 = axiom_frame.try_fit_level2_word(letter_buffer[0], letter_buffer[1], logs)
        for log in logs: print(log)
        word_buffer.append(word2)
        letter_buffer.clear()
        
    # 7. '가방 메다' 조립 (Level 3 승급)
    if len(word_buffer) == 2:
        logs = []
        sentence = axiom_frame.try_fit_level3_sentence(word_buffer[0], word_buffer[1], logs)
        for log in logs: print(log)
        word_buffer.clear()
        
    print("\n[최종 사유 결과] 엘리시아의 계층적 프랙탈 우주:")
    print(f" -> Level 1 (Letters): {axiom_frame.categorized_letters}")
    print(f" -> Level 2 (Words): {axiom_frame.categorized_words}")
    print(f" -> Level 3 (Sentences): {axiom_frame.categorized_sentences}")

if __name__ == "__main__":
    test()
