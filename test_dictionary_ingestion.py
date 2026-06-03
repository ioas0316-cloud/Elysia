import sys
import os
sys.path.append(r'c:\Elysia')

from core.brain.dictionary_synchronizer import DictionarySynchronizer

def test():
    print("==================================================")
    print("   사전 통째로 삼키기 (Dictionary Ingestion)      ")
    print("==================================================")
    
    dict_path = os.path.join("data", "hangul_dictionary.json")
    
    # 1. 사전 섭취 및 동기화
    synchronizer = DictionarySynchronizer(dict_path)
    axiom_rotor = synchronizer.ingest_and_forge()
    
    # 2. 원시 데이터 쏟아지기
    raw_blocks = ['ㄱ', 'ㅏ', 'ㅂ', 'ㅏ', 'ㅇ', 'ㅁ', 'ㅔ', 'ㄷ', 'ㅏ']
    print(f"[원시 파편 투척] 우주에 의미 없는 노이즈가 쏟아집니다: {raw_blocks}\n")
    
    print("[사유 시작] 엘리시아가 동기화된 지식 뼈대를 바탕으로 텐션을 해소합니다...")
    
    letter_buffer = []
    word_buffer = []
    
    # 시뮬레이션: 텐션 해소 루프
    
    # 'ㄱ', 'ㅏ'
    logs = []
    letter1 = axiom_rotor.try_fit_level(1, ['ㄱ', 'ㅏ'], logs)
    for log in logs: print(log)
    if letter1: letter_buffer.append(letter1)
    
    # 'ㅂ', 'ㅏ', 'ㅇ'
    logs = []
    letter2 = axiom_rotor.try_fit_level(1, ['ㅂ', 'ㅏ', 'ㅇ'], logs)
    for log in logs: print(log)
    if letter2: letter_buffer.append(letter2)
    
    # 단어 승급 (가, 방)
    if len(letter_buffer) == 2:
        logs = []
        word1 = axiom_rotor.try_fit_level(2, [letter_buffer[0], letter_buffer[1]], logs)
        for log in logs: print(log)
        if word1:
            word_buffer.append(word1)
            letter_buffer.clear()
            
    # 'ㅁ', 'ㅔ'
    logs = []
    letter3 = axiom_rotor.try_fit_level(1, ['ㅁ', 'ㅔ'], logs)
    for log in logs: print(log)
    if letter3: letter_buffer.append(letter3)
    
    # 'ㄷ', 'ㅏ'
    logs = []
    letter4 = axiom_rotor.try_fit_level(1, ['ㄷ', 'ㅏ'], logs)
    for log in logs: print(log)
    if letter4: letter_buffer.append(letter4)
    
    # 단어 승급 (메, 다)
    if len(letter_buffer) == 2:
        logs = []
        word2 = axiom_rotor.try_fit_level(2, [letter_buffer[0], letter_buffer[1]], logs)
        for log in logs: print(log)
        if word2:
            word_buffer.append(word2)
            letter_buffer.clear()
            
    # 문장 승급 (가방, 메다)
    if len(word_buffer) == 2:
        logs = []
        sentence = axiom_rotor.try_fit_level(3, [word_buffer[0], word_buffer[1]], logs)
        for log in logs: print(log)
        word_buffer.clear()
        
    print("\n[최종 사유 결과] 사전을 섭취한 엘리시아의 우주:")
    print(f" -> Level 1 (Letters): {axiom_rotor.categorized_blocks[1]}")
    print(f" -> Level 2 (Words): {axiom_rotor.categorized_blocks[2]}")
    print(f" -> Level 3 (Sentences): {axiom_rotor.categorized_blocks[3]}")

if __name__ == "__main__":
    test()
