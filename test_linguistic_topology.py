import sys
sys.path.append(r'c:\Elysia')
from core.brain.linguistic_rotorizer import LinguisticRotorizer

def test():
    print("--- [시뮬레이션: 언어의 기하학적 매핑 (Linguistic Topology)] ---")
    
    rotorizer = LinguisticRotorizer()
    
    word = "가방"
    print(f"\n1. 아카이브에서 단어 '{word}' 수집 시작...")
    
    # "가방" 이라는 단어가 아카이브에서 100번 발견되었다고 가정 (반복 주입)
    for i in range(1, 101):
        coord, logs = rotorizer.process_word(word)
        
        if i == 1 or i == 100:
            print(f"\n[입력 {i}회차]")
            print(f"   -> 4D 교차점 생성: Q({coord.w:.4f}, {coord.x:.4f}, {coord.y:.4f}, {coord.z:.4f})")
            
            # 동기화 로그(기쁨)가 있다면 출력
            for log in logs:
                if "Joy" in log:
                    print(log)
                    
    # 결과 확인
    well = rotorizer.word_gravity_wells[word]
    final_coord = well['coord']
    tau = well['tau']
    
    print(f"\n✨ [최종 토폴로지 사전 구축 완료]")
    print(f"단어 '{word}'는 더 이상 텍스트가 아닙니다.")
    print(f"4D 좌표 Q({final_coord.w:.4f}, {final_coord.x:.4f}, {final_coord.y:.4f}, {final_coord.z:.4f})에")
    print(f"질량(Tau) {tau:.1f}을 가진 거대한 중력계(은하)로 우주에 새겨졌습니다.")

if __name__ == "__main__":
    test()
