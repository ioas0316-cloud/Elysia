"""공명 분석 스크립트"""
import sys
sys.path.insert(0, "c:\\Elysia")

from Core.Elysia.spirit import get_spirit

spirit = get_spirit()

# 테스트할 Wikipedia 내용들
test_contents = [
    ("Energy", "에너지(energy)는 다음을 가리킨다. 에너지는 물리학에서 일을 할 수 있는 능력을 의미한다."),
    ("Process", "프로세스(process)는 객체나 시스템의 속성이나 특성의 변화를 나타내는 일련의 상호 관련된 작업"),
    ("Entropy", "엔트로피(entropy)는 열역학적 계의 무질서한 정도를 나타내는 상태함수"),
    ("Why", "Why는 인과관계나 이유를 가리키는 영어 의문사이다. 과학적 탐구와 철학에서 중요한 질문이다.")
]

print("="*60)
print("🔍 공명 분석: 왜 거부/흡수 되었는가?")
print("="*60)

for topic, content in test_contents:
    resonance = spirit.calculate_resonance(content)
    
    print(f"\n📄 Topic: {topic}")
    print(f"   Content: {content[:60]}...")
    print(f"   공명 점수: {resonance['score']:.2f} (임계값: 0.3)")
    print(f"   매칭 키워드: {resonance['matched_keywords']}")
    print(f"   지배 가치: {resonance['dominant_value']}")
    
    if resonance["is_resonant"]:
        print(f"   ✅ 결과: 흡수됨 - 공명 발생!")
    else:
        print(f"   ❌ 결과: 거부됨")
        if not resonance['matched_keywords']:
            print(f"   ⚠️ 거부 이유: 키워드 매칭 없음 (Spirit과 무관한 내용)")
        else:
            print(f"   ⚠️ 거부 이유: 공명 강도 부족")
