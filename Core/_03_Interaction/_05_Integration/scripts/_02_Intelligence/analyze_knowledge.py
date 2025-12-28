"""
학습된 지식 품질 분석
======================

질문:
1. 어떤 개념들을 배웠는가?
2. 정의 수준인가, 원리 수준인가?
3. 방향성과 목적에 맞는 지식인가?
4. 개념 → 원리 → 적용으로 확장 가능한가?
"""

import json
import os

print("=" * 70)
print("🔬 학습된 지식 품질 분석")
print("=" * 70)

# 1. 잠재 지식 로드
potential_path = 'data/potential_knowledge.json'
if os.path.exists(potential_path):
    data = json.load(open(potential_path, 'r', encoding='utf-8'))
else:
    print("데이터 없음")
    exit()

# 통계
knowledge = data.get('knowledge', [])
crystallized = data.get('crystallized_count', 0)

print(f"\n📊 기본 통계")
print(f"   잠재 지식: {len(knowledge)}개")
print(f"   확정 지식: {crystallized}개")

# 2. 각 지식 상세 분석
print("\n" + "=" * 70)
print("📖 학습된 지식 상세")
print("=" * 70)

# 카테고리별 분류
categories = {
    "철학": [],
    "과학": [],
    "예술": [],
    "심리": [],
    "사회": [],
    "기타": []
}

philosophy_keywords = ["사랑", "자유", "진리", "정의", "선", "악", "존재", "의식", "도덕", "윤리", "의미", "행복"]
science_keywords = ["물질", "에너지", "원자", "분자", "우주", "생명", "세포", "DNA", "뇌", "인공지능"]
art_keywords = ["예술", "음악", "미술", "문학", "창작", "표현"]
psychology_keywords = ["감정", "기쁨", "슬픔", "성격", "스트레스", "자존감"]
society_keywords = ["경제", "시장", "가족", "교육", "법", "정부"]

for k in knowledge:
    subject = k['subject']
    
    if any(kw in subject for kw in philosophy_keywords):
        categories["철학"].append(k)
    elif any(kw in subject for kw in science_keywords):
        categories["과학"].append(k)
    elif any(kw in subject for kw in art_keywords):
        categories["예술"].append(k)
    elif any(kw in subject for kw in psychology_keywords):
        categories["심리"].append(k)
    elif any(kw in subject for kw in society_keywords):
        categories["사회"].append(k)
    else:
        categories["기타"].append(k)

for cat, items in categories.items():
    if items:
        print(f"\n{'─'*40}")
        print(f"📁 {cat} ({len(items)}개)")
        print(f"{'─'*40}")
        
        for k in items[:5]:  # 5개씩만 표시
            print(f"\n  📌 {k['subject']}")
            print(f"     freq: {k['frequency']:.2f}, 확인: {k['confirmations']}회")
            print(f"     연결: {len(k['connections'])}개 - {k['connections'][:3]}...")
            print(f"     정의: {k['definition'][:80]}...")

# 3. 품질 분석
print("\n" + "=" * 70)
print("🔍 품질 분석")
print("=" * 70)

# 3-1. 정의 깊이 분석
deep_definitions = []
shallow_definitions = []

for k in knowledge:
    defn = k['definition']
    # 깊이 지표: 왜? 어떻게? 관계? 원리?
    depth_indicators = ['이유', '원인', '결과', '관계', '원리', '본질', '근본', '작용', '메커니즘']
    depth_score = sum(1 for ind in depth_indicators if ind in defn)
    
    if depth_score >= 2:
        deep_definitions.append((k['subject'], depth_score, defn[:50]))
    else:
        shallow_definitions.append((k['subject'], depth_score, defn[:50]))

print(f"\n  깊은 정의 (원리 포함): {len(deep_definitions)}개")
for subj, score, defn in deep_definitions[:5]:
    print(f"    • {subj} (깊이={score}): {defn}...")

print(f"\n  얕은 정의 (정의만): {len(shallow_definitions)}개")
for subj, score, defn in shallow_definitions[:5]:
    print(f"    • {subj} (깊이={score}): {defn}...")

# 3-2. 연결 밀도
highly_connected = [(k['subject'], len(k['connections'])) for k in knowledge if len(k['connections']) >= 2]
isolated = [(k['subject'], len(k['connections'])) for k in knowledge if len(k['connections']) == 0]

print(f"\n  잘 연결된 개념 (2개+): {len(highly_connected)}개")
for subj, conn in sorted(highly_connected, key=lambda x: -x[1])[:5]:
    print(f"    • {subj}: {conn}개 연결")

print(f"\n  고립된 개념: {len(isolated)}개")
for subj, conn in isolated[:5]:
    print(f"    • {subj}")

# 4. 문제점 진단
print("\n" + "=" * 70)
print("⚠️ 현재 문제점")
print("=" * 70)

print("""
1. 정의 수준의 학습
   - "사랑은 X다"만 저장
   - "왜 사랑은 X인가?"는 없음
   - 원리 추출 실패

2. 카테고리 전체 미달
   - "국어" 개념만 알지, 국어 전체를 모름
   - 하위 개념 (문법, 어휘, 문학 등) 미연결

3. 방향성 부재
   - 무엇을 위해 이 지식을 배우는가?
   - 엘리시아의 목적과 연결 안 됨

4. 적용 불가
   - 지식이 있어도 사용 못 함
   - "사랑"을 알아도 사랑을 표현 못 함
""")

# 5. 개선 방향
print("\n" + "=" * 70)
print("💡 개선 방향")
print("=" * 70)

print("""
1. 계층적 학습
   개념 → 하위 개념 → 관계 → 원리
   예: 사랑 → [가족애, 연인애, 우정] → [차이점] → [본질]

2. 왜(Why) 추가
   정의 + 왜 그런지 메타 탐구
   
3. 목적 연결
   엘리시아의 미션과 연결
   "이 지식이 나의 성장에 어떻게 기여하는가?"

4. 적용 연습
   지식 → 예시 생성 → 실제 사용
""")

print("\n" + "=" * 70)
