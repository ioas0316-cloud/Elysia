"""잠재 지식 상세 분석"""
import json
import os

# 현재 상태
current_path = 'data/potential_knowledge.json'

print("=" * 70)
print("💭 잠재 지식 상세 분석")
print("=" * 70)

if os.path.exists(current_path):
    data = json.load(open(current_path, 'r', encoding='utf-8'))
    
    print(f"\n📊 현재 잠재 지식: {len(data['knowledge'])}개")
    print(f"💎 확정된 지식: {data['crystallized_count']}개")
    
    if data['knowledge']:
        print("\n" + "-" * 70)
        print("📖 잠재 지식 목록:")
        print("-" * 70)
        
        for k in data['knowledge']:
            print(f"\n  📌 {k['subject']}")
            print(f"     정의: {k['definition'][:80]}...")
            print(f"     소스: {k['source']}")
            print(f"     주파수: {k['frequency']:.2f} (0.70 이상이면 확정 가능)")
            print(f"     확인 횟수: {k['confirmations']}")
            print(f"     연결: {k['connections'] if k['connections'] else '없음'}")
            print(f"     생성일: {k['created_at'][:19]}")
            if k['last_connected']:
                print(f"     마지막 연결: {k['last_connected'][:19]}")
    else:
        print("\n  (모든 잠재 지식이 확정되어 비어있음)")
else:
    print("  파일이 없습니다.")

# 테스트 저장소도 확인
test_path = 'data/test_potential.json'
if os.path.exists(test_path):
    print("\n" + "=" * 70)
    print("🧪 테스트 저장소")
    print("=" * 70)
    
    test_data = json.load(open(test_path, 'r', encoding='utf-8'))
    print(f"\n📊 테스트 잠재 지식: {len(test_data['knowledge'])}개")
    
    for k in test_data['knowledge']:
        print(f"\n  📌 {k['subject']}: freq={k['frequency']:.2f}")
        print(f"     연결: {k['connections']}")

print("\n" + "=" * 70)
