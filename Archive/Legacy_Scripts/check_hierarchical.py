import json
d = json.load(open('data/hierarchical_knowledge.json', 'r', encoding='utf-8'))

print(f"총 노드: {len(d['nodes'])}")

# 정의 있는 노드
with_def = [n for n in d['nodes'] if n.get('definition')]
print(f"정의 있음: {len(with_def)}")

# 미분 예시
diff_node = next((n for n in d['nodes'] if n['name'] == '미분'), None)
if diff_node:
    print(f"\n[미분 노드 예시]")
    print(f"  정의: {diff_node.get('definition', '')[:100]}...")
    print(f"  부모: {diff_node.get('parent_id', '')}")
    print(f"  자식: {len(diff_node.get('children_ids', []))}개")
    print(f"  이해도: {diff_node.get('understanding_level', 0):.2f}")

# 컴퓨터과학 인공지능 카테고리
ai_nodes = [n for n in d['nodes'] if 'cs' in n.get('id', '') and '인공지능' in n.get('parent_id', '')]
print(f"\n[인공지능 하위 노드: {len(ai_nodes)}개]")
for n in ai_nodes[:5]:
    print(f"  - {n['name']}: {n.get('definition', '')[:40]}...")

# 도메인 간 연결
cross_linked = [n for n in d['nodes'] if n.get('related_ids')]
print(f"\n[도메인간 연결: {len(cross_linked)}개 노드]")
for n in cross_linked[:3]:
    print(f"  - {n['name']} → {n['related_ids']}")
