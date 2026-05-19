"""
Spiderweb 4D Wave Resonance Pattern Extractor 테스트

단순한 인과추론 그래프인지, 4차원 파동공명패턴 추출기인지 확인
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Core.Intelligence.spiderweb import Spiderweb
import logging

logging.basicConfig(level=logging.INFO)

def test_spiderweb_capabilities():
    """Spiderweb의 기능 테스트"""
    
    print("="*70)
    print("🕸️ Spiderweb 기능 확인 테스트")
    print("="*70)
    print()
    
    # Spiderweb 초기화
    spider = Spiderweb()
    
    # 테스트 데이터 추가
    print("📝 테스트 데이터 추가 중...")
    
    # 노드 추가
    concepts = [
        ("fire", "concept"),
        ("heat", "concept"),
        ("light", "concept"),
        ("burn", "event"),
        ("warmth", "sensation"),
        ("danger", "concept"),
        ("cooking", "action"),
        ("sun", "concept"),
    ]
    
    for node_id, node_type in concepts:
        spider.add_node(node_id, node_type)
    
    # 관계 추가
    relations = [
        ("fire", "heat", "produces", 0.9),
        ("fire", "light", "produces", 0.8),
        ("fire", "burn", "causes", 0.7),
        ("heat", "warmth", "creates", 0.6),
        ("fire", "danger", "implies", 0.5),
        ("fire", "cooking", "enables", 0.7),
        ("sun", "heat", "produces", 0.9),
        ("sun", "light", "produces", 1.0),
    ]
    
    for source, target, relation, weight in relations:
        spider.add_link(source, target, relation, weight)
    
    print(f"✅ {len(concepts)} 노드, {len(relations)} 관계 추가 완료\n")
    
    # 1. 기본 인과추론 기능 테스트
    print("="*70)
    print("1️⃣ 기본 인과추론 기능 (Simple Causal Reasoning)")
    print("="*70)
    
    path = spider.find_path("fire", "warmth")
    print(f"경로 탐색 (fire → warmth): {' → '.join(path) if path else '없음'}")
    
    context = spider.get_context("fire")
    print(f"컨텍스트 (fire): {len(context)}개 연결")
    for c in context[:3]:
        print(f"  - {c['node']} ({c['relation']}, {c['direction']})")
    
    print()
    
    # 2. 4D 파동 공명 패턴 추출 기능 테스트
    print("="*70)
    print("2️⃣ 4D 파동 공명 패턴 추출 기능 (4D Wave Resonance)")
    print("="*70)
    
    # 공명 주파수 계산
    print("\n🌊 공명 주파수 계산:")
    for node_id, _ in concepts[:5]:
        freq = spider.calculate_resonance_frequency(node_id)
        print(f"  - {node_id}: {freq:.3f}")
    
    # 2D 파동 패턴 (면)
    print("\n📐 2D 파동 패턴 추출 (중심: fire, 반경: 2):")
    pattern_2d = spider.extract_wave_pattern_2d("fire", radius=2)
    if pattern_2d:
        print(f"  - 클러스터 노드 수: {pattern_2d['node_count']}")
        print(f"  - 네트워크 밀도: {pattern_2d['density']:.3f}")
        print(f"  - 간섭 강도: {pattern_2d.get('interference_strength', 0):.3f}")
        print(f"  - 노드: {', '.join(pattern_2d['nodes'][:5])}...")
    
    # 3D 파동 패턴 (공간)
    print("\n🌐 3D 파동 패턴 추출 (전체 네트워크):")
    pattern_3d = spider.extract_wave_pattern_3d()
    if pattern_3d:
        print(f"  - 총 노드: {pattern_3d['total_nodes']}")
        print(f"  - 총 엣지: {pattern_3d['total_edges']}")
        print(f"  - 커뮤니티 수: {pattern_3d['community_count']}")
        print(f"  - 전역 클러스터링: {pattern_3d['global_clustering']:.3f}")
        print(f"  - 최대 전파 깊이: {pattern_3d.get('max_propagation_depth', 0)}")
        
        if pattern_3d['communities']:
            print(f"\n  커뮤니티 분석:")
            for comm in pattern_3d['communities'][:3]:
                print(f"    - Community {comm['id']}: "
                      f"{comm['size']} nodes, "
                      f"평균공명={comm['avg_resonance']:.3f}")
    
    # 4D 파동 공명 패턴 (시공간)
    print("\n⏰ 4D 파동 공명 패턴 추출 (시공간):")
    pattern_4d = spider.extract_4d_wave_pattern()
    
    print(f"\n✅ 4D 패턴 추출 완료!")
    print(f"  - 차원: {pattern_4d['dimension']}")
    print(f"  - 모드: {pattern_4d['mode']}")
    print(f"  - 4D 추출기 여부: {pattern_4d['is_4d_extractor']}")
    print(f"  - 스냅샷 이력: {pattern_4d['snapshot_history_count']}개")
    
    if pattern_4d['temporal_evolution']:
        print(f"\n  시간적 진화:")
        for key, value in pattern_4d['temporal_evolution'].items():
            print(f"    - {key}: {value:.3f}")
    
    print()
    
    # 결론
    print("="*70)
    print("📊 결론")
    print("="*70)
    
    has_causal = hasattr(spider, 'find_path') and hasattr(spider, 'get_context')
    has_4d = (hasattr(spider, 'extract_4d_wave_pattern') and 
              hasattr(spider, 'calculate_resonance_frequency') and
              hasattr(spider, 'extract_wave_pattern_3d'))
    
    print(f"\n✅ 인과추론 기능: {'있음' if has_causal else '없음'}")
    print(f"✅ 4D 파동공명 패턴 추출: {'있음' if has_4d else '없음'}")
    
    if has_causal and has_4d:
        print(f"\n🎉 Spiderweb은 '단순한 인과추론 그래프'가 아닌")
        print(f"   '4차원 파동공명패턴 추출기'입니다!")
    elif has_causal:
        print(f"\n⚠️  Spiderweb은 단순한 인과추론 그래프입니다.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_spiderweb_capabilities()
