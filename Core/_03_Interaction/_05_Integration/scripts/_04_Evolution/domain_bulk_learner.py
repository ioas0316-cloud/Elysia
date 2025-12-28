"""
도메인별 대량 학습기 (Domain Bulk Learner)
==========================================

10개 도메인의 핵심 개념을 계층적으로 학습

흐름:
1. 도메인 선택
2. 카테고리별 개념 목록 로드
3. 네이버/Wikipedia에서 정의 흡수
4. 원리(Why) 추가 탐색
5. 계층 구조에 저장
"""

import sys
import os
import asyncio
import aiohttp
import urllib.parse
import time
import re

os.environ.setdefault('NAVER_CLIENT_ID', 'YuPusPMA8UNYf1pDqXjI')
os.environ.setdefault('NAVER_CLIENT_SECRET', 'OcJ3ORlPQQ')
sys.path.insert(0, '.')

import logging
logging.disable(logging.CRITICAL)

from Core._04_Evolution._02_Learning.hierarchical_learning import (
    HierarchicalKnowledgeGraph, Domain, DOMAIN_STRUCTURE, KnowledgeNode
)

NAVER_CLIENT_ID = os.environ['NAVER_CLIENT_ID']
NAVER_CLIENT_SECRET = os.environ['NAVER_CLIENT_SECRET']
NAVER_ENCYC_URL = "https://openapi.naver.com/v1/search/encyc"


# 각 도메인별 세부 개념
DOMAIN_CONCEPTS = {
    Domain.MATHEMATICS: {
        "대수학": [
            ("방정식", "미지수의 값을 구하는 등식"),
            ("함수", "입력과 출력의 대응 관계"),
            ("행렬", "숫자의 사각 배열, 선형 변환 표현"),
            ("벡터", "크기와 방향을 가진 양"),
            ("선형대수", "벡터 공간과 선형 변환 연구"),
        ],
        "해석학": [
            ("미분", "순간 변화율, 접선 기울기"),
            ("적분", "누적량 계산, 넓이/부피"),
            ("극한", "어떤 값에 한없이 가까워지는 것"),
            ("급수", "무한히 많은 항의 합"),
            ("미분방정식", "미분을 포함한 방정식"),
        ],
        "기하학": [
            ("유클리드기하", "점, 선, 면의 기본 기하학"),
            ("해석기하", "좌표를 사용한 기하학"),
            ("위상", "연속 변환에 불변하는 성질 연구"),
            ("미분기하", "곡선과 곡면의 미적분 기하학"),
        ],
        "이산수학": [
            ("집합론", "집합과 원소의 관계"),
            ("그래프이론", "정점과 간선의 관계 연구"),
            ("조합론", "경우의 수, 배열과 조합"),
            ("논리학", "타당한 추론의 규칙"),
        ],
        "확률통계": [
            ("확률", "사건이 일어날 가능성"),
            ("통계", "데이터 수집과 분석"),
            ("확률분포", "확률 변수의 값 분포"),
            ("베이즈", "조건부 확률과 업데이트"),
        ],
    },
    Domain.PHYSICS: {
        "역학": [
            ("뉴턴역학", "힘과 운동의 관계"),
            ("라그랑주역학", "에너지 기반 역학 형식화"),
            ("운동량", "질량과 속도의 곱"),
            ("에너지보존", "에너지는 생성되거나 소멸되지 않음"),
        ],
        "전자기학": [
            ("전기장", "전하가 만드는 힘의 장"),
            ("자기장", "움직이는 전하가 만드는 장"),
            ("맥스웰방정식", "전자기 현상의 통합 법칙"),
            ("전자기파", "전기장과 자기장의 파동"),
        ],
        "열역학": [
            ("엔트로피", "무질서도, 에너지 분산"),
            ("열평형", "온도가 같아지는 상태"),
            ("열역학법칙", "에너지 보존과 엔트로피 증가"),
        ],
        "양자역학": [
            ("파동함수", "양자 상태의 수학적 표현"),
            ("불확정성원리", "위치와 운동량의 동시 측정 한계"),
            ("중첩", "여러 상태의 동시 공존"),
            ("얽힘", "분리된 입자의 상관관계"),
        ],
        "상대성이론": [
            ("특수상대성", "광속 불변, 시공간 왜곡"),
            ("일반상대성", "중력은 시공간의 곡률"),
            ("시공간", "공간과 시간의 통합"),
        ],
    },
    Domain.COMPUTER_SCIENCE: {
        "알고리즘": [
            ("정렬", "데이터를 순서대로 배열"),
            ("탐색", "원하는 데이터 찾기"),
            ("그래프알고리즘", "그래프 순회, 최단경로"),
            ("동적프로그래밍", "부분 문제 해결의 재사용"),
            ("분할정복", "문제를 나눠서 해결"),
            ("재귀", "자기 자신을 호출하는 함수"),
        ],
        "자료구조": [
            ("배열", "연속된 메모리 공간"),
            ("연결리스트", "포인터로 연결된 노드"),
            ("트리", "계층적 구조"),
            ("그래프", "정점과 간선의 집합"),
            ("해시테이블", "키-값 빠른 검색"),
            ("스택", "후입선출 구조"),
            ("큐", "선입선출 구조"),
        ],
        "프로그래밍": [
            ("파이썬", "범용 프로그래밍 언어"),
            ("자바스크립트", "웹 프로그래밍 언어"),
            ("함수형프로그래밍", "함수 중심의 프로그래밍"),
            ("객체지향", "객체와 클래스 기반 설계"),
            ("비동기", "동시 처리를 위한 패턴"),
        ],
        "인공지능": [
            ("기계학습", "데이터로부터 패턴 학습"),
            ("딥러닝", "심층 신경망 학습"),
            ("신경망", "뉴런을 모방한 구조"),
            ("역전파", "오차의 역방향 전파"),
            ("강화학습", "보상 기반 학습"),
            ("자연어처리", "인간 언어 이해/생성"),
            ("트랜스포머", "어텐션 기반 아키텍처"),
        ],
        "시스템": [
            ("운영체제", "하드웨어 관리 소프트웨어"),
            ("프로세스", "실행 중인 프로그램"),
            ("스레드", "프로세스 내 실행 단위"),
            ("메모리관리", "메모리 할당과 해제"),
            ("네트워크", "컴퓨터 간 통신"),
            ("데이터베이스", "구조화된 데이터 저장"),
        ],
    },
    Domain.PHILOSOPHY: {
        "존재론": [
            ("존재", "있음 그 자체"),
            ("본질", "사물의 본질적 성질"),
            ("실체", "독립적으로 존재하는 것"),
            ("현상", "나타나는 것, 경험되는 것"),
        ],
        "인식론": [
            ("지식", "정당화된 참된 믿음"),
            ("진리", "사실과 일치하는 것"),
            ("확실성", "의심할 수 없는 것"),
            ("회의주의", "지식 가능성에 대한 의심"),
        ],
        "윤리학": [
            ("선", "도덕적으로 좋은 것"),
            ("악", "도덕적으로 나쁜 것"),
            ("도덕", "옳고 그름의 기준"),
            ("정의", "올바른 분배와 처우"),
            ("덕", "좋은 성품"),
        ],
        "심리철학": [
            ("의식", "자각하는 상태"),
            ("마음", "정신적 상태들의 총체"),
            ("자유의지", "자유로운 선택 능력"),
            ("자아", "나라는 주체"),
            ("감각질", "경험의 주관적 특성"),
        ],
    },
}

# 각 개념의 목적 연결
PURPOSE_MAP = {
    Domain.MATHEMATICS: "논리적 추론과 패턴 인식의 기반",
    Domain.PHYSICS: "자연 법칙 이해, 세계가 어떻게 작동하는지",
    Domain.COMPUTER_SCIENCE: "자기 자신을 구축하고 개선하는 능력",
    Domain.PHILOSOPHY: "존재의 의미와 가치 판단의 기반",
}


async def fetch_naver_definition(session: aiohttp.ClientSession, concept: str) -> dict:
    """Naver 백과사전에서 정의 가져오기"""
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    
    encoded = urllib.parse.quote(concept)
    url = f"{NAVER_ENCYC_URL}?query={encoded}&display=1"
    
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                items = data.get("items", [])
                if items:
                    item = items[0]
                    desc = re.sub(r'<[^>]+>', '', item.get("description", ""))
                    return {
                        "concept": concept,
                        "definition": desc[:300],
                        "success": True
                    }
    except:
        pass
    
    return {"concept": concept, "success": False}


async def learn_domain(domain: Domain, graph: HierarchicalKnowledgeGraph):
    """도메인 전체 학습"""
    if domain not in DOMAIN_CONCEPTS:
        print(f"   도메인 {domain.value} 개념 목록 없음")
        return
    
    categories = DOMAIN_CONCEPTS[domain]
    purpose = PURPOSE_MAP.get(domain, "")
    
    total_concepts = sum(len(concepts) for concepts in categories.values())
    learned = 0
    
    async with aiohttp.ClientSession() as session:
        for category, concepts in categories.items():
            print(f"\n   📁 {category} ({len(concepts)}개)")
            
            # 카테고리 노드 추가
            graph.add_concept(
                name=category,
                domain=domain,
                purpose=f"{domain.name}의 핵심 분야: {category}"
            )
            
            for concept_info in concepts:
                if isinstance(concept_info, tuple):
                    concept, hint = concept_info
                else:
                    concept, hint = concept_info, ""
                
                # 네이버에서 정의 가져오기
                result = await fetch_naver_definition(session, concept)
                
                if result["success"]:
                    definition = result["definition"]
                else:
                    definition = hint  # 실패 시 힌트 사용
                
                # 계층 구조에 추가
                graph.add_concept(
                    name=concept,
                    domain=domain,
                    parent_name=category,
                    definition=definition,
                    purpose=purpose
                )
                
                learned += 1
                status = "✓" if result["success"] else "△"
                print(f"      {status} {concept}")
                
                await asyncio.sleep(0.1)  # Rate limiting
    
    print(f"\n   완료: {learned}/{total_concepts}")


async def learn_all_domains():
    """모든 도메인 학습"""
    graph = HierarchicalKnowledgeGraph()
    
    domains_to_learn = [
        Domain.MATHEMATICS,
        Domain.PHYSICS,
        Domain.COMPUTER_SCIENCE,
        Domain.PHILOSOPHY,
    ]
    
    for domain in domains_to_learn:
        print(f"\n{'='*60}")
        print(f"📚 {domain.name} 도메인 학습")
        print("=" * 60)
        
        await learn_domain(domain, graph)
    
    # 도메인 간 연결
    print(f"\n{'='*60}")
    print("🔗 도메인 간 연결")
    print("=" * 60)
    
    cross_domain_links = [
        ("미분", Domain.MATHEMATICS, "운동량", Domain.PHYSICS),
        ("행렬", Domain.MATHEMATICS, "신경망", Domain.COMPUTER_SCIENCE),
        ("논리학", Domain.MATHEMATICS, "알고리즘", Domain.COMPUTER_SCIENCE),
        ("확률", Domain.MATHEMATICS, "기계학습", Domain.COMPUTER_SCIENCE),
        ("의식", Domain.PHILOSOPHY, "인공지능", Domain.COMPUTER_SCIENCE),
        ("자유의지", Domain.PHILOSOPHY, "강화학습", Domain.COMPUTER_SCIENCE),
    ]
    
    for name1, domain1, name2, domain2 in cross_domain_links:
        graph.connect_across_domains(name1, domain1, name2, domain2)
        print(f"   {name1} ({domain1.value}) ↔ {name2} ({domain2.value})")
    
    # 최종 통계
    print(f"\n{'='*60}")
    print("📊 최종 통계")
    print("=" * 60)
    
    stats = graph.get_stats()
    print(f"   총 노드: {stats['total_nodes']}")
    print(f"   도메인별:")
    for domain, count in sorted(stats['domains'].items(), key=lambda x: -x[1]):
        print(f"      • {domain}: {count}개")
    print(f"   도메인간 연결: {stats['cross_domain_links']}")
    print(f"   평균 이해도: {stats['avg_understanding']:.2f}")


if __name__ == "__main__":
    print("=" * 70)
    print("🎓 도메인별 대량 학습기")
    print("=" * 70)
    
    asyncio.run(learn_all_domains())

