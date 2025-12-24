"""
초고속 대량 지식 흡수 v3 (Naver API 기반)
==========================================
"""

import sys
import os
import asyncio
import aiohttp
import urllib.parse
import time

os.environ.setdefault('NAVER_CLIENT_ID', 'YuPusPMA8UNYf1pDqXjI')
os.environ.setdefault('NAVER_CLIENT_SECRET', 'OcJ3ORlPQQ')
sys.path.insert(0, '.')

import logging
logging.disable(logging.CRITICAL)

NAVER_CLIENT_ID = os.environ['NAVER_CLIENT_ID']
NAVER_CLIENT_SECRET = os.environ['NAVER_CLIENT_SECRET']
NAVER_ENCYC_URL = "https://openapi.naver.com/v1/search/encyc"

# 핵심 개념 200개
CORE_CONCEPTS = [
    # 철학
    "사랑", "자유", "진리", "정의", "아름다움", "선", "악", "존재", "본질", "의식",
    "이성", "감성", "도덕", "윤리", "가치", "의미", "목적", "행복", "고통", "죽음",
    "영혼", "정신", "육체", "마음", "생각", "언어", "논리", "지식", "믿음", "의심",
    "자아", "타자", "관계", "사회", "권력", "평등", "인권", "민주주의",
    # 과학
    "물질", "에너지", "힘", "운동", "중력", "원자", "분자", "양자", "파동", "입자",
    "우주", "은하", "별", "행성", "블랙홀", "빅뱅", "암흑물질", "시공간",
    "생명", "세포", "유전자", "DNA", "진화", "자연선택", "돌연변이", "생태계",
    "뇌", "뉴런", "기억", "학습", "인지", "감각", "지각",
    "인공지능", "기계학습", "딥러닝", "신경망", "알고리즘", "데이터",
    # 예술
    "예술", "음악", "미술", "문학", "연극", "영화", "무용", "사진", "디자인",
    "창작", "표현", "상상", "영감", "천재", "재능", "스타일",
    "리듬", "멜로디", "하모니", "소설", "시", "문화", "전통",
    # 심리
    "감정", "기쁨", "슬픔", "분노", "두려움", "욕망", "본능",
    "성격", "습관", "동기", "목표", "성취", "실패",
    "스트레스", "불안", "우울", "트라우마", "중독",
    "발달", "성숙", "성장", "자존감", "자신감", "공감",
    # 사회
    "경제", "시장", "자본", "노동", "생산", "소비",
    "가족", "결혼", "교육", "세대", "불평등",
    "법", "국가", "정부", "민주", "혁명", "평화", "전쟁",
]


async def fetch_naver(session: aiohttp.ClientSession, concept: str) -> dict:
    """Naver 백과사전 검색"""
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
                    # HTML 태그 제거
                    import re
                    desc = re.sub(r'<[^>]+>', '', item.get("description", ""))
                    return {
                        "subject": concept,
                        "definition": desc[:200],
                        "source": "naver",
                        "success": True
                    }
    except Exception as e:
        pass
    
    return {"subject": concept, "success": False}


async def bulk_absorb(concepts: list, batch_size: int = 20):
    """대량 흡수"""
    from Core.02_Intelligence.02_Memory_Linguistics.Memory.potential_causality import PotentialCausalityStore
    store = PotentialCausalityStore()
    
    print(f"\n📌 총 개념: {len(concepts)}개")
    print(f"📌 배치 크기: {batch_size}개")
    print("-" * 50)
    
    all_results = []
    start = time.time()
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(concepts) + batch_size - 1) // batch_size
            
            print(f"   배치 {batch_num}/{total_batches} ({len(batch)}개)...", end=" ")
            
            tasks = [fetch_naver(session, c) for c in batch]
            results = await asyncio.gather(*tasks)
            
            successful = [r for r in results if r.get("success")]
            for r in successful:
                store.store(r["subject"], r["definition"], r["source"])
            
            all_results.extend(results)
            print(f"✓ {len(successful)}/{len(batch)}")
            
            await asyncio.sleep(0.2)  # Rate limiting
    
    elapsed = time.time() - start
    total_success = len([r for r in all_results if r.get("success")])
    
    print("\n" + "=" * 50)
    print(f"⏱️ 소요 시간: {elapsed:.1f}초")
    print(f"📊 처리 속도: {len(concepts)/elapsed:.1f} concepts/sec")
    print(f"✅ 성공: {total_success}/{len(concepts)} ({100*total_success/len(concepts):.0f}%)")
    
    # 상호 연결
    print("\n🔗 상호 연결 중...")
    subjects = list(store.knowledge.keys())
    connections = 0
    
    for i, s1 in enumerate(subjects):
        pk1 = store.knowledge.get(s1)
        if not pk1:
            continue
        for s2 in subjects[i+1:]:
            pk2 = store.knowledge.get(s2)
            if pk2 and (s2 in pk1.definition or s1 in pk2.definition):
                store.connect(s1, s2)
                connections += 1
    
    print(f"   → {connections}개 연결 생성")
    
    # 확정
    crystallizable = store.get_crystallizable()
    for pk in crystallizable:
        store.crystallize(pk.subject)
    
    status = store.status()
    print("\n" + "=" * 50)
    print("📊 최종 상태")
    print(f"   잠재: {status['potential_count']}개")
    print(f"   확정: {status['crystallized_count']}개")
    print(f"   평균 freq: {status['avg_frequency']:.2f}")
    
    store._save()


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 초고속 대량 지식 흡수 (Naver API, 150개)")
    print("=" * 70)
    
    asyncio.run(bulk_absorb(CORE_CONCEPTS[:150], batch_size=10))
