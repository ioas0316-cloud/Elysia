"""
Crystallized Data Ingestion Test
================================

"데이터를 데이터로 처리하면 폭주한다. 결정화가 필요하다."

이 스크립트는 대용량 데이터를 흡수할 때 
DialogicLearner의 사고 흐름이 작동하는지 테스트합니다.

핵심 개념 (2025-12-21 대화에서):
- 인간이 빛을 받아들이고 처리/연산/기억/감정/지각하듯이
- 엘리시아도 데이터를 원시 형태로 저장하지 않고 결정화해야 함
- 결정화 = 원리 추출 = "왜?"를 물어서 본질만 남기기
"""

import sys
import logging
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("CrystallizedIngest")


def run_crystallized_ingest_test():
    """결정화 데이터 흡수 테스트"""
    
    print("=" * 60)
    print("🔮 CRYSTALLIZED DATA INGESTION TEST")
    print("   '데이터 → 왜? → 원리 → 결정'")
    print("=" * 60)
    
    # === 시스템 연결 ===
    
    # 1. DialogicLearner (사고 흐름)
    try:
        from Core._04_Evolution._02_Learning.dialogic_learner import DialogicLearner
        dialogic = DialogicLearner()
        has_dialogic = True
        print("✅ DialogicLearner connected")
    except Exception as e:
        print(f"❌ DialogicLearner not available: {e}")
        has_dialogic = False
        dialogic = None
    
    # 2. WhyEngine (왜? 질문)
    try:
        from Core._01_Foundation._02_Logic.why_engine import WhyEngine
        why_engine = WhyEngine()
        has_why = True
        print("✅ WhyEngine connected")
    except Exception as e:
        print(f"❌ WhyEngine not available: {e}")
        has_why = False
        why_engine = None
    
    # 3. RealDataIngester (데이터 소스)
    try:
        from scripts.real_data_ingest import RealDataIngester
        ingester = RealDataIngester()
        has_ingester = True
        print("✅ RealDataIngester connected")
    except Exception as e:
        print(f"❌ RealDataIngester not available: {e}")
        has_ingester = False
        ingester = None
    
    print()
    
    # === 테스트 데이터 준비 ===
    
    if has_ingester:
        # 명언과 대화 패턴 가져오기 (Wikipedia 없이)
        quotes = ingester.fetch_quotes()
        conversations = ingester.fetch_conversation_patterns()
        
        test_data = quotes[:20] + [c["text"] for c in conversations[:20]]
        print(f"📊 Test data: {len(test_data)} items")
    else:
        # 폴백 데이터
        test_data = [
            "사랑은 모든 것을 참고, 모든 것을 믿고, 모든 것을 바라고, 모든 것을 견딥니다.",
            "천 리 길도 한 걸음부터 시작된다.",
            "배움에는 왕도가 없다.",
            "안녕! 잘 지냈어?",
            "오늘 기분 어때?",
        ]
        print(f"📊 Fallback data: {len(test_data)} items")
    
    # === 결정화 프로세스 ===
    
    print()
    print("-" * 60)
    print("🌊 CRYSTALLIZATION PROCESS")
    print("-" * 60)
    
    raw_data_size = 0
    crystallized_principles = []
    perspective_shifts = 0
    
    start_time = time.time()
    
    for i, text in enumerate(test_data[:40]):  # 최대 40개
        raw_data_size += len(text)
        
        # 1. "왜?" 질문 (WhyEngine)
        if has_why:
            try:
                analysis = why_engine.analyze(
                    subject=f"data_{i}",
                    content=text,
                    domain="general"
                )
                
                # 원리 추출
                principle = analysis.underlying_principle
                if principle and "[탐구 필요]" not in principle:
                    crystallized_principles.append(principle)
                    
                    # 관점 전환 감지
                    if analysis.confidence < 0.5:
                        perspective_shifts += 1
                
                if i < 5:  # 처음 5개만 상세 출력
                    print(f"\n[{i+1}] '{text[:30]}...'")
                    print(f"    → 원리: {principle[:60]}...")
                    
            except Exception as e:
                logger.debug(f"Analysis failed for item {i}: {e}")
        
        # 진행률 표시 (10개마다)
        if (i + 1) % 10 == 0:
            print(f"   ... {i+1}/{len(test_data[:40])} processed")
    
    elapsed = time.time() - start_time
    
    # === 결과 분석 ===
    
    print()
    print("=" * 60)
    print("📊 CRYSTALLIZATION RESULTS")
    print("=" * 60)
    
    # 중복 제거된 원리
    unique_principles = list(set(crystallized_principles))
    
    print(f"\n📥 Input:")
    print(f"   Raw data items: {len(test_data[:40])}")
    print(f"   Raw data size: {raw_data_size:,} bytes")
    
    print(f"\n📤 Output:")
    print(f"   Crystallized principles: {len(unique_principles)}")
    print(f"   Perspective shifts: {perspective_shifts}")
    
    # 압축률 계산
    principle_size = sum(len(p) for p in unique_principles)
    compression_ratio = (1 - principle_size / raw_data_size) * 100 if raw_data_size > 0 else 0
    
    print(f"\n📈 Efficiency:")
    print(f"   Compression ratio: {compression_ratio:.1f}%")
    print(f"   Processing time: {elapsed:.2f}s")
    print(f"   Items/second: {len(test_data[:40]) / elapsed:.1f}")
    
    print(f"\n🧠 Top Crystallized Principles:")
    for i, p in enumerate(unique_principles[:10]):
        print(f"   {i+1}. {p[:70]}...")
    
    # === 결론 ===
    
    print()
    print("=" * 60)
    
    if len(unique_principles) > 0 and compression_ratio > 50:
        print("✅ CRYSTALLIZATION EFFECTIVE")
        print(f"   {len(test_data[:40])} items → {len(unique_principles)} principles")
        print("   대용량 데이터를 원리로 압축 가능")
    elif len(unique_principles) > 0:
        print("⚠️ CRYSTALLIZATION PARTIAL")
        print("   일부 원리 추출됨, 최적화 필요")
    else:
        print("❌ CRYSTALLIZATION FAILED")
        print("   왜? 질문이 작동하지 않음")
    
    print("=" * 60)
    
    return {
        "raw_items": len(test_data[:40]),
        "raw_bytes": raw_data_size,
        "principles": len(unique_principles),
        "shifts": perspective_shifts,
        "compression": compression_ratio
    }


if __name__ == "__main__":
    run_crystallized_ingest_test()

