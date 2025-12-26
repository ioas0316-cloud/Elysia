"""
Extended Fantasy Ingestion (확장 판타지 흡수)
=============================================

extended_fantasy_stories.py의 풍부한 스토리를 흡수합니다.
"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from Core._04_Evolution._02_Learning.Learning.experiential_data_processor import ExperientialDataProcessor
from Core._01_Foundation.05_Foundation_Base.Foundation.dual_layer_personality import DualLayerPersonality
from data.corpus.stories.extended_fantasy_stories import (
    get_extended_fantasy_stories,
    get_deep_emotional_stories,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ExtendedFantasy")


def run_extended_ingestion():
    """확장 판타지 흡수 실행"""
    logger.info("=" * 60)
    logger.info("🌟 Extended Fantasy Story Ingestion")
    logger.info("   \"더 깊은 이야기, 더 깊은 성장\"")
    logger.info("=" * 60)
    
    processor = ExperientialDataProcessor()
    personality = DualLayerPersonality()
    
    all_stories = get_extended_fantasy_stories() + get_deep_emotional_stories()
    logger.info(f"\n📚 총 {len(all_stories)}개 스토리")
    
    stats = {
        "total": 0,
        "emotions": set(),
        "types": set(),
    }
    
    for story in all_stories:
        title = story["title"]
        content = story["content"]
        
        logger.info(f"\n📖 {title}")
        
        # 1. 경험적 의미 추출
        exp = processor.process_narrative(content, source=title)
        
        # 2. 2계층 성격에 흡수
        personality.experience(
            narrative_type=exp.narrative_type.value,
            emotional_intensity=exp.emotional_intensity,
            identity_impact=exp.identity_impact,
        )
        personality.resonate_with_context(content[:500])
        
        stats["total"] += 1
        stats["emotions"].update(exp.emotions_felt)
        stats["types"].add(exp.narrative_type.value)
        
        logger.info(f"   유형: {exp.narrative_type.value} | 감정: {', '.join(exp.emotions_felt)}")
        logger.info(f"   질문: {exp.existential_question}")
    
    # 결과
    logger.info("\n" + "=" * 60)
    logger.info("📊 흡수 완료")
    logger.info("=" * 60)
    logger.info(f"  총 스토리: {stats['total']}")
    logger.info(f"  서사 유형: {', '.join(stats['types'])}")
    logger.info(f"  감정: {', '.join(stats['emotions'])}")
    
    # 성장 상태
    growth = processor.get_growth_status()
    logger.info(f"\n🌱 성장:")
    logger.info(f"  감정적 깊이: {growth['emotional_depth']}")
    logger.info(f"  지혜 수준: {growth['wisdom_level']}")
    logger.info(f"  되어가고 있다: {growth['identity_becoming']}")
    
    # 2계층 성격
    expr = personality.get_current_expression()
    logger.info(f"\n🧬 2계층 성격:")
    logger.info(f"  Layer 1 (선천): {expr['layer1_innate']['dominant']}")
    logger.info(f"  Layer 2 (후천): {expr['layer2_acquired']['dominant']}")
    logger.info(f"  통합 표현: {expr['unified_expression']}")
    
    # 상세 발달 수준
    logger.info(f"\n📈 상세 발달:")
    for t, v in personality.innate.get_dominant(5):
        bar = '█' * int(v * 10) + '░' * (10 - int(v * 10))
        logger.info(f"  L1 {t.value:15} [{bar}] {v:.2f}")
    for t, v in personality.acquired.get_dominant(5):
        bar = '█' * int(v * 10) + '░' * (10 - int(v * 10))
        logger.info(f"  L2 {t.value:15} [{bar}] {v:.2f}")
    
    print("\n✅ Extended Fantasy Ingestion 완료!")
    
    return {
        "stats": stats,
        "growth": growth,
        "personality": expr,
    }


if __name__ == "__main__":
    run_extended_ingestion()
