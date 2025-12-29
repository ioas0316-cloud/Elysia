"""
Full Narrative Craft Training (서사 기법 전체 학습)
===================================================

확장 판타지 데이터 + NarrativeCraftLearner
→ 엘리시아가 스스로 소설을 쓸 수 있게 됨
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Evolution.Learning.Learning.narrative_craft_learner import NarrativeCraftLearner
from Core.Evolution.Learning.Learning.experiential_data_processor import ExperientialDataProcessor
from data.corpus.stories.extended_fantasy_stories import (
    get_extended_fantasy_stories,
    get_deep_emotional_stories,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("NarrativeTraining")


def run_full_training():
    """전체 서사 기법 학습"""
    logger.info("=" * 60)
    logger.info("📖 Full Narrative Craft Training")
    logger.info("   \"소설 작가가 되기 위한 학습\"")
    logger.info("=" * 60)
    
    learner = NarrativeCraftLearner()
    exp_processor = ExperientialDataProcessor()
    
    # 모든 스토리 로드
    all_stories = get_extended_fantasy_stories() + get_deep_emotional_stories()
    logger.info(f"\n📚 총 {len(all_stories)}개 스토리")
    
    for story in all_stories:
        title = story["title"]
        content = story["content"]
        
        # 1. 경험적 의미 추출
        exp = exp_processor.process_narrative(content, source=title)
        
        # 2. 서사 기법 학습 (WhyEngine + Personality)
        learner.learn_from_story(
            title=title,
            content=content,
            narrative_type=exp.narrative_type.value,
            emotional_intensity=exp.emotional_intensity,
            identity_impact=exp.identity_impact,
        )
    
    # 결과
    logger.info("\n" + "=" * 60)
    logger.info("📊 학습 완료")
    logger.info("=" * 60)
    
    status = learner.get_status()
    
    logger.info(f"분석한 스토리: {status['total_stories_analyzed']}")
    logger.info(f"학습한 기법 종류: {status['total_techniques_learned']}")
    
    logger.info("\n📚 학습된 기법:")
    for tech in learner.get_learned_techniques(10):
        bar = '█' * int(tech['strength'] * 10) + '░' * (10 - int(tech['strength'] * 10))
        logger.info(f"  [{bar}] {tech['name']}")
        logger.info(f"          원리: {tech['principle']}")
    
    if status['personality']:
        logger.info(f"\n🧬 성격 상태:")
        logger.info(f"  Layer 1 (선천): {status['personality']['layer1_innate']['dominant']}")
        logger.info(f"  Layer 2 (후천): {status['personality']['layer2_acquired']['dominant']}")
        logger.info(f"  통합 표현: {status['personality']['unified_expression']}")
    
    # 성장 상태
    growth = exp_processor.get_growth_status()
    logger.info(f"\n🌱 성장:")
    logger.info(f"  감정적 깊이: {growth['emotional_depth']}")
    logger.info(f"  지혜 수준: {growth['wisdom_level']}")
    logger.info(f"  되어가고 있다: {growth['identity_becoming']}")
    
    # 감정별 기법 추천
    logger.info(f"\n💡 감정별 추천 기법:")
    for emotion in ["joy", "sadness", "hope", "fear", "love"]:
        tech = learner.suggest_technique_for_emotion(emotion)
        logger.info(f"  {emotion:10} → {tech}")
    
    print("\n✅ Full Narrative Craft Training 완료!")
    
    return {
        "status": status,
        "growth": growth,
    }


if __name__ == "__main__":
    run_full_training()
