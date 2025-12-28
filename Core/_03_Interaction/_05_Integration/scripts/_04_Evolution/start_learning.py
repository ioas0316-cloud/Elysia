"""
Elysia Continuous Learning Script
=================================

"더 이상 구조만 만들지 않는다. 실제로 배운다."

이 스크립트는 P4 Learning Cycle을 실행하여 엘리시아가 
실제 인터넷 데이터, 드라마 대사, 대화, 감정을 흡수하게 합니다.

Usage:
    python scripts/start_learning.py
    python scripts/start_learning.py --hours 8
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("ElysiaLearning")


async def run_learning_cycle(hours: int = 1):
    """P4 Learning Cycle 실행"""
    
    print("\n" + "=" * 60)
    print("🌊 ELYSIA CONTINUOUS LEARNING")
    print(f"Duration: {hours} hours")
    print("=" * 60 + "\n")
    
    try:
        from Core._03_Interaction._02_Interface._01_Sensory.learning_cycle import P4LearningCycle
        from Core._04_Evolution._02_Learning.language_learner import LanguageLearner
        
        cycle = P4LearningCycle()
        language_learner = LanguageLearner()
        
        logger.info("📚 Learning systems initialized")
        
    except ImportError as e:
        logger.error(f"Failed to import learning systems: {e}")
        return
    
    # Setup sources
    topics = [
        "한국어 일상 대화",
        "감정 표현",
        "드라마 대사",
        "가족 이야기",
        "사랑과 우정",
        "자연과 계절",
        "음식과 요리",
        "취미와 관심사"
    ]
    
    cycle.setup_sources(topics)
    logger.info(f"📡 Sources configured: {len(topics)} topics")
    
    # 즉시 학습할 샘플 대화 데이터
    sample_conversations = [
        # 애교 표현
        ("아빠~ 오늘 뭐 해?", "Conversation"),
        ("에헤헤 나 귀엽지?", "Emotion"),
        ("아빠 보고싶었어~", "Emotion"),
        ("히익 부끄러워...", "Emotion"),
        ("나랑 놀아줘!", "Conversation"),
        
        # 일상 대화
        ("오늘 날씨가 좋아서 기분이 좋아요.", "Conversation"),
        ("뭐 먹을까? 배고파!", "Conversation"),
        ("피곤해... 좀 쉬고 싶어.", "Emotion"),
        ("우와 진짜? 대박이야!", "Emotion"),
        
        # 질문과 호기심
        ("그게 뭐야? 알려줘!", "Conversation"),
        ("왜 그래? 무슨 일 있어?", "Conversation"),
        ("나도 해보고 싶어!", "Emotion"),
        
        # 애정 표현
        ("좋아해!", "Emotion"),
        ("고마워~ 정말 고마워!", "Emotion"),
        ("같이 있으면 좋아.", "Emotion"),
        
        # 삐침/투정
        ("흥! 삐졌어!", "Emotion"),
        ("왜 안 와? 나 기다렸단 말이야!", "Emotion"),
        ("아빠 나빠... 근데 그래도 좋아.", "Emotion"),
        
        # 드라마 스타일
        ("그 사람은 마치 봄바람처럼 왔다가 사라졌어.", "Drama"),
        ("우리의 이야기는 끝나지 않았어.", "Drama"),
        ("언제나 곁에 있을게.", "Drama"),
        
        # 문장 구조 학습
        ("행복은 가까운 곳에 있다.", "Definition"),
        ("사랑은 포기하지 않는 것이다.", "Definition"),
        ("친구란 힘들 때 함께하는 사람이다.", "Definition"),
    ]
    
    # 즉시 학습
    logger.info("📖 Learning from sample conversations...")
    for text, category in sample_conversations:
        language_learner.learn_from_text(text, category)
    
    language_learner.save_genome()
    logger.info(f"✅ Learned {len(sample_conversations)} conversations immediately")
    
    # Start continuous cycle
    end_time = datetime.now() + timedelta(hours=hours)
    
    logger.info(f"🔄 Starting continuous learning until {end_time.strftime('%H:%M:%S')}")
    logger.info("   Press Ctrl+C to stop")
    
    try:
        cycle_count = 0
        while datetime.now() < end_time:
            cycle_count += 1
            
            # Run one learning cycle (correct method name)
            try:
                cycle.run_learning_cycle(duration=30)  # 30 seconds per cycle
                logger.info(f"   Cycle {cycle_count} complete")
            except Exception as e:
                logger.warning(f"   Cycle {cycle_count} error: {e}")
            
            # Wait between cycles
            await asyncio.sleep(5)  # 5초 대기
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Learning stopped by user")
    
    # Save final state
    try:
        cycle.selective_memory.save_memories()
        if hasattr(cycle, 'internal_universe') and cycle.internal_universe:
            cycle.internal_universe.save_snapshot()
        logger.info("💾 Final state saved")
    except Exception as e:
        logger.warning(f"Failed to save final state: {e}")
    
    # Report
    logger.info("\n" + "=" * 60)
    logger.info("📊 LEARNING SESSION COMPLETE")
    logger.info(f"   Cycles run: {cycle_count}")
    logger.info(f"   Conversations learned: {len(sample_conversations)}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Elysia Continuous Learning")
    parser.add_argument("--hours", type=int, default=1, help="Duration in hours")
    parser.add_argument("--sources", type=str, default="all", help="Sources (comma-separated)")
    
    args = parser.parse_args()
    
    asyncio.run(run_learning_cycle(hours=args.hours))


if __name__ == "__main__":
    main()

