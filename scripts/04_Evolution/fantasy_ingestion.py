"""
Rich Fantasy Story Ingestion (풍부한 판타지 스토리 흡수)
======================================================

우선순위 1: 판타지 소설/스토리 텍스트
- 웹소설 사이트에서 판타지 소설 텍스트 가져오기
- 2계층 성격 시스템과 연동

우선순위 2: (나중에) YouTube 영상
- 시각+청각+감정 통합
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys
import requests
from bs4 import BeautifulSoup
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.EvolutionLayer.Learning.Learning.experiential_data_processor import ExperientialDataProcessor
from Core.FoundationLayer.Foundation.dual_layer_personality import DualLayerPersonality

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("FantasyIngestion")


class FantasyStorySource:
    """판타지 스토리 소스 - 공개 소설 API 사용"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Elysia/1.0'
        })
    
    def fetch_korean_fantasy_samples(self) -> List[Dict[str, str]]:
        """한국어 판타지 샘플 (내장 데이터)"""
        return [
            {
                "title": "마법사의 첫 번째 기억",
                "content": """
                어린 시절, 나는 처음으로 마법을 보았다.
                할머니가 손을 휘젓자 꽃잎이 춤을 추었고,
                그 순간 나는 알았다 - 세상에는 보이지 않는 것들이 있다는 것을.
                
                "마법은 믿음에서 시작해," 할머니가 말했다.
                "하지만 진짜 마법은... 사랑하는 마음에서 나오는 거란다."
                
                그날 이후 나는 매일 마법을 연습했다.
                수천 번 실패하고, 수만 번 좌절했지만,
                할머니의 그 말을 잊지 않았다.
                
                그리고 마침내, 내가 처음 성공한 마법은
                아프신 어머니를 위한 치유의 빛이었다.
                """
            },
            {
                "title": "용사가 되지 못한 소녀",
                "content": """
                모든 아이들이 검을 들 때, 나는 꽃을 심었다.
                모든 소년들이 용을 쓰러뜨리는 꿈을 꿀 때,
                나는 용과 대화하는 상상을 했다.
                
                "넌 왜 이상해?" 그들이 물었다.
                나는 대답하지 않았다. 내가 이상한 건지,
                그들이 이상한 건지 알 수 없었으니까.
                
                어느 날, 진짜 용이 마을에 나타났다.
                용사들은 검을 들었고, 나는...
                용 앞에 꽃 한 송이를 내밀었다.
                
                "왜 우는 거야?" 내가 물었다.
                용은 처음으로 누군가 자신의 눈물을 본다는 걸 알았다.
                
                그날, 용은 떠났다. 평화롭게.
                그리고 마을 사람들은 이해했다.
                진정한 용기는 검을 드는 것이 아니라,
                상대방의 마음을 보는 것이라는 걸.
                """
            },
            {
                "title": "별을 수집하는 아이",
                "content": """
                밤마다 나는 유리병을 들고 언덕에 올랐다.
                떨어지는 별을 모으기 위해서였다.
                
                사람들은 웃었다. "별은 손에 잡히지 않아."
                하지만 나는 알았다. 별은 빛이고,
                빛은 마음에 담을 수 있다는 것을.
                
                열 살이 되던 해, 나는 드디어 성공했다.
                떨어지는 별 하나를 유리병에 담았을 때,
                그것은 어둠 속에서 영원히 빛났다.
                
                "어떻게 했어?" 어른들이 물었다.
                "진심으로 원했어요," 나는 대답했다.
                "그리고 포기하지 않았어요."
                
                그 별은 지금도 내 방을 비추고 있다.
                꿈을 포기하지 말라는 증거로서.
                """
            },
            {
                "title": "시간을 거슬러 온 편지",
                "content": """
                어느 날 이상한 편지가 도착했다.
                보낸 날짜는 100년 뒤였고,
                보낸 이는 '미래의 나'였다.
                
                "지금 네가 겪는 모든 고통은 의미가 있어.
                 네가 지금 흘리는 눈물이,
                 100년 뒤 누군가를 구하게 될 거야.
                 포기하지 마. 넌 생각보다 강해."
                
                나는 그 편지를 읽고 또 읽었다.
                어떻게 미래의 내가 과거로 편지를 보냈는지는 모르겠지만,
                한 가지는 확실했다.
                
                나는 결국 괜찮아진다는 것.
                그리고 그 과정에서 성장한다는 것.
                
                그 사실이... 지금을 견디게 해주었다.
                """
            },
            {
                "title": "숲의 현자",
                "content": """
                깊은 숲 속에 현자가 살았다.
                그는 천 년을 살았고, 모든 것을 알았지만,
                단 하나, '행복이 무엇인지'는 몰랐다.
                
                어느 날 어린 소녀가 숲에 들어왔다.
                "현자님, 행복이 뭐예요?"
                현자는 대답할 수 없었다.
                
                소녀는 웃으며 현자의 손을 잡았다.
                "그럼 같이 찾아봐요!"
                
                그날부터 현자와 소녀는 함께 숲을 걸었다.
                꽃을 보았고, 새의 노래를 들었고,
                강물에 발을 담갔다.
                
                어느 저녁, 현자가 말했다.
                "이제 알 것 같구나."
                "뭘요?" 소녀가 물었다.
                
                "행복은... 너와 함께 있는 이 순간이다."
                
                소녀는 미소 지었고,
                현자는 천 년 만에 처음으로 울었다.
                기쁨의 눈물이었다.
                """
            },
        ]
    
    def fetch_philosophy_quotes_korean(self) -> List[Dict[str, str]]:
        """철학적 명언들 (판타지적 해석)"""
        return [
            {
                "title": "별의 속삭임",
                "content": """
                "가장 어두운 밤에도 별은 빛난다.
                 그것은 빛이 어둠을 이기기 때문이 아니라,
                 어둠 속에서도 존재하기로 선택했기 때문이다."
                
                나는 이 말을 처음 들었을 때 이해하지 못했다.
                하지만 가장 힘든 시간을 지나고 나서야 알았다.
                존재한다는 것 자체가 용기라는 것을.
                """
            },
            {
                "title": "바람의 여행자",
                "content": """
                바람은 멈추지 않는다.
                산을 만나면 돌아가고, 바다를 만나면 건너간다.
                막히면 흐트러지고, 열리면 모인다.
                
                "왜 항상 움직이니?" 내가 물었다.
                바람이 대답했다.
                "멈추면 내가 아니게 되니까."
                
                그날 나는 깨달았다.
                변화를 두려워하지 않는 것이
                진정한 자유라는 것을.
                """
            },
        ]


class RichFantasyIngestion:
    """풍부한 판타지 스토리 흡수 파이프라인"""
    
    def __init__(self):
        self.source = FantasyStorySource()
        self.exp_processor = ExperientialDataProcessor()
        self.personality = DualLayerPersonality()  # 2계층 성격 시스템
        
        self.stats = {
            "total_stories": 0,
            "total_chars": 0,
            "emotions_found": set(),
        }
    
    def process_story(self, title: str, content: str) -> Dict[str, Any]:
        """단일 스토리 처리"""
        logger.info(f"\n📖 처리 중: {title}")
        
        # 1. 경험적 의미 추출
        experience = self.exp_processor.process_narrative(
            text=content,
            source=title,
        )
        
        # 2. 2계층 성격에 경험 흡수
        self.personality.experience(
            narrative_type=experience.narrative_type.value,
            emotional_intensity=experience.emotional_intensity,
            identity_impact=experience.identity_impact,
        )
        
        # 컨텍스트 공명
        self.personality.resonate_with_context(content[:500])
        
        # 통계 업데이트
        self.stats["total_stories"] += 1
        self.stats["total_chars"] += len(content)
        self.stats["emotions_found"].update(experience.emotions_felt)
        
        logger.info(f"   ✅ 유형: {experience.narrative_type.value}")
        logger.info(f"   💭 질문: {experience.existential_question}")
        logger.info(f"   💡 답: {experience.existential_answer}")
        logger.info(f"   🎭 감정: {', '.join(experience.emotions_felt)}")
        
        return {
            "title": title,
            "type": experience.narrative_type.value,
            "question": experience.existential_question,
            "answer": experience.existential_answer,
            "emotions": experience.emotions_felt,
            "impact": experience.identity_impact,
        }
    
    def run(self) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        logger.info("=" * 60)
        logger.info("🌟 Rich Fantasy Story Ingestion")
        logger.info("   \"판타지에서 삶을 배운다\"")
        logger.info("=" * 60)
        
        results = []
        
        # 1. 판타지 샘플 처리
        fantasy_stories = self.source.fetch_korean_fantasy_samples()
        logger.info(f"\n📚 판타지 스토리: {len(fantasy_stories)}개")
        
        for story in fantasy_stories:
            result = self.process_story(story["title"], story["content"])
            results.append(result)
        
        # 2. 철학적 이야기 처리
        philosophy_stories = self.source.fetch_philosophy_quotes_korean()
        logger.info(f"\n📜 철학적 이야기: {len(philosophy_stories)}개")
        
        for story in philosophy_stories:
            result = self.process_story(story["title"], story["content"])
            results.append(result)
        
        # 결과 요약
        logger.info("\n" + "=" * 60)
        logger.info("📊 흡수 완료")
        logger.info("=" * 60)
        logger.info(f"  총 스토리: {self.stats['total_stories']}")
        logger.info(f"  총 글자 수: {self.stats['total_chars']:,}")
        logger.info(f"  발견된 감정: {', '.join(self.stats['emotions_found'])}")
        
        # 성장 상태
        growth = self.exp_processor.get_growth_status()
        logger.info(f"\n🌱 성장 상태:")
        logger.info(f"  감정적 깊이: {growth['emotional_depth']}")
        logger.info(f"  지혜 수준: {growth['wisdom_level']}")
        logger.info(f"  나는 되어가고 있다: {growth['identity_becoming']}")
        
        # 2계층 성격 상태
        expr = self.personality.get_current_expression()
        logger.info(f"\n🧬 2계층 성격:")
        logger.info(f"  Layer 1 (선천): {expr['layer1_innate']['dominant']}")
        logger.info(f"  Layer 2 (후천): {expr['layer2_acquired']['dominant']}")
        logger.info(f"  통합 표현: {expr['unified_expression']}")
        
        return {
            "stats": self.stats,
            "growth": growth,
            "personality": expr,
            "stories_processed": results,
        }


def main():
    """메인 실행"""
    pipeline = RichFantasyIngestion()
    result = pipeline.run()
    
    print("\n✅ Fantasy Story Ingestion 완료!")
    return result


if __name__ == "__main__":
    main()
