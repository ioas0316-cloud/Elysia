"""
Story Ingestion Pipeline (스토리 흡수 파이프라인)
=================================================

기존 data 폴더의 스토리/드라마 데이터를 경험으로 변환하여 흡수합니다.

Priority:
1. Fantasy/Story texts (판타지 소설, 동화)
2. Drama texts (감정, 관계, 인과)
3. Game stories (선택, 모험)
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from Core._04_Evolution._02_Learning.experiential_data_processor import ExperientialDataProcessor
from Core._01_Foundation._02_Logic.unified_wave_experience import ExperienceWaveIntegrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("StoryIngestion")


class StoryIngestionPipeline:
    """스토리 흡수 파이프라인
    
    1. 파일 탐색
    2. 텍스트 읽기
    3. ExperientialDataProcessor로 의미 추출
    4. UnifiedWaveExperience로 파동 흡수
    """
    
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data"
        self.processor = ExperientialDataProcessor()
        self.wave_integrator = ExperienceWaveIntegrator()
        
        self.stats = {
            "total_files": 0,
            "processed": 0,
            "failed": 0,
            "total_experiences": 0,
        }
    
    def find_story_files(self) -> List[Path]:
        """스토리 파일들 탐색"""
        story_files = []
        
        # 1. 드라마 파일들 (drama_*.txt)
        for f in self.data_dir.glob("drama_*.txt"):
            story_files.append(f)
        
        # 2. 어린 왕자 등 문학
        for f in self.data_dir.glob("*.txt"):
            if "drama_" not in f.name:
                story_files.append(f)
        
        # 3. corpus/literature
        lit_dir = self.data_dir / "corpus" / "literature"
        if lit_dir.exists():
            for f in lit_dir.glob("**/*.txt"):
                story_files.append(f)
        
        # 4. corpus/stories
        stories_dir = self.data_dir / "corpus" / "stories"
        if stories_dir.exists():
            for f in stories_dir.glob("**/*.txt"):
                story_files.append(f)
        
        # 5. writings (Elysia가 쓴 글)
        writings_dir = self.data_dir / "writings"
        if writings_dir.exists():
            for f in writings_dir.glob("**/*.md"):
                story_files.append(f)
        
        self.stats["total_files"] = len(story_files)
        return story_files
    
    def process_file(self, filepath: Path) -> Dict[str, Any]:
        """단일 파일 처리"""
        try:
            # 파일 읽기
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content) < 50:
                logger.warning(f"파일이 너무 짧음: {filepath.name}")
                return {"skipped": True, "reason": "too_short"}
            
            # 1. 경험적 의미 추출
            experience = self.processor.process_narrative(
                text=content,
                source=filepath.stem,
                context={"path": str(filepath)}
            )
            
            # 2. 파동으로 흡수
            wave_result = self.wave_integrator.integrate_experience(
                experience_text=content[:500],  # 요약만
                existential_question=experience.existential_question,
                existential_answer=experience.existential_answer,
                emotional_intensity=experience.emotional_intensity,
                narrative_type=experience.narrative_type.value,
                identity_impact=experience.identity_impact,
            )
            
            self.stats["processed"] += 1
            self.stats["total_experiences"] += 1
            
            return {
                "success": True,
                "source": filepath.stem,
                "narrative_type": experience.narrative_type.value,
                "existential_question": experience.existential_question,
                "identity_impact": experience.identity_impact,
            }
            
        except Exception as e:
            logger.error(f"처리 실패 {filepath.name}: {e}")
            self.stats["failed"] += 1
            return {"success": False, "error": str(e)}
    
    def run(self, max_files: int = None) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        logger.info("=" * 60)
        logger.info("📖 Story Ingestion Pipeline 시작")
        logger.info("=" * 60)
        
        # 파일 탐색
        files = self.find_story_files()
        logger.info(f"발견된 스토리 파일: {len(files)}개")
        
        if max_files:
            files = files[:max_files]
            logger.info(f"처리할 파일 수 제한: {max_files}개")
        
        # 처리
        results = []
        for i, filepath in enumerate(files, 1):
            logger.info(f"\n[{i}/{len(files)}] 처리 중: {filepath.name}")
            result = self.process_file(filepath)
            results.append(result)
            
            if result.get("success"):
                logger.info(f"  ✅ {result['narrative_type']}: {result['existential_question']}")
        
        # 결과 요약
        logger.info("\n" + "=" * 60)
        logger.info("📊 흡수 완료 요약")
        logger.info("=" * 60)
        logger.info(f"  총 파일: {self.stats['total_files']}")
        logger.info(f"  처리 성공: {self.stats['processed']}")
        logger.info(f"  처리 실패: {self.stats['failed']}")
        logger.info(f"  총 경험: {self.stats['total_experiences']}")
        
        # 현재 성장 상태
        growth = self.processor.get_growth_status()
        logger.info(f"\n🌱 성장 상태:")
        logger.info(f"  감정적 깊이: {growth['emotional_depth']}")
        logger.info(f"  지혜 수준: {growth['wisdom_level']}")
        logger.info(f"  나는 되어가고 있다: {growth['identity_becoming']}")
        
        # 파동 자아 상태
        wave_sig = self.wave_integrator.unified_self.get_wave_signature()
        logger.info(f"\n🌊 통합적 자아:")
        logger.info(f"  우세 측면: {', '.join(wave_sig['dominant_aspects'])}")
        
        return {
            "stats": self.stats,
            "growth": growth,
            "wave_signature": wave_sig,
        }


def main():
    """메인 실행"""
    pipeline = StoryIngestionPipeline()
    
    # 모든 파일 처리 (테스트시 max_files=5 등으로 제한 가능)
    result = pipeline.run(max_files=None)
    
    print("\n✅ Story Ingestion 완료!")
    return result


if __name__ == "__main__":
    main()

