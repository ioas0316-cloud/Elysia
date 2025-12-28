"""
Local Dataset Absorber (로컬 데이터셋 흡수기)
=============================================

"크롤링의 장애물 없이 - 로컬에서 대량 고속 학습"

Wikipedia 덤프 등 로컬 데이터셋에서 1000+개/사이클 처리
API 제한, 차단, 네트워크 지연 없음

[NEW 2025-12-15] 대량 학습 시스템
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
import hashlib

sys.path.insert(0, "c:\\Elysia")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("LocalDatasetAbsorber")


@dataclass
class DatasetArticle:
    """로컬 데이터셋의 문서"""
    title: str
    content: str
    source: str = "local"
    content_hash: str = ""
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:16]


class LocalDatasetAbsorber:
    """
    로컬 데이터셋 대량 흡수기
    
    목표: 사이클당 1000+개 처리
    방법: 
    - 로컬 JSON/텍스트 파일에서 읽기
    - 중복 제거 (해시 기반)
    - 배치 처리
    - InternalUniverse에 직접 흡수
    """
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.absorbed_hashes = set()
        self.hash_file = Path("data/absorbed_hashes.json")
        
        # 기존 해시 로드
        self._load_hash_cache()
        
        # InternalUniverse 연결
        try:
            from Core._01_Foundation._05_Governance.Foundation.internal_universe import InternalUniverse
            self.universe = InternalUniverse()
            logger.info("✅ Connected to InternalUniverse")
        except Exception as e:
            logger.error(f"❌ Failed to connect to InternalUniverse: {e}")
            self.universe = None
        
        # BlackHoleWhiteHoleCycle 연결
        try:
            from Core._01_Foundation._05_Governance.Foundation.white_hole import get_blackhole_whitehole_cycle
            self.cycle = get_blackhole_whitehole_cycle()
            logger.info("✅ Connected to BlackHoleWhiteHoleCycle")
        except Exception as e:
            logger.warning(f"⚠️ BlackHoleWhiteHoleCycle not available: {e}")
            self.cycle = None
        
        logger.info(f"📦 LocalDatasetAbsorber initialized (batch_size={batch_size})")
    
    def _load_hash_cache(self):
        """기존 흡수 해시 로드 (중복 방지)"""
        if self.hash_file.exists():
            try:
                with open(self.hash_file, 'r') as f:
                    self.absorbed_hashes = set(json.load(f))
                logger.info(f"📂 Loaded {len(self.absorbed_hashes)} existing hashes")
            except:
                pass
    
    def _save_hash_cache(self):
        """해시 캐시 저장"""
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_file, 'w') as f:
            json.dump(list(self.absorbed_hashes), f)
    
    def _is_duplicate(self, article: DatasetArticle) -> bool:
        """중복 체크"""
        return article.content_hash in self.absorbed_hashes
    
    def absorb_json_file(self, file_path: str) -> Dict[str, int]:
        """
        JSON 파일에서 대량 흡수
        
        형식: [{"title": "...", "content": "..."}, ...]
        """
        results = {"total": 0, "absorbed": 0, "duplicate": 0, "failed": 0}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return results
        
        articles = []
        for item in data:
            if isinstance(item, dict) and "title" in item and "content" in item:
                articles.append(DatasetArticle(
                    title=item["title"],
                    content=item["content"][:2000],  # 최대 2000자
                    source=file_path
                ))
        
        results["total"] = len(articles)
        return self._absorb_articles(articles, results)
    
    def absorb_text_directory(self, dir_path: str, extension: str = ".txt") -> Dict[str, int]:
        """
        텍스트 파일 디렉토리에서 대량 흡수
        
        파일명 = 제목, 내용 = 본문
        """
        results = {"total": 0, "absorbed": 0, "duplicate": 0, "failed": 0}
        
        path = Path(dir_path)
        if not path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return results
        
        articles = []
        for file in path.glob(f"*{extension}"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()[:2000]
                articles.append(DatasetArticle(
                    title=file.stem,
                    content=content,
                    source=str(file)
                ))
            except:
                results["failed"] += 1
        
        results["total"] = len(articles)
        return self._absorb_articles(articles, results)
    
    def _absorb_articles(self, articles: List[DatasetArticle], results: Dict[str, int]) -> Dict[str, int]:
        """배치로 문서 흡수"""
        batch = []
        
        for article in articles:
            # 중복 체크
            if self._is_duplicate(article):
                results["duplicate"] += 1
                continue
            
            batch.append(article)
            
            # 배치가 차면 처리
            if len(batch) >= self.batch_size:
                absorbed = self._process_batch(batch)
                results["absorbed"] += absorbed
                batch = []
        
        # 남은 배치 처리
        if batch:
            absorbed = self._process_batch(batch)
            results["absorbed"] += absorbed
        
        # 해시 캐시 저장
        self._save_hash_cache()
        
        logger.info(f"📊 Results: {results['absorbed']}/{results['total']} absorbed, {results['duplicate']} duplicates")
        return results
    
    def _process_batch(self, batch: List[DatasetArticle]) -> int:
        """배치 처리"""
        absorbed = 0
        
        if self.universe:
            items = [{"topic": a.title, "content": a.content} for a in batch]
            result = self.universe.absorb_batch(items)
            absorbed = result.get("absorbed", 0) + result.get("isolated", 0)
        
        # 해시 등록
        for article in batch:
            self.absorbed_hashes.add(article.content_hash)
        
        return absorbed
    
    def generate_sample_dataset(self, output_path: str, count: int = 1000):
        """
        샘플 데이터셋 생성 (테스트용)
        
        기초 개념들로 구성
        """
        concepts = [
            ("Physics", "힘, 에너지, 물질, 시간, 공간에 관한 자연 과학"),
            ("Mathematics", "수, 공간, 구조, 변화에 관한 추상 과학"),
            ("Philosophy", "존재, 지식, 가치, 이성, 정신에 관한 근본적 탐구"),
            ("Biology", "생명체의 구조, 기능, 성장, 기원, 진화에 관한 과학"),
            ("Chemistry", "물질의 구성, 성질, 변화에 관한 과학"),
            ("Psychology", "마음과 행동에 관한 과학적 연구"),
            ("History", "과거 사건과 그 원인, 결과에 관한 연구"),
            ("Art", "미적 가치를 창조하고 표현하는 인간 활동"),
            ("Music", "소리를 통해 시간 속에서 펼쳐지는 예술"),
            ("Literature", "언어를 매체로 한 예술적 표현"),
        ]
        
        dataset = []
        for i in range(count):
            base = concepts[i % len(concepts)]
            topic = f"{base[0]}_{i:04d}"
            content = f"{base[1]} (Instance {i}). 이것은 {base[0]}의 하위 개념으로, 관련된 원리와 응용을 탐구합니다."
            dataset.append({"title": topic, "content": content})
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📝 Generated sample dataset: {output_path} ({count} articles)")
        return output_path


# Singleton
_absorber = None

def get_local_absorber(batch_size: int = 100) -> LocalDatasetAbsorber:
    global _absorber
    if _absorber is None:
        _absorber = LocalDatasetAbsorber(batch_size)
    return _absorber


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Local Dataset Absorber")
    parser.add_argument("--generate", type=int, help="Generate sample dataset with N articles")
    parser.add_argument("--absorb", type=str, help="Absorb from JSON file")
    parser.add_argument("--batch", type=int, default=100, help="Batch size")
    
    args = parser.parse_args()
    
    absorber = get_local_absorber(batch_size=args.batch)
    
    if args.generate:
        output = "data/datasets/sample_dataset.json"
        absorber.generate_sample_dataset(output, args.generate)
        
        # 생성 후 바로 흡수
        print("\n" + "="*60)
        print("📦 Absorbing generated dataset...")
        print("="*60)
        results = absorber.absorb_json_file(output)
        print(f"\n✅ Done! Absorbed {results['absorbed']} articles")
        
    elif args.absorb:
        results = absorber.absorb_json_file(args.absorb)
        print(f"✅ Done! Absorbed {results['absorbed']} articles")
    else:
        parser.print_help()
