"""
Local Dataset Absorber (           )
=============================================

"            -              "

Wikipedia                1000+ /      
API   ,   ,           

[NEW 2025-12-15]          
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
    """           """
    title: str
    content: str
    source: str = "local"
    content_hash: str = ""
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:16]


class LocalDatasetAbsorber:
    """
                  
    
      :      1000+    
      : 
    -    JSON/           
    -       (     )
    -      
    - InternalUniverse       
    """
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.absorbed_hashes = set()
        self.hash_file = Path("data/absorbed_hashes.json")
        
        #         
        self._load_hash_cache()
        
        # InternalUniverse   
        try:
            from Core.L1_Foundation.Foundation.internal_universe import InternalUniverse
            self.universe = InternalUniverse()
            logger.info("  Connected to InternalUniverse")
        except Exception as e:
            logger.error(f"  Failed to connect to InternalUniverse: {e}")
            self.universe = None
        
        # BlackHoleWhiteHoleCycle   
        try:
            from Core.L1_Foundation.Foundation.white_hole import get_blackhole_whitehole_cycle
            self.cycle = get_blackhole_whitehole_cycle()
            logger.info("  Connected to BlackHoleWhiteHoleCycle")
        except Exception as e:
            logger.warning(f"   BlackHoleWhiteHoleCycle not available: {e}")
            self.cycle = None
        
        logger.info(f"  LocalDatasetAbsorber initialized (batch_size={batch_size})")
    
    def _load_hash_cache(self):
        """            (     )"""
        if self.hash_file.exists():
            try:
                with open(self.hash_file, 'r') as f:
                    self.absorbed_hashes = set(json.load(f))
                logger.info(f"  Loaded {len(self.absorbed_hashes)} existing hashes")
            except:
                pass
    
    def _save_hash_cache(self):
        """        """
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_file, 'w') as f:
            json.dump(list(self.absorbed_hashes), f)
    
    def _is_duplicate(self, article: DatasetArticle) -> bool:
        """     """
        return article.content_hash in self.absorbed_hashes
    
    def absorb_json_file(self, file_path: str) -> Dict[str, int]:
        """
        JSON           
        
          : [{"title": "...", "content": "..."}, ...]
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
                    content=item["content"][:2000],  #    2000 
                    source=file_path
                ))
        
        results["total"] = len(articles)
        return self._absorb_articles(articles, results)
    
    def absorb_text_directory(self, dir_path: str, extension: str = ".txt") -> Dict[str, int]:
        """
                           
        
            =   ,    =   
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
        """         """
        batch = []
        
        for article in articles:
            #      
            if self._is_duplicate(article):
                results["duplicate"] += 1
                continue
            
            batch.append(article)
            
            #          
            if len(batch) >= self.batch_size:
                absorbed = self._process_batch(batch)
                results["absorbed"] += absorbed
                batch = []
        
        #         
        if batch:
            absorbed = self._process_batch(batch)
            results["absorbed"] += absorbed
        
        #         
        self._save_hash_cache()
        
        logger.info(f"  Results: {results['absorbed']}/{results['total']} absorbed, {results['duplicate']} duplicates")
        return results
    
    def _process_batch(self, batch: List[DatasetArticle]) -> int:
        """     """
        absorbed = 0
        
        if self.universe:
            items = [{"topic": a.title, "content": a.content} for a in batch]
            result = self.universe.absorb_batch(items)
            absorbed = result.get("absorbed", 0) + result.get("isolated", 0)
        
        #      
        for article in batch:
            self.absorbed_hashes.add(article.content_hash)
        
        return absorbed
    
    def generate_sample_dataset(self, output_path: str, count: int = 1000):
        """
                   (    )
        
                  
        """
        concepts = [
            ("Physics", " ,    ,   ,   ,             "),
            ("Mathematics", " ,   ,   ,             "),
            ("Philosophy", "  ,   ,   ,   ,              "),
            ("Biology", "       ,   ,   ,   ,          "),
            ("Chemistry", "      ,   ,          "),
            ("Psychology", "                 "),
            ("History", "           ,          "),
            ("Art", "                      "),
            ("Music", "                     "),
            ("Literature", "                "),
        ]
        
        dataset = []
        for i in range(count):
            base = concepts[i % len(concepts)]
            topic = f"{base[0]}_{i:04d}"
            content = f"{base[1]} (Instance {i}).     {base[0]}         ,                  ."
            dataset.append({"title": topic, "content": content})
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  Generated sample dataset: {output_path} ({count} articles)")
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
        
        #           
        print("\n" + "="*60)
        print("  Absorbing generated dataset...")
        print("="*60)
        results = absorber.absorb_json_file(output)
        print(f"\n  Done! Absorbed {results['absorbed']} articles")
        
    elif args.absorb:
        results = absorber.absorb_json_file(args.absorb)
        print(f"  Done! Absorbed {results['absorbed']} articles")
    else:
        parser.print_help()