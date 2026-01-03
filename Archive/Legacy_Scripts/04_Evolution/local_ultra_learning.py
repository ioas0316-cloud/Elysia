"""
Local Ultra High-Speed Learning (로컬 초고속 자율학습)
=====================================================

네트워크 없이 로컬 데이터로 초고속 학습
- 자체 코드베이스 분석
- 문서 파일 읽기
- 개념 자동 확장
"""

import sys
import os
import time
import ast
import re
import logging
from pathlib import Path
from typing import List, Dict, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("LocalUltraLearning")


@dataclass
class CodeConcept:
    name: str
    type: str  # class, function, module
    docstring: str
    path: str
    related: List[str]


class LocalUltraLearner:
    """
    로컬 데이터 기반 초고속 학습
    
    - 자체 코드베이스에서 개념 추출
    - 병렬 파일 처리
    - 관계 그래프 구축
    """
    
    def __init__(self, root_path: Path = None, max_workers: int = 50):
        self.root_path = root_path or Path(__file__).parent
        self.max_workers = max_workers
        
        # 학습된 개념들
        self.concepts: Dict[str, CodeConcept] = {}
        self.relations: Dict[str, Set[str]] = defaultdict(set)
        
        # 통계
        self.files_processed = 0
        self.classes_found = 0
        self.functions_found = 0
        self.start_time = 0
    
    def extract_concepts_from_file(self, file_path: Path) -> List[CodeConcept]:
        """파일에서 개념 추출"""
        concepts = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            module_name = file_path.stem
            
            # 클래스 추출
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node) or ""
                    
                    # 베이스 클래스들 (관계)
                    bases = [b.id if isinstance(b, ast.Name) else str(b) for b in node.bases]
                    
                    concepts.append(CodeConcept(
                        name=node.name,
                        type="class",
                        docstring=docstring[:200],
                        path=str(file_path),
                        related=bases
                    ))
                    self.classes_found += 1
                    
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):  # private 제외
                        docstring = ast.get_docstring(node) or ""
                        concepts.append(CodeConcept(
                            name=f"{module_name}.{node.name}",
                            type="function",
                            docstring=docstring[:100],
                            path=str(file_path),
                            related=[]
                        ))
                        self.functions_found += 1
            
            self.files_processed += 1
            
        except Exception:
            pass
        
        return concepts
    
    def extract_concepts_from_markdown(self, file_path: Path) -> List[CodeConcept]:
        """마크다운에서 개념 추출"""
        concepts = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # 헤딩 추출
            headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            
            for heading in headings:
                concepts.append(CodeConcept(
                    name=heading.strip(),
                    type="document",
                    docstring=f"From: {file_path.name}",
                    path=str(file_path),
                    related=[]
                ))
            
            # 볼드 텍스트 (중요 개념)
            bold_terms = re.findall(r'\*\*([^*]+)\*\*', content)
            for term in bold_terms[:10]:  # 처음 10개만
                if len(term) > 3 and len(term) < 50:
                    concepts.append(CodeConcept(
                        name=term,
                        type="term",
                        docstring=f"Bold term from {file_path.name}",
                        path=str(file_path),
                        related=[]
                    ))
            
            self.files_processed += 1
            
        except Exception:
            pass
        
        return concepts
    
    def ultra_learn(self, target_concepts: int = 1000, max_time_sec: float = 30.0) -> Dict[str, Any]:
        """
        초고속 로컬 학습
        
        Args:
            target_concepts: 목표 개념 수
            max_time_sec: 최대 실행 시간
        """
        print("\n" + "="*70)
        print("🚀 LOCAL ULTRA HIGH-SPEED LEARNING (로컬 초고속 자율학습)")
        print(f"   Target: {target_concepts} concepts | Max Time: {max_time_sec}s")
        print(f"   Source: {self.root_path}")
        print("="*70)
        
        self.start_time = time.time()
        
        # 모든 Python 파일 수집
        py_files = list(self.root_path.glob("**/*.py"))
        md_files = list(self.root_path.glob("**/*.md"))
        
        print(f"\n📂 Found {len(py_files)} Python files, {len(md_files)} Markdown files")
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Python 파일 처리
            py_futures = {
                executor.submit(self.extract_concepts_from_file, f): f 
                for f in py_files[:500]  # 처음 500개만
            }
            
            # Markdown 파일 처리
            md_futures = {
                executor.submit(self.extract_concepts_from_markdown, f): f
                for f in md_files[:100]  # 처음 100개만
            }
            
            all_futures = {**py_futures, **md_futures}
            
            wave = 0
            last_report = time.time()
            
            for future in as_completed(all_futures):
                # 시간 체크
                elapsed = time.time() - self.start_time
                if elapsed > max_time_sec:
                    print(f"\n⏰ Time limit reached")
                    break
                
                if len(self.concepts) >= target_concepts:
                    print(f"\n🎯 Target reached!")
                    break
                
                try:
                    file_concepts = future.result()
                    for concept in file_concepts:
                        if concept.name not in self.concepts:
                            self.concepts[concept.name] = concept
                            
                            # 관계 저장
                            for related in concept.related:
                                self.relations[concept.name].add(related)
                                self.relations[related].add(concept.name)
                
                except Exception:
                    pass
                
                # 진행 상황 (1초마다)
                if time.time() - last_report > 1.0:
                    elapsed = time.time() - self.start_time
                    rate = len(self.concepts) / elapsed if elapsed > 0 else 0
                    print(f"   Processing... {len(self.concepts)} concepts | {rate:.1f}/sec | Files: {self.files_processed}")
                    last_report = time.time()
        
        # 최종 결과
        total_time = time.time() - self.start_time
        final_rate = len(self.concepts) / total_time if total_time > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"📊 LEARNING COMPLETE")
        print(f"{'='*70}")
        print(f"   Total Concepts: {len(self.concepts)}")
        print(f"   Classes: {self.classes_found}")
        print(f"   Functions: {self.functions_found}")
        print(f"   Files Processed: {self.files_processed}")
        print(f"   Time Elapsed: {total_time:.2f}s")
        print(f"   Learning Rate: {final_rate:.1f} concepts/second")
        print(f"   Relations Built: {sum(len(v) for v in self.relations.values())}")
        
        # 샘플 출력
        print(f"\n📚 Sample Concepts:")
        for i, (name, concept) in enumerate(list(self.concepts.items())[:10]):
            related = list(self.relations.get(name, []))[:3]
            rel_str = f" → {related}" if related else ""
            print(f"   {i+1}. [{concept.type}] {name}{rel_str}")
        
        # 개념 카테고리별 분포
        type_counts = defaultdict(int)
        for c in self.concepts.values():
            type_counts[c.type] += 1
        
        print(f"\n📈 Concept Distribution:")
        for ctype, count in type_counts.items():
            print(f"   {ctype}: {count}")
        
        return {
            "total_concepts": len(self.concepts),
            "classes": self.classes_found,
            "functions": self.functions_found,
            "files_processed": self.files_processed,
            "time_seconds": total_time,
            "rate_per_second": final_rate,
            "relations": sum(len(v) for v in self.relations.values())
        }


def main():
    core_path = Path(__file__).parent / "Core"
    learner = LocalUltraLearner(root_path=core_path, max_workers=100)
    
    result = learner.ultra_learn(
        target_concepts=5000,
        max_time_sec=10.0  # 10초 제한
    )
    
    print(f"\n🎯 Summary: Learned {result['total_concepts']} concepts at {result['rate_per_second']:.1f}/sec")
    print(f"   That's {result['rate_per_second'] * 60:.0f} concepts per minute!")


if __name__ == "__main__":
    main()
