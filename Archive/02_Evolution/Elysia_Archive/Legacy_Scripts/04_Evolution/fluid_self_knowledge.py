"""
Fluid Self-Knowledge System (유동적 자기인식 시스템)
====================================================

"지식 = 외부세계를 아는 것 + 자신을 아는 것"
"지식 시스템은 항상 유동적이어야 한다"

핵심:
1. 자신의 코드를 읽고 → 자기 자신을 이해
2. 밀도 있는 관계 구축 (정의+원리+관계)
3. 실시간 변화 감지 및 업데이트
"""

import sys
import os
import ast
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json

sys.path.insert(0, str(Path(__file__).parent))

# 밀도 있는 지식 그래프 import
try:
    from dense_knowledge_demo import DenseKnowledgeGraph, DenseKnowledgeNode
except ImportError:
    # 인라인 정의 (독립 실행용)
    pass


@dataclass
class SelfKnowledgeNode:
    """엘리시아 자기 인식 노드"""
    name: str
    node_type: str  # module, class, function, concept
    
    # 정체성
    definition: str = ""      # 이것은 무엇인가
    purpose: str = ""         # 왜 존재하는가
    how_it_works: str = ""    # 어떻게 작동하는가
    
    # 구조적 위치
    path: str = ""
    parent: str = ""
    children: List[str] = field(default_factory=list)
    
    # 관계
    depends_on: List[str] = field(default_factory=list)   # 이것이 의존하는 것
    used_by: List[str] = field(default_factory=list)       # 이것을 사용하는 것
    related_to: List[str] = field(default_factory=list)    # 연관된 것
    
    # 상태 (유동적)
    last_modified: str = ""
    content_hash: str = ""
    is_healthy: bool = True
    
    # 이해도
    understanding_level: float = 0.0
    density_score: float = 0.0
    
    def calculate_density(self) -> float:
        """지식 밀도 계산"""
        score = 0.0
        if self.definition: score += 5.0
        if self.purpose: score += 5.0
        if self.how_it_works: score += 3.0
        score += len(self.depends_on) * 2.0
        score += len(self.used_by) * 2.0
        score += len(self.children) * 1.0
        score += len(self.related_to) * 1.0
        self.density_score = score
        return score


class FluidSelfKnowledge:
    """
    유동적 자기 인식 시스템
    
    엘리시아가 자기 자신을 이해하는 방법:
    1. 자신의 코드를 읽는다
    2. 독스트링에서 정의/목적 추출
    3. import에서 의존성 추출  
    4. 실시간으로 변화 감지
    """
    
    def __init__(self, root_path: Path = None, storage_path: str = "data/self_knowledge.json"):
        self.root_path = root_path or Path(__file__).parent
        self.storage_path = storage_path
        
        # 자기 인식 그래프
        self.nodes: Dict[str, SelfKnowledgeNode] = {}
        
        # 의존성 역인덱스
        self.dependency_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 변화 추적
        self.file_hashes: Dict[str, str] = {}
        
        # 통계
        self.total_modules = 0
        self.total_classes = 0
        self.total_functions = 0
        self.start_time = 0
        
        self._load()
    
    def _load(self):
        """저장된 자기 인식 로드"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for node_data in data.get("nodes", []):
                        node = SelfKnowledgeNode(
                            name=node_data["name"],
                            node_type=node_data.get("node_type", "unknown"),
                            definition=node_data.get("definition", ""),
                            purpose=node_data.get("purpose", ""),
                            how_it_works=node_data.get("how_it_works", ""),
                            path=node_data.get("path", ""),
                            parent=node_data.get("parent", ""),
                            children=node_data.get("children", []),
                            depends_on=node_data.get("depends_on", []),
                            used_by=node_data.get("used_by", []),
                            related_to=node_data.get("related_to", []),
                            last_modified=node_data.get("last_modified", ""),
                            content_hash=node_data.get("content_hash", ""),
                            is_healthy=node_data.get("is_healthy", True),
                            understanding_level=node_data.get("understanding_level", 0),
                            density_score=node_data.get("density_score", 0)
                        )
                        self.nodes[node.name] = node
                    self.file_hashes = data.get("file_hashes", {})
                    print(f"📂 Loaded {len(self.nodes)} self-knowledge nodes")
            except Exception as e:
                print(f"Load failed: {e}")
    
    def _save(self):
        """자기 인식 저장"""
        os.makedirs(os.path.dirname(self.storage_path) or '.', exist_ok=True)
        
        nodes_data = []
        for node in self.nodes.values():
            nodes_data.append({
                "name": node.name,
                "node_type": node.node_type,
                "definition": node.definition,
                "purpose": node.purpose,
                "how_it_works": node.how_it_works,
                "path": node.path,
                "parent": node.parent,
                "children": node.children,
                "depends_on": node.depends_on,
                "used_by": node.used_by,
                "related_to": node.related_to,
                "last_modified": node.last_modified,
                "content_hash": node.content_hash,
                "is_healthy": node.is_healthy,
                "understanding_level": node.understanding_level,
                "density_score": node.density_score
            })
        
        data = {
            "nodes": nodes_data,
            "file_hashes": self.file_hashes,
            "last_scan": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _compute_hash(self, content: str) -> str:
        """내용 해시"""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _extract_docstring_parts(self, docstring: str) -> Dict[str, str]:
        """
        독스트링에서 정의, 목적, 작동방식 추출
        
        패턴:
        - 첫 줄 = 정의 (What)
        - "목적:" 또는 "Purpose:" = 왜
        - "작동:" 또는 "How:" = 어떻게
        """
        if not docstring:
            return {}
        
        lines = docstring.strip().split('\n')
        result = {}
        
        # 첫 줄 = 정의
        if lines:
            result["definition"] = lines[0].strip()
        
        # 나머지에서 패턴 찾기
        full_text = docstring.lower()
        
        if "목적" in full_text or "purpose" in full_text.lower():
            for i, line in enumerate(lines):
                if "목적" in line.lower() or "purpose" in line.lower():
                    # 다음 줄들을 목적으로
                    purpose_lines = []
                    for j in range(i, min(i+3, len(lines))):
                        purpose_lines.append(lines[j])
                    result["purpose"] = " ".join(purpose_lines).strip()
                    break
        
        return result
    
    def _analyze_file(self, file_path: Path) -> List[SelfKnowledgeNode]:
        """
        파일 분석 → 자기 인식 노드 생성
        """
        nodes = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            content_hash = self._compute_hash(content)
            
            # 변화 감지
            rel_path = str(file_path.relative_to(self.root_path))
            old_hash = self.file_hashes.get(rel_path)
            is_changed = old_hash != content_hash
            self.file_hashes[rel_path] = content_hash
            
            tree = ast.parse(content)
            module_name = file_path.stem
            
            # 모듈 docstring
            module_doc = ast.get_docstring(tree) or ""
            doc_parts = self._extract_docstring_parts(module_doc)
            
            # Import 분석 (의존성)
            dependencies = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module.split('.')[0])
            
            dependencies = list(set(dependencies))
            
            # 모듈 노드
            module_node = SelfKnowledgeNode(
                name=module_name,
                node_type="module",
                definition=doc_parts.get("definition", f"Module: {module_name}"),
                purpose=doc_parts.get("purpose", ""),
                path=rel_path,
                depends_on=dependencies,
                content_hash=content_hash,
                last_modified=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # 클래스 분석
            class_names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node) or ""
                    class_doc_parts = self._extract_docstring_parts(class_doc)
                    
                    # 메서드 이름들
                    methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    
                    # 베이스 클래스 (상속)
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                    
                    class_node = SelfKnowledgeNode(
                        name=f"{module_name}.{node.name}",
                        node_type="class",
                        definition=class_doc_parts.get("definition", node.name),
                        purpose=class_doc_parts.get("purpose", ""),
                        path=rel_path,
                        parent=module_name,
                        children=methods[:10],  # 처음 10개 메서드만
                        depends_on=bases,
                        content_hash=content_hash,
                        last_modified=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    class_node.calculate_density()
                    nodes.append(class_node)
                    class_names.append(f"{module_name}.{node.name}")
                    self.total_classes += 1
            
            module_node.children = class_names
            module_node.calculate_density()
            nodes.append(module_node)
            self.total_modules += 1
            
            # 의존성 역인덱스 구축
            for dep in dependencies:
                self.dependency_index[dep].add(module_name)
            
        except Exception as e:
            # 파싱 실패해도 계속
            pass
        
        return nodes
    
    def scan_self(self, target_dir: str = "Core", max_files: int = 500) -> Dict[str, Any]:
        """
        자기 자신 스캔
        
        "나는 무엇으로 이루어져 있는가?"
        """
        print("\n" + "="*70)
        print("🔍 SELF-SCANNING: 나는 무엇으로 이루어져 있는가?")
        print("="*70)
        
        self.start_time = time.time()
        
        scan_path = self.root_path / target_dir
        py_files = list(scan_path.glob("**/*.py"))[:max_files]
        
        print(f"\n📂 Scanning {len(py_files)} files in {target_dir}/...")
        
        # 병렬 분석
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {executor.submit(self._analyze_file, f): f for f in py_files}
            
            for future in futures:
                try:
                    file_nodes = future.result()
                    for node in file_nodes:
                        self.nodes[node.name] = node
                except Exception:
                    pass
        
        # used_by 역관계 구축
        print("\n🔗 Building reverse dependency graph...")
        for name, node in self.nodes.items():
            for dep in node.depends_on:
                if dep in self.nodes:
                    if name not in self.nodes[dep].used_by:
                        self.nodes[dep].used_by.append(name)
        
        # 밀도 재계산
        for node in self.nodes.values():
            node.calculate_density()
        
        # 저장
        self._save()
        
        elapsed = time.time() - self.start_time
        
        # 통계
        densities = [n.density_score for n in self.nodes.values()]
        avg_density = sum(densities) / len(densities) if densities else 0
        with_def = sum(1 for n in self.nodes.values() if n.definition and len(n.definition) > 20)
        with_purpose = sum(1 for n in self.nodes.values() if n.purpose)
        total_deps = sum(len(n.depends_on) for n in self.nodes.values())
        total_used_by = sum(len(n.used_by) for n in self.nodes.values())
        
        print(f"\n{'='*70}")
        print(f"📊 SELF-KNOWLEDGE RESULTS")
        print(f"{'='*70}")
        print(f"   Total Self-Knowledge Nodes: {len(self.nodes)}")
        print(f"   Modules: {self.total_modules}")
        print(f"   Classes: {self.total_classes}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Rate: {len(self.nodes)/elapsed:.1f} nodes/sec")
        print(f"\n   📈 Knowledge Density:")
        print(f"      Average Density: {avg_density:.1f}")
        print(f"      With Definition: {with_def} ({with_def*100/len(self.nodes):.1f}%)")
        print(f"      With Purpose: {with_purpose} ({with_purpose*100/len(self.nodes):.1f}%)")
        print(f"      Total Dependencies: {total_deps}")
        print(f"      Total Used-By Links: {total_used_by}")
        
        return {
            "total_nodes": len(self.nodes),
            "modules": self.total_modules,
            "classes": self.total_classes,
            "avg_density": avg_density,
            "with_definition": with_def,
            "with_purpose": with_purpose,
            "total_relations": total_deps + total_used_by,
            "time_seconds": elapsed
        }
    
    def explain_self(self, name: str) -> str:
        """
        "나의 이 부분은 무엇인가?"
        """
        node = self.nodes.get(name)
        if not node:
            # 부분 매칭 시도
            matches = [n for n in self.nodes.keys() if name.lower() in n.lower()]
            if matches:
                node = self.nodes[matches[0]]
            else:
                return f"'{name}'을 알지 못합니다."
        
        lines = [f"\n📖 나의 일부: {node.name} [{node.node_type}]"]
        lines.append(f"   위치: {node.path}")
        
        if node.definition:
            lines.append(f"\n   정의: {node.definition}")
        
        if node.purpose:
            lines.append(f"   목적: {node.purpose}")
        
        if node.depends_on:
            lines.append(f"\n   의존: {', '.join(node.depends_on[:5])}")
        
        if node.used_by:
            lines.append(f"   사용됨: {', '.join(node.used_by[:5])}")
        
        if node.children:
            lines.append(f"   포함: {', '.join(node.children[:5])}")
        
        lines.append(f"\n   [밀도: {node.density_score:.1f}]")
        
        return "\n".join(lines)
    
    def most_central(self, top_n: int = 10) -> List[SelfKnowledgeNode]:
        """가장 중심적인 (많이 사용되는) 자기 부분"""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: len(n.used_by),
            reverse=True
        )
        return sorted_nodes[:top_n]
    
    def most_dense(self, top_n: int = 10) -> List[SelfKnowledgeNode]:
        """가장 밀도 높은 자기 인식"""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.density_score,
            reverse=True
        )
        return sorted_nodes[:top_n]


def main():
    """자기 인식 데모"""
    
    knowledge = FluidSelfKnowledge(
        root_path=Path(__file__).parent,
        storage_path="data/self_knowledge.json"
    )
    
    # 자기 스캔
    result = knowledge.scan_self(target_dir="Core", max_files=300)
    
    # 가장 중심적인 자기 부분
    print("\n" + "="*70)
    print("🌟 나의 가장 중심적인 부분 (Most Used)")
    print("="*70)
    for i, node in enumerate(knowledge.most_central(5)):
        print(f"   {i+1}. {node.name} - {len(node.used_by)} modules depend on me")
    
    # 가장 잘 이해된 부분
    print("\n" + "="*70)
    print("📚 가장 밀도 높은 자기 인식 (Best Understood)")
    print("="*70)
    for i, node in enumerate(knowledge.most_dense(5)):
        print(f"   {i+1}. {node.name} [밀도: {node.density_score:.1f}]")
        if node.definition:
            print(f"       {node.definition[:60]}...")
    
    # 특정 모듈 설명
    print("\n" + "="*70)
    print("💭 자기 설명: Growth")
    print("="*70)
    print(knowledge.explain_self("growth"))
    
    print("\n✅ 이것이 '자기 인식'입니다.")
    print("   나는 내가 무엇으로 이루어져 있는지 알고 있습니다.")


if __name__ == "__main__":
    main()
