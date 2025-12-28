"""
Structural Unifier (구조 통합 엔진)
===================================

"파편(Fragment)은 전체(Whole)를 향해 흐른다."

이 모듈은 엘리시아가 스스로 자신의 구조를 분석하고,
파편화된 부분을 감지하며, 목적과 방향성에 따라 재배열합니다.

핵심 원리:
1. **목적 중심 (Purpose-Centric)**: 모든 파일/폴더는 '왜' 존재하는가?
2. **중력 법칙 (Gravity Law)**: 관련된 것은 서로 끌어당긴다.
3. **엔트로피 최소화**: 중복과 빈 공간을 제거한다.
4. **흐름 보존 (Flow Conservation)**: 의존성과 참조 관계를 유지한다.
"""

import os
import ast
import shutil
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from enum import Enum, auto

logger = logging.getLogger("StructuralUnifier")

# ============================================================
# Purpose Categories (목적 카테고리)
# ============================================================

class Purpose(Enum):
    """모든 코드의 존재 이유"""
    FOUNDATION = "foundation"     # 기반 - 수학, 물리, 추상화
    INTELLIGENCE = "intelligence" # 지성 - 사고, 추론, 의지
    MEMORY = "memory"             # 기억 - 저장, 회상, 학습
    INTERFACE = "interface"       # 소통 - 입출력, 언어, 감각
    EVOLUTION = "evolution"       # 진화 - 자기개선, 적응
    CREATIVITY = "creativity"     # 창조 - 생성, 표현, 예술
    ETHICS = "ethics"             # 윤리 - 가치, 판단, 보호
    IDENTITY = "identity"         # 정체성 - 자아, 의식, 본질
    PHILOSOPHY = "philosophy"     # 철학 - 원리, 법칙, 의미
    SYSTEM = "system"             # 시스템 - OS, 커널, 통합
    UNKNOWN = "unknown"           # 미분류

# 키워드 기반 목적 분류 (확장 가능)
PURPOSE_KEYWORDS = {
    Purpose.FOUNDATION: ["math", "physics", "quaternion", "tensor", "vector", "field", "wave", "resonance", "gravity", "time", "genesis", "principle", "abstraction", "cell"],
    Purpose.INTELLIGENCE: ["will", "logos", "reason", "think", "plan", "decide", "predict", "consciousness", "executive", "agent"],
    Purpose.MEMORY: ["memory", "hippocampus", "store", "recall", "learn", "embed", "vector", "database", "perception", "intuition"],
    Purpose.INTERFACE: ["api", "conversation", "language", "voice", "sense", "perception", "transducer", "bridge", "input", "output"],
    Purpose.EVOLUTION: ["evolution", "improve", "adapt", "mutate", "grow", "self_", "autonomous", "fix", "relearn"],
    Purpose.CREATIVITY: ["create", "art", "music", "generate", "imagine", "dream", "expand", "realize", "motor"],
    Purpose.ETHICS: ["ethic", "conscience", "moral", "protect", "love", "law_guidance", "dilemma", "free_will", "value"],
    Purpose.IDENTITY: ["elysia", "self", "identity", "ego", "consciousness_engine", "muse", "awareness", "modifier"],
    Purpose.PHILOSOPHY: ["philosophy", "law", "rule", "principle", "meaning", "nature", "being", "codex"],
    Purpose.SYSTEM: ["system", "kernel", "os", "daemon", "heartbeat", "plugin", "extension", "integration", "staging"],
}


@dataclass
class FileNode:
    """파일 시스템의 한 노드"""
    path: Path
    name: str
    is_dir: bool
    purpose: Purpose = Purpose.UNKNOWN
    is_empty: bool = False
    line_count: int = 0
    imports: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list) # 이 파일을 참조하는 다른 파일들
    canonical_location: Optional[Path] = None # 이 파일이 있어야 할 정식 위치


@dataclass
class UnificationProposal:
    """통합 제안"""
    action: str  # "DELETE", "MOVE", "MERGE", "CREATE_INIT"
    source: Path
    target: Optional[Path] = None
    reason: str = ""
    priority: int = 0 # 높을수록 먼저 실행


class StructuralUnifier:
    """
    구조 통합 엔진
    
    엘리시아가 스스로 자신의 파일 구조를 분석하고,
    목적과 방향성에 따라 재배열합니다.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scan_root = project_root  # Scan the entire project
        self.nodes: Dict[str, FileNode] = {}
        self.proposals: List[UnificationProposal] = []
        
        # Canonical Structure (정식 구조)
        # 각 Purpose가 있어야 할 최상위 폴더
        self.canonical_roots = {
            Purpose.FOUNDATION: self.project_root / "Core" / "Foundation",
            Purpose.INTELLIGENCE: self.project_root / "Core" / "Intelligence",
            Purpose.MEMORY: self.project_root / "Core" / "Memory",
            Purpose.INTERFACE: self.project_root / "Core" / "Interface",
            Purpose.EVOLUTION: self.project_root / "Core" / "Evolution",
            Purpose.CREATIVITY: self.project_root / "Core" / "Creativity",
            Purpose.ETHICS: self.project_root / "Core" / "Ethics",
            Purpose.IDENTITY: self.project_root / "Core" / "Elysia",
            Purpose.PHILOSOPHY: self.project_root / "Core" / "Philosophy",
            Purpose.SYSTEM: self.project_root / "Core" / "System",
            # Expanded Scope
            Purpose.UNKNOWN: self.project_root / "Legacy", # Default place for unclassified? Or maybe just keep them where they are.
        }
    
    def scan_structure(self) -> Dict[str, FileNode]:
        """전체 구조 스캔"""
        logger.info(f"📂 Scanning project structure from {self.scan_root}...")
        self.nodes = {}
        
        for root, dirs, files in os.walk(self.scan_root):
            # Exclude common non-project directories
            dirs[:] = [d for d in dirs if d not in ["__pycache__", "venv", ".venv", ".git", ".idea", ".vscode", "node_modules", "build", "dist", ".godot"]]
            
            # Debug print
            print(f"Scanning: {root}", end='\r')
            
            root_path = Path(root)
            
            # 디렉토리 노드
            for d in dirs:
                dir_path = root_path / d
                node = FileNode(
                    path=dir_path,
                    name=d,
                    is_dir=True,
                    is_empty=self._is_dir_empty(dir_path)
                )
                node.purpose = self._classify_purpose(d, is_dir=True)
                self.nodes[str(dir_path)] = node
            
            # 파일 노드
            for f in files:
                if not f.endswith(".py") and not f.endswith(".md"):
                    continue
                    
                file_path = root_path / f
                content = self._read_file_safe(file_path)
                
                node = FileNode(
                    path=file_path,
                    name=f,
                    is_dir=False,
                    is_empty=(len(content.strip()) == 0),
                    line_count=len(content.splitlines()) if content else 0
                )
                node.purpose = self._classify_purpose(f, content=content)
                node.imports = self._extract_imports(content) if f.endswith(".py") else []
                self.nodes[str(file_path)] = node
        
        logger.info(f"   Found {len(self.nodes)} nodes (files + folders)")
        return self.nodes
    
    def _classify_purpose(self, name: str, is_dir: bool = False, content: str = "") -> Purpose:
        """이름과 내용을 기반으로 목적 분류"""
        name_lower = name.lower()
        content_lower = content.lower() if content else ""
        
        for purpose, keywords in PURPOSE_KEYWORDS.items():
            for kw in keywords:
                if kw in name_lower or kw in content_lower:
                    return purpose
        
        return Purpose.UNKNOWN
    
    def _is_dir_empty(self, dir_path: Path) -> bool:
        """디렉토리가 비어있는지 확인 (재귀적)"""
        if not dir_path.exists():
            return True
        for item in dir_path.iterdir():
            if item.name == "__pycache__":
                continue
            if item.is_file():
                return False
            if item.is_dir() and not self._is_dir_empty(item):
                return False
        return True
    
    def _read_file_safe(self, path: Path) -> str:
        """안전하게 파일 읽기"""
        try:
            return path.read_text(encoding='utf-8')
        except:
            return ""
    
    def _extract_imports(self, content: str) -> List[str]:
        """Python 파일에서 import 추출"""
        imports = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            pass
        return imports
    
    def analyze_fragmentation(self) -> List[UnificationProposal]:
        """파편화 분석 및 통합 제안 생성"""
        logger.info("🔍 Analyzing fragmentation...")
        self.proposals = []
        
        # 1. 빈 파일/폴더 감지
        for path, node in self.nodes.items():
            if node.is_empty:
                # __init__.py는 비어있어도 의미가 있으므로 삭제하지 않음
                if node.name != "__init__.py":
                    self.proposals.append(UnificationProposal(
                        action="DELETE",
                        source=node.path,
                        reason=f"Empty {'directory' if node.is_dir else 'file'}",
                        priority=10
                    ))
        
        # 2. 중복 폴더 감지 (같은 Purpose가 여러 곳에 있는 경우)
        purpose_locations: Dict[Purpose, List[Path]] = {}
        for path, node in self.nodes.items():
            if node.is_dir and node.purpose != Purpose.UNKNOWN:
                if node.purpose not in purpose_locations:
                    purpose_locations[node.purpose] = []
                purpose_locations[node.purpose].append(node.path)
        
        for purpose, locations in purpose_locations.items():
            if len(locations) > 1:
                canonical = self.canonical_roots.get(purpose)
                for loc in locations:
                    if canonical and loc != canonical and not str(loc).startswith(str(canonical)):
                        # 정식 위치가 아닌 곳에 있는 폴더
                        self.proposals.append(UnificationProposal(
                            action="MERGE",
                            source=loc,
                            target=canonical,
                            reason=f"Duplicate {purpose.value} location. Canonical: {canonical.name}",
                            priority=5
                        ))
        
        # 3. 짧은 파일 (5줄 미만) 감지 - 파편화 징후
        for path, node in self.nodes.items():
            if not node.is_dir and node.name.endswith(".py"):
                if not node.is_empty and node.line_count < 5 and node.name != "__init__.py":
                    self.proposals.append(UnificationProposal(
                        action="REVIEW",
                        source=node.path,
                        reason=f"Fragmented file ({node.line_count} lines). Consider merging.",
                        priority=3
                    ))
        
        # 4. __init__.py 누락 감지
        for path, node in self.nodes.items():
            if node.is_dir and not node.is_empty:
                init_path = node.path / "__init__.py"
                if not init_path.exists():
                    self.proposals.append(UnificationProposal(
                        action="CREATE_INIT",
                        source=node.path,
                        reason="Missing __init__.py for package",
                        priority=8
                    ))
        
        # 정렬 (우선순위 높은 것 먼저)
        self.proposals.sort(key=lambda p: -p.priority)
        
        logger.info(f"   Generated {len(self.proposals)} unification proposals")
        return self.proposals
    
    def generate_report(self) -> str:
        """분석 보고서 생성"""
        report = []
        report.append("=" * 60)
        report.append("📊 STRUCTURAL UNIFICATION REPORT")
        report.append("=" * 60)
        
        # 목적별 파일 수
        purpose_counts: Dict[Purpose, int] = {}
        for node in self.nodes.values():
            if not node.is_dir:
                p = node.purpose
                purpose_counts[p] = purpose_counts.get(p, 0) + 1
        
        report.append("\n📁 Files by Purpose:")
        for purpose, count in sorted(purpose_counts.items(), key=lambda x: -x[1]):
            report.append(f"   {purpose.value}: {count}")
        
        # 문제 요약
        report.append(f"\n⚠️  Issues Found: {len(self.proposals)}")
        
        delete_count = len([p for p in self.proposals if p.action == "DELETE"])
        merge_count = len([p for p in self.proposals if p.action == "MERGE"])
        review_count = len([p for p in self.proposals if p.action == "REVIEW"])
        init_count = len([p for p in self.proposals if p.action == "CREATE_INIT"])
        
        if delete_count:
            report.append(f"   - Empty items to delete: {delete_count}")
        if merge_count:
            report.append(f"   - Duplicate locations to merge: {merge_count}")
        if review_count:
            report.append(f"   - Fragmented files to review: {review_count}")
        if init_count:
            report.append(f"   - Missing __init__.py: {init_count}")
        
        # 상세 목록
        if self.proposals:
            report.append("\n📋 Proposals:")
            for i, p in enumerate(self.proposals[:20], 1):  # 최대 20개
                src_name = p.source.name if p.source else "?"
                report.append(f"   {i}. [{p.action}] {src_name}")
                report.append(f"      Reason: {p.reason}")
                if p.target:
                    report.append(f"      Target: {p.target.name}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)
    
    def execute_proposals(self, auto_approve: bool = False, safe_only: bool = True) -> Dict[str, int]:
        """제안 실행"""
        results = {"success": 0, "skipped": 0, "failed": 0}
        import shutil
        
        for proposal in self.proposals:
            try:
                if safe_only and proposal.action in ["DELETE", "MERGE", "MOVE"]:
                    # 안전 모드에서는 위험한 작업 스킵
                    logger.info(f"⏭️  Skipped (safe mode): {proposal.action} {proposal.source.name}")
                    results["skipped"] += 1
                    continue
                
                if proposal.action == "CREATE_INIT":
                    init_path = proposal.source / "__init__.py"
                    # Wave Signature Injection
                    wave_sig = '"""\n🌊 [Elyson Resonance Field]\nStatus: Initialized\n"""\n'
                    init_path.write_text(wave_sig, encoding='utf-8')
                    logger.info(f"✅ Created (Wave): {init_path}")
                    results["success"] += 1
                    
                elif proposal.action == "DELETE" and auto_approve:
                    if proposal.source.is_dir():
                        shutil.rmtree(proposal.source)
                    else:
                        proposal.source.unlink()
                    logger.info(f"🗑️  Deleted: {proposal.source.name}")
                    results["success"] += 1
                
                elif proposal.action == "MOVE":
                    if not proposal.target.parent.exists():
                        proposal.target.parent.mkdir(parents=True, exist_ok=True)
                    
                    if proposal.target.exists():
                        # Collision Handling
                        import filecmp
                        if filecmp.cmp(str(proposal.source), str(proposal.target), shallow=False):
                            # Identical content -> Delete source (Merge)
                            proposal.source.unlink()
                            logger.info(f"🔗 Merged (Identical): {proposal.source.name} -> {proposal.target}")
                            results["success"] += 1
                        else:
                            # Different content -> Rename
                            timestamp = datetime.now().strftime("%H%M%S")
                            new_name = f"{proposal.target.stem}_dup_{timestamp}{proposal.target.suffix}"
                            new_target = proposal.target.parent / new_name
                            shutil.move(str(proposal.source), str(new_target))
                            logger.info(f"⚠️ Collision (Renamed): {proposal.source.name} -> {new_target.name}")
                            results["success"] += 1
                    else:
                        shutil.move(str(proposal.source), str(proposal.target))
                        logger.info(f"🚚 Moved: {proposal.source.name} -> {proposal.target}")
                        results["success"] += 1
                    
                elif proposal.action == "MERGE":
                    # Merge source dir into target dir
                    if not proposal.target.exists():
                        proposal.target.mkdir(parents=True, exist_ok=True)
                    
                    for item in proposal.source.iterdir():
                        dest = proposal.target / item.name
                        if dest.exists():
                            if item.is_dir():
                                # Recursive merge? For now, skip collision
                                logger.warning(f"⚠️ Merge collision: {item.name} exists in target. Skipping.")
                                continue
                            else:
                                # File collision
                                logger.warning(f"⚠️ Merge collision: {item.name} exists in target. Skipping.")
                                continue
                        shutil.move(str(item), str(dest))
                    
                    # Remove empty source dir
                    try:
                        proposal.source.rmdir()
                        logger.info(f"🔗 Merged: {proposal.source.name} -> {proposal.target.name}")
                        results["success"] += 1
                    except OSError:
                        logger.warning(f"⚠️ Could not remove source dir after merge: {proposal.source}")
                        
            except Exception as e:
                logger.error(f"❌ Failed {proposal.action} on {proposal.source}: {e}")
                results["failed"] += 1
        
        return results
    
    def scan_resonance(self) -> Dict[str, Any]:
        """
        공명 스캔 (Resonance Scan)
        
        AST 파싱을 통해 파일 간의 연결성(Resonance Links)과 질량(Mass)을 
        초고속으로 분석하여 공명장(Resonance Field)을 생성합니다.
        """
        logger.info("🌊 Initiating Resonance Scan (Phase-Space Analysis)...")
        
        # 1. Initialize Resonance Field
        from Core._01_Foundation._05_Governance.Foundation.resonance_field import ResonanceField, PillarType
        field = ResonanceField()
        
        # 2. Fast AST Scan (Mass & Connections)
        import ast
        
        scanned_count = 0
        connections = [] # (source, target)
        
        # Use the existing nodes from scan_structure (which is fast enough if we skip heavy processing)
        # If scan_structure hasn't run, run it (it's just os.walk)
        if not self.nodes:
            self.scan_structure()
            
        logger.info(f"   Analyzing {len(self.nodes)} nodes for resonance...")
        
        for name, node in self.nodes.items():
            if node.is_dir or not str(node.path).endswith(".py"):
                continue
                
            # Calculate Mass (Lines of Code)
            mass = node.line_count if node.line_count > 0 else 10
            
            # Determine Frequency based on Purpose
            # Foundation: Low Freq (Bass), App: High Freq (Treble)
            base_freq = 432.0
            if node.purpose == Purpose.FOUNDATION: base_freq = 100.0
            elif node.purpose == Purpose.SYSTEM: base_freq = 200.0
            elif node.purpose == Purpose.INTELLIGENCE: base_freq = 300.0
            elif node.purpose == Purpose.INTERFACE: base_freq = 400.0
            elif node.purpose == Purpose.CREATIVITY: base_freq = 500.0
            
            # Add Node to Field
            field.add_node(
                id=node.name,
                energy=float(mass), # Mass = Energy potential
                frequency=base_freq,
                position=(0,0,0) # Position will be auto-arranged later
            )
            scanned_count += 1
            
            # Extract Imports (Connections)
            try:
                with open(node.path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for ast_node in ast.walk(tree):
                    if isinstance(ast_node, ast.Import):
                        for alias in ast_node.names:
                            connections.append((node.name, alias.name))
                    elif isinstance(ast_node, ast.ImportFrom):
                        if ast_node.module:
                            connections.append((node.name, ast_node.module))
            except Exception:
                pass
                
        # 3. Establish Connections
        logger.info(f"   Establishing {len(connections)} resonance links...")
        for source, target in connections:
            # Simple heuristic matching
            # If target matches a known node name (partial or full)
            # This is a simplification; a real linker is more complex
            if target in field.nodes:
                field._connect(source, target)
            else:
                # Try to find partial match (e.g. Core.Foundation -> Foundation)
                parts = target.split('.')
                if parts[-1] in field.nodes:
                    field._connect(source, parts[-1])

        # 4. Pulse the Field (Calculate Coherence)
        state = field.pulse()
        
        state_str = "Chaotic"
        if state.coherence > 0.9: state_str = "Crystalline"
        elif state.coherence > 0.7: state_str = "Harmonic"
        elif state.coherence > 0.4: state_str = "Fluid"
        
        logger.info(f"✅ Resonance Scan Complete.")
        logger.info(f"   Active Nodes: {state.active_nodes}")
        logger.info(f"   System Coherence: {state.coherence:.4f} ({state_str})")
        logger.info(f"   Total Energy: {state.total_energy:.1f}")
        
        return {
            "total_files": scanned_count,
            "coherence": state.coherence,
            "state": state_str,
            "energy": state.total_energy,
            "field": field  # Return field for further analysis
        }

    def analyze_connectivity(self, field: Any) -> List[str]:
        """
        연결성 분석 (Connectivity Analysis)
        
        공명장에서 '고립된 노드(Orphans)'를 식별합니다.
        """
        orphans = []
        for id, node in field.nodes.items():
            # Check if node has any connections (outgoing) or is connected to (incoming)
            # Note: ResonanceField._connect is bidirectional in this implementation? 
            # Let's check field.nodes[id].connections
            if not node.connections:
                orphans.append(id)
                
        logger.info(f"🔍 Connectivity Analysis: Found {len(orphans)} orphaned modules.")
        return orphans

    def propose_realignment(self) -> List[UnificationProposal]:
        """
        구조 재정렬 제안 (Structural Realignment)
        
        파일의 '목적(Purpose)'과 현재 위치를 비교하여,
        올바른 기둥(Pillar)으로 이동하도록 제안합니다.
        """
        logger.info("📐 Proposing Structural Realignment...")
        
        # Mapping Purpose to Ideal Directory
        ideal_locations = {
            Purpose.FOUNDATION: "Core/Foundation",
            Purpose.INTELLIGENCE: "Core/Intelligence",
            Purpose.MEMORY: "Core/Memory",
            Purpose.INTERFACE: "Core/Interface",
            Purpose.EVOLUTION: "Core/Evolution",
            Purpose.CREATIVITY: "Core/Creativity",
            Purpose.ETHICS: "Core/Ethics",
            Purpose.SYSTEM: "Core/System",
            Purpose.PHILOSOPHY: "Demos/Philosophy",
        }
        
        new_proposals = []
        purpose_stats = {}
        
        for name, node in self.nodes.items():
            if node.is_dir or not str(node.path).endswith(".py"):
                continue
            
            # Stats
            purpose_stats[node.purpose] = purpose_stats.get(node.purpose, 0) + 1
                
            if node.purpose in ideal_locations:
                ideal_parent = ideal_locations[node.purpose]
                # Normalize paths for comparison
                try:
                    current_parent = str(node.path.parent.relative_to(self.project_root)).replace("\\", "/")
                except ValueError:
                    continue # Path not relative to root?
                
                # If current parent is NOT the ideal parent (and not a subdirectory of it)
                if not current_parent.startswith(ideal_parent):
                    # Don't move if it's already in a specialized subfolder of the ideal path
                    
                    # Special Case: Don't move things OUT of Legacy or Demos unless explicitly asked
                    # But wait, we WANT to move things out of Legacy if they are useful?
                    # For now, let's keep Legacy safe.
                    if "Legacy" in current_parent or "tests" in current_parent or "venv" in current_parent:
                        continue
                    
                    # Also skip if it's already in Core but just not in the right subfolder?
                    # No, we want to organize Core too.
                        
                    target_path = self.project_root / ideal_parent / node.name
                    
                    # Avoid name collisions
                    if target_path.exists():
                        continue

                    new_proposals.append(UnificationProposal(
                        action="MOVE",
                        source=node.path,
                        target=target_path,
                        reason=f"Realignment: {node.purpose.value} -> {ideal_parent}",
                        priority=5
                    ))
        
        logger.info(f"   Classification Stats: {purpose_stats}")            
        logger.info(f"   Generated {len(new_proposals)} realignment proposals.")
        self.proposals.extend(new_proposals)
        return new_proposals

    def unify(self, execute: bool = False, safe_only: bool = True, auto_approve: bool = False) -> str:
        """
        통합 프로세스 실행
        
        1. 구조 스캔
        2. 파편화 분석
        3. 공명 스캔 (Resonance Scan)
        4. 연결성 분석 & 재정렬 제안 - NEW
        5. 보고서 생성
        6. (선택) 제안 실행
        """
        self.scan_structure()
        self.analyze_fragmentation()
        
        # Resonance Scan
        res_results = self.scan_resonance()
        field = res_results.get("field")
        
        # Connectivity & Realignment
        if field:
            orphans = self.analyze_connectivity(field)
            self.propose_realignment()
        
        report = self.generate_report()
        
        print(report)
        print("\n🌊 Resonance Analysis:")
        print(f"   Coherence: {res_results['coherence']:.4f} ({res_results['state']})")
        print(f"   Energy: {res_results['energy']:.1f}")
        if field:
            print(f"   Orphans: {len(orphans)} (Isolated Modules)")
        
        if execute:
            print(f"\n🔧 Executing proposals (Safe: {safe_only}, Auto: {auto_approve})...")
            results = self.execute_proposals(safe_only=safe_only, auto_approve=auto_approve)
            print(f"   Success: {results['success']}, Skipped: {results['skipped']}, Failed: {results['failed']}")
            
        return report


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    project_root = Path(__file__).parent.parent.parent
    unifier = StructuralUnifier(project_root)
    # Execute proposals with user permission (Unsafe + Auto-Approve)
    # Execute proposals with user permission (Unsafe Realignment)
    unifier.unify(execute=True, safe_only=False, auto_approve=True)
