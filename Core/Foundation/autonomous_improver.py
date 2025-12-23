"""
Autonomous Self-Improvement Engine (자율적 자기 개선 엔진)
=========================================================

초월 AI를 향한 핵심 모듈.

핵심 기능:
1. 자체 파동 언어(Gravitational Linguistics)를 이용한 코드 분석
2. Causal/Topological Prediction을 통한 구조적 개선 검증
3. Safety Constraint Verifier (Immune System) 적용

Philosophical Integration:
이 모듈은 단순히 코드를 고치는 도구가 아닙니다.
Elysia의 '불편함(Cognitive Dissonance)'을 해소하고 '성장 의지(Intention)'를 실현하는 신체적 행위입니다.
"""

from __future__ import annotations

import ast
import os
import sys
import logging
import subprocess
import time
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto

logger = logging.getLogger("AutonomousImprover")


# 코드 분석 임계값
COMPLEXITY_LINES_THRESHOLD = 500
COMPLEXITY_FUNCTIONS_THRESHOLD = 20


class ImprovementType(Enum):
    """개선 유형"""
    CODE_OPTIMIZATION = auto()
    BUG_FIX = auto()
    NEW_FEATURE = auto()
    DOCUMENTATION = auto()
    REFACTORING = auto()
    PERFORMANCE = auto()
    LEARNING = auto()


class SafetyLevel(Enum):
    """안전 수준"""
    READ_ONLY = auto()
    SUGGEST_ONLY = auto()
    SANDBOX_MODIFY = auto()
    SUPERVISED_MODIFY = auto()
    AUTONOMOUS_MODIFY = auto()


@dataclass
class CodeAnalysis:
    """코드 분석 결과"""
    file_path: str
    total_lines: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: float
    issues: List[str]
    suggestions: List[str]


@dataclass
class ImprovementProposal:
    """개선 제안"""
    id: str
    improvement_type: ImprovementType
    target_file: str
    description: str
    description_kr: str
    original_code: str
    proposed_code: str
    reasoning: str
    confidence: float
    safety_level: SafetyLevel
    predicted_topology_change: Optional[str] = None  # Causal Prediction (e.g., "REMOVE_LINK: A->B")
    approved: bool = False
    applied: bool = False
    timestamp: float = field(default_factory=time.time)


class CodeIntrospector:
    """코드 자기 성찰 엔진"""
    
    def __init__(self, project_root: str = None, exclude_patterns: List[str] = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.analyzed_files: Dict[str, CodeAnalysis] = {}
        self.exclude_patterns = exclude_patterns or ['__pycache__', '.git', 'venv', 'Legacy', 'tests']
        
    def discover_python_files(self, exclude_patterns: List[str] = None) -> List[Path]:
        patterns = exclude_patterns or self.exclude_patterns
        python_files = []
        for py_file in self.project_root.rglob("*.py"):
            if not any(pattern in str(py_file) for pattern in patterns):
                python_files.append(py_file)
        return python_files
    
    def analyze_file(self, file_path: Path) -> CodeAnalysis:
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            lines = len(content.split('\n'))
            complexity = min(1.0, 
                (lines / COMPLEXITY_LINES_THRESHOLD) + 
                (len(functions) / COMPLEXITY_FUNCTIONS_THRESHOLD)
            )
            
            analysis = CodeAnalysis(
                file_path=str(file_path),
                total_lines=lines,
                functions=functions,
                classes=classes,
                imports=imports,
                complexity_score=complexity,
                issues=[],
                suggestions=[]
            )
            self.analyzed_files[str(file_path)] = analysis
            return analysis
            
        except Exception as e:
            return CodeAnalysis(
                file_path=str(file_path),
                total_lines=0,
                functions=[],
                classes=[],
                imports=[],
                complexity_score=0.0,
                issues=[f"Parse error: {str(e)}"],
                suggestions=[]
            )
    
    def analyze_self(self) -> Dict[str, Any]:
        """자기 자신(Core 디렉토리) 분석"""
        core_path = self.project_root / "Core"
        stats = {"total_files": 0, "total_lines": 0, "total_functions": 0, "total_classes": 0, "modules": {}, "complexity_avg": 0.0}
        
        for py_file in core_path.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                analysis = self.analyze_file(py_file)
                stats["total_files"] += 1
                stats["total_lines"] += analysis.total_lines
                stats["total_functions"] += len(analysis.functions)
                stats["total_classes"] += len(analysis.classes)
        
        if stats["total_files"] > 0:
            stats["complexity_avg"] = sum(a.complexity_score for a in self.analyzed_files.values()) / len(self.analyzed_files)
            
        return stats


class WaveLanguageAnalyzer:
    """파동 언어 기반 코드 분석기"""
    
    HIGH_MASS_PATTERNS = {"error": 90, "critical": 85, "main": 75, "core": 75}
    QUALITY_PATTERNS = {"todo": "IMPROVEMENT", "fixme": "BUG_FIX"}
    
    def analyze_code_quality(self, code: str, file_path: str = "") -> Dict[str, Any]:
        lines = code.split('\n')
        analysis = {
            "file": file_path,
            "total_lines": len(lines),
            "quality_issues": [],
            "resonance_score": 0.5, # Mock score
            "suggestions": []
        }
        
        if "TODO" in code:
            analysis["quality_issues"].append({"line": 0, "type": "TODO", "description": "Found TODO"})
            analysis["suggestions"].append({"type": "CLEANUP", "description_kr": "TODO 정리 필요"})
            
        return analysis


class LLMCodeImprover:
    """코드 개선 엔진"""
    def __init__(self, llm_bridge = None):
        self.llm_bridge = llm_bridge
        self.wave_analyzer = WaveLanguageAnalyzer()
        self.improvement_history = []


class SystemMonitor:
    """시스템 모니터링"""
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        import platform
        return {"platform": platform.system(), "timestamp": time.time()}


class ConstraintVerifier:
    """
    안전 제약 검증기 (면역 시스템)
    Safety Constraint Verifier (Immune System)
    """
    
    CRITICAL_FILES = ["living_elysia.py", "autonomous_improver.py", "causal_narrative_engine.py"]
    FORBIDDEN_PATTERNS = ["shutil.rmtree", "os.remove", "subprocess.call", "del ", "drop database"]
    
    @staticmethod
    def check_safety(proposal: ImprovementProposal) -> Tuple[bool, str]:
        """제안의 안전성 검증"""
        
        # 1. Syntax Check
        try:
            ast.parse(proposal.proposed_code)
        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)}"
            
        # 2. Forbidden Patterns
        code_lower = proposal.proposed_code.lower()
        for pattern in ConstraintVerifier.FORBIDDEN_PATTERNS:
            if pattern in code_lower and "test" not in proposal.target_file:
                return False, f"Forbidden Pattern Detected: {pattern}"
                
        # 3. Critical Files Check
        if any(cf in proposal.target_file for cf in ConstraintVerifier.CRITICAL_FILES):
            if len(proposal.proposed_code) < 100: 
                return False, "Proposed code too short for critical file (Risk of deletion)"
                
        return True, "Safe"


class AutonomousImprover:
    """자율적 자기 개선 엔진"""
    
    def __init__(self, project_root: str = None, llm_bridge = None, safety_level: SafetyLevel = SafetyLevel.SUGGEST_ONLY):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.introspector = CodeIntrospector(project_root)
        self.llm_improver = LLMCodeImprover(llm_bridge)
        self.system_monitor = SystemMonitor()
        self.safety_level = safety_level
        self.improvement_queue = []
        self.applied_improvements = []
        self.learning_log = []

    def verify_causal_outcome(self, proposal: ImprovementProposal, universe: Any) -> Dict[str, Any]:
        """
        인과적/위상적 결과 검증 (Topological Verification)
        """
        if not proposal.predicted_topology_change:
            return {"verified": True, "reason": "No topological prediction"}
            
        prediction = proposal.predicted_topology_change
        logger.info(f"Verifying Topo-Prediction: {prediction}")
        
        if prediction.startswith("REMOVE_LINK:"):
            try:
                parts = prediction.replace("REMOVE_LINK:", "").split("->")
                source_id = parts[0].strip()
                target_id = parts[1].strip()
                
                link_exists = False
                for line in universe.lines.values():
                    if line.source_point_id == source_id and line.target_point_id == target_id:
                        link_exists = True
                        break
                        
                if not link_exists:
                    return {"verified": True, "message": f"Link {source_id}->{target_id} successfully removed."}
                else:
                    return {"verified": False, "message": f"Link {source_id}->{target_id} still exists."}
            except Exception as e:
                return {"verified": False, "message": f"Verification Error: {str(e)}"}
                
        elif prediction.startswith("ADD_NODE:"):
            node_id = prediction.replace("ADD_NODE:", "").strip()
            if node_id in universe.points:
                return {"verified": True, "message": f"Node {node_id} successfully created."}
            else:
                return {"verified": False, "message": f"Node {node_id} not found."}

        return {"verified": False, "message": "Unknown prediction format"}

    def sense_tension(self, universe: Any, visions: List[Any]) -> List[Any]:
        """
        Hyper-Spatial Tension Sensing
        Detects mismatch between ArchitecturalVision (Blueprint) and Reality (Universe).
        """
        from Core.Foundation.metacognition import StructuralTension
        tensions = []
        
        # Simple simulation: Check for Missing Links defined in Vision
        for vision in visions:
            for intended_link in vision.intended_connections:
                # "Source -> Target"
                if "->" not in intended_link: continue
                
                src, tgt = [x.strip() for x in intended_link.split("->")]
                
                # Check reality
                link_exists = False
                for line in universe.lines.values():
                    if line.source_point_id == src and line.target_point_id == tgt:
                        link_exists = True
                        break
                
                if not link_exists:
                    tensions.append(StructuralTension(
                        source_id=src,
                        target_id=tgt,
                        tension_type="MISSING_LINK",
                        intensity=0.8,
                        vision_ref=vision.scope_id
                    ))
                    
        return tensions

    def propose_rewiring(self, tension: Any) -> Optional[ImprovementProposal]:
        """
        Generate a proposal to resolve Structural Tension (Rewiring).
        """
        import uuid
        if tension.tension_type == "MISSING_LINK":
            predicted_change = f"ADD_LINK: {tension.source_id} -> {tension.target_id}"
            return ImprovementProposal(
                id=str(uuid.uuid4())[:8],
                improvement_type=ImprovementType.REFACTORING,
                target_file="[HyperSpatial]", # Virtual Target
                description=f"Wire {tension.source_id} to {tension.target_id}",
                description_kr=f"누락된 신경망 연결: {tension.source_id} -> {tension.target_id}",
                original_code="",
                proposed_code=f"# Rewiring Action\n# Connect {tension.source_id} to {tension.target_id}",
                reasoning=f"Resolving tension in {tension.vision_ref}",
                confidence=tension.intensity,
                safety_level=SafetyLevel.SUGGEST_ONLY,
                predicted_topology_change=predicted_change
            )
        return None

    def self_analyze(self) -> Dict[str, Any]:
        """자기 분석 수행"""
        return {
            "timestamp": time.time(),
            "code_analysis": self.introspector.analyze_self(),
            "system_info": self.system_monitor.get_system_info()
        }

    def explain_capabilities(self) -> str:
        """Current capabilities description"""
        return (
            "Autonomous Self-Improvement Engine\n"
            "Analysis: Gravitational Linguistics\n"
            "Safety: Monitor Only (No execution without permission)"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Autonomous Self-Improvement Engine Demo")
    engine = AutonomousImprover()
    try:
        analysis = engine.self_analyze()
        print(f"Files: {analysis['code_analysis']['total_files']}")
    except Exception as e:
        print(f"Error: {e}")
