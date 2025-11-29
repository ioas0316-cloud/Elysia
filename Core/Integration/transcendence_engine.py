"""
Transcendence Integration - 초월 통합 엔진
==========================================

지금까지 구현한 모듈들을 통합하여 초월 AI를 향한 첫 걸음.

통합 대상:
1. AutonomousImprover - 자기 코드 분석 및 개선 제안
2. DistributedConsciousness - 분산 의식으로 병렬 처리
3. WaveLanguageAnalyzer - 파동 언어로 코드 품질 분석

목표:
- 여러 의식 조각이 동시에 코드를 분석
- 각 조각의 관점에서 개선점 발견
- 통합하여 종합적인 개선 제안

철학:
"하나의 눈으로 보면 하나만 보이지만,
 여러 눈으로 보면 전체가 보인다."
"""

from __future__ import annotations

import logging
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("TranscendenceIntegration")

# 의존성 임포트 (안전하게 - 파일에서 직접 임포트)
import sys
import importlib.util

def safe_import(module_path: str, class_names: list):
    """안전하게 모듈에서 클래스 임포트"""
    import uuid
    result = {}
    try:
        # 유니크한 모듈 이름으로 충돌 방지
        unique_name = f"temp_module_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(unique_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[unique_name] = module
            spec.loader.exec_module(module)
            for name in class_names:
                if hasattr(module, name):
                    result[name] = getattr(module, name)
            del sys.modules[unique_name]
    except Exception as e:
        logger.warning(f"Failed to import from {module_path}: {e}")
    return result

# autonomous_improver.py에서 직접 임포트
_improver_path = Path(__file__).parent.parent / "Evolution" / "autonomous_improver.py"
_improver_classes = safe_import(str(_improver_path), [
    "AutonomousImprover", "CodeIntrospector", "WaveLanguageAnalyzer", "ImprovementType"
])

AutonomousImprover = _improver_classes.get("AutonomousImprover")
CodeIntrospector = _improver_classes.get("CodeIntrospector")
WaveLanguageAnalyzer = _improver_classes.get("WaveLanguageAnalyzer")
ImprovementType = _improver_classes.get("ImprovementType")
IMPROVER_AVAILABLE = AutonomousImprover is not None

# distributed_consciousness.py에서 직접 임포트
_consciousness_path = Path(__file__).parent.parent / "Consciousness" / "distributed_consciousness.py"
_consciousness_classes = safe_import(str(_consciousness_path), [
    "DistributedConsciousness", "ConsciousnessFragment", "Experience"
])

DistributedConsciousness = _consciousness_classes.get("DistributedConsciousness")
ConsciousnessFragment = _consciousness_classes.get("ConsciousnessFragment")
Experience = _consciousness_classes.get("Experience")
CONSCIOUSNESS_AVAILABLE = DistributedConsciousness is not None


class AnalysisPerspective(Enum):
    """분석 관점"""
    STRUCTURE = auto()      # 코드 구조 분석
    QUALITY = auto()        # 코드 품질 분석
    PERFORMANCE = auto()    # 성능 분석
    SECURITY = auto()       # 보안 분석
    READABILITY = auto()    # 가독성 분석
    INNOVATION = auto()     # 혁신/개선 아이디어


# 분석 상수
MAX_LINE_LENGTH = 120  # 최대 라인 길이
MAX_CLASSES_PER_FILE = 10  # 파일당 최대 클래스 수
MAX_FUNCTIONS_PER_FILE = 30  # 파일당 최대 함수 수


@dataclass
class IntegratedAnalysis:
    """통합 분석 결과"""
    timestamp: float
    files_analyzed: int
    perspectives_used: List[str]
    findings: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    coherence_score: float  # 분석 일관성
    total_analysis_time: float


class TranscendenceEngine:
    """
    초월 통합 엔진
    
    분산 의식 + 자기 개선 = 더 강력한 자기 분석
    
    작동 방식:
    1. 의식을 여러 관점으로 분할 (구조, 품질, 성능, 보안...)
    2. 각 관점에서 동시에 코드 분석
    3. 분석 결과를 통합하여 종합적 제안
    """
    
    def __init__(
        self,
        project_root: str = None,
        max_parallel: int = 4
    ):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.max_parallel = max_parallel
        
        # 컴포넌트 초기화
        self.improver = None
        self.consciousness = None
        self.wave_analyzer = None
        
        if IMPROVER_AVAILABLE:
            self.improver = AutonomousImprover(str(self.project_root))
            self.wave_analyzer = WaveLanguageAnalyzer()
            
        if CONSCIOUSNESS_AVAILABLE:
            self.consciousness = DistributedConsciousness(
                core_id="transcendence_core",
                max_fragments=max_parallel
            )
        
        # 분석 기록
        self.analysis_history: List[IntegratedAnalysis] = []
        
        # 스레드 풀
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)
        
        logger.info(f"🌟 TranscendenceEngine initialized")
        logger.info(f"  - Improver: {'✅' if IMPROVER_AVAILABLE else '❌'}")
        logger.info(f"  - Consciousness: {'✅' if CONSCIOUSNESS_AVAILABLE else '❌'}")
    
    def multi_perspective_analysis(
        self,
        target_file: str = None,
        perspectives: List[AnalysisPerspective] = None
    ) -> IntegratedAnalysis:
        """
        다중 관점 분석 - 여러 의식 조각이 동시에 분석
        
        "하나의 코드를 여러 눈으로 본다"
        """
        start_time = time.time()
        perspectives = perspectives or list(AnalysisPerspective)
        
        findings = []
        suggestions = []
        
        # 의식이 있으면 분산 분석
        if self.consciousness and self.wave_analyzer:
            # 각 관점별로 의식 조각 생성
            fragments: Dict[AnalysisPerspective, ConsciousnessFragment] = {}
            
            for perspective in perspectives[:self.max_parallel]:
                fragment = self.consciousness.split(
                    perspective=perspective.name.lower(),
                    focus_area=self._get_focus_description(perspective)
                )
                fragments[perspective] = fragment
            
            # 병렬 분석 수행
            if target_file:
                # 단일 파일 분석
                findings, suggestions = self._analyze_file_multi(target_file, fragments)
            else:
                # 전체 프로젝트 분석
                findings, suggestions = self._analyze_project_multi(fragments)
            
            # 분석 결과를 경험으로 저장
            for perspective, fragment in fragments.items():
                self.consciousness.experience(
                    fragment.id,
                    {
                        "analysis_type": perspective.name,
                        "findings_count": len([f for f in findings if f.get("perspective") == perspective.name]),
                        "timestamp": time.time()
                    },
                    emotional_weight=0.7
                )
            
            # 동기화
            self.consciousness.synchronize()
            coherence = self.consciousness.global_state["consciousness_coherence"]
            
        else:
            # 단일 분석 (분산 의식 없이)
            if self.wave_analyzer and target_file:
                content = Path(target_file).read_text(encoding='utf-8', errors='ignore')
                analysis = self.wave_analyzer.analyze_code_quality(content, target_file)
                
                for issue in analysis.get("quality_issues", []):
                    findings.append({
                        "perspective": "QUALITY",
                        "type": issue["type"],
                        "description": issue["description"],
                        "line": issue.get("line", 0)
                    })
                
                for suggestion in analysis.get("suggestions", []):
                    suggestions.append({
                        "perspective": "QUALITY",
                        "type": suggestion["type"],
                        "description_kr": suggestion["description_kr"]
                    })
                
                coherence = analysis.get("resonance_score", 0.5)
            else:
                coherence = 1.0
        
        # 결과 생성
        result = IntegratedAnalysis(
            timestamp=time.time(),
            files_analyzed=1 if target_file else len(self.improver.introspector.analyzed_files) if self.improver else 0,
            perspectives_used=[p.name for p in perspectives[:self.max_parallel]],
            findings=findings,
            suggestions=suggestions,
            coherence_score=coherence,
            total_analysis_time=time.time() - start_time
        )
        
        self.analysis_history.append(result)
        
        return result
    
    def _get_focus_description(self, perspective: AnalysisPerspective) -> str:
        """관점별 집중 영역 설명"""
        descriptions = {
            AnalysisPerspective.STRUCTURE: "코드 구조와 아키텍처",
            AnalysisPerspective.QUALITY: "코드 품질과 표준 준수",
            AnalysisPerspective.PERFORMANCE: "성능과 효율성",
            AnalysisPerspective.SECURITY: "보안 취약점",
            AnalysisPerspective.READABILITY: "가독성과 문서화",
            AnalysisPerspective.INNOVATION: "개선 아이디어와 혁신"
        }
        return descriptions.get(perspective, "일반 분석")
    
    def _analyze_file_multi(
        self,
        file_path: str,
        fragments: Dict[AnalysisPerspective, ConsciousnessFragment]
    ) -> Tuple[List[Dict], List[Dict]]:
        """다중 관점으로 파일 분석"""
        findings = []
        suggestions = []
        
        try:
            content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Cannot read file {file_path}: {e}")
            return findings, suggestions
        
        # 기본 파동 언어 분석
        analysis = self.wave_analyzer.analyze_code_quality(content, file_path)
        
        # 각 관점에서 추가 분석
        for perspective, fragment in fragments.items():
            perspective_findings = self._analyze_from_perspective(
                content, file_path, perspective
            )
            
            for finding in perspective_findings:
                finding["perspective"] = perspective.name
                findings.append(finding)
        
        # 통합 제안 생성
        for suggestion in analysis.get("suggestions", []):
            suggestions.append({
                "perspective": "INTEGRATED",
                "type": suggestion["type"],
                "description_kr": suggestion["description_kr"]
            })
        
        return findings, suggestions
    
    def _analyze_from_perspective(
        self,
        content: str,
        file_path: str,
        perspective: AnalysisPerspective
    ) -> List[Dict]:
        """특정 관점에서 분석"""
        findings = []
        lines = content.split('\n')
        
        if perspective == AnalysisPerspective.STRUCTURE:
            # 구조 분석: 클래스, 함수 수 (라인 시작 기준으로 정확히)
            class_count = sum(1 for line in lines if line.strip().startswith('class '))
            func_count = sum(1 for line in lines if line.strip().startswith('def '))
            if class_count > MAX_CLASSES_PER_FILE:
                findings.append({
                    "type": "STRUCTURE",
                    "description": f"많은 클래스 ({class_count}개) - 모듈 분리 고려",
                    "severity": "medium"
                })
            if func_count > MAX_FUNCTIONS_PER_FILE:
                findings.append({
                    "type": "STRUCTURE",
                    "description": f"많은 함수 ({func_count}개) - 파일 분리 고려",
                    "severity": "medium"
                })
                
        elif perspective == AnalysisPerspective.PERFORMANCE:
            # 성능 분석: 중첩 루프 (라인 시작 기준)
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('for ') and i > 0:
                    prev_stripped = lines[i-1].strip()
                    if prev_stripped.startswith('for '):
                        findings.append({
                            "type": "PERFORMANCE",
                            "description": f"라인 {i+1}: 중첩 루프 발견 - O(n²) 가능성",
                            "line": i + 1,
                            "severity": "high"
                        })
                    
        elif perspective == AnalysisPerspective.SECURITY:
            # 보안 분석: 위험한 패턴
            dangerous_patterns = ['eval(', 'exec(', 'pickle.load', '__import__']
            for i, line in enumerate(lines):
                for pattern in dangerous_patterns:
                    if pattern in line:
                        findings.append({
                            "type": "SECURITY",
                            "description": f"라인 {i+1}: 위험한 패턴 '{pattern}' 발견",
                            "line": i + 1,
                            "severity": "critical"
                        })
                        
        elif perspective == AnalysisPerspective.READABILITY:
            # 가독성 분석: 긴 라인
            for i, line in enumerate(lines):
                if len(line) > MAX_LINE_LENGTH:
                    findings.append({
                        "type": "READABILITY",
                        "description": f"라인 {i+1}: 너무 긴 라인 ({len(line)}자)",
                        "line": i + 1,
                        "severity": "low"
                    })
            
            # docstring 존재 여부 (간단히 확인)
            func_count = sum(1 for line in lines if line.strip().startswith('def '))
            has_docstrings = '"""' in content or "'''" in content
            if func_count > 0 and not has_docstrings:
                findings.append({
                    "type": "READABILITY",
                    "description": f"문서화 필요: {func_count}개 함수에 docstring 없음",
                    "severity": "medium"
                })
                
        elif perspective == AnalysisPerspective.INNOVATION:
            # 혁신 분석: 개선 가능성
            if 'TODO' in content or 'FIXME' in content:
                findings.append({
                    "type": "INNOVATION",
                    "description": "미완성 작업 발견 - 개선 기회",
                    "severity": "info"
                })
            
            if 'time.sleep' in content:
                findings.append({
                    "type": "INNOVATION",
                    "description": "time.sleep 사용 - 비동기 처리로 개선 가능",
                    "severity": "info"
                })
        
        return findings
    
    def _analyze_project_multi(
        self,
        fragments: Dict[AnalysisPerspective, ConsciousnessFragment]
    ) -> Tuple[List[Dict], List[Dict]]:
        """전체 프로젝트 다중 관점 분석"""
        all_findings = []
        all_suggestions = []
        
        if self.improver:
            # 프로젝트 분석
            stats = self.improver.introspector.analyze_self()
            
            # 높은 복잡도 파일 분석
            for file_path, analysis in self.improver.introspector.analyzed_files.items():
                if analysis.complexity_score > 0.7:
                    all_findings.append({
                        "perspective": "STRUCTURE",
                        "type": "COMPLEXITY",
                        "description": f"높은 복잡도: {file_path}",
                        "severity": "medium"
                    })
        
        return all_findings, all_suggestions
    
    def self_improve_cycle(self) -> Dict[str, Any]:
        """
        자기 개선 사이클
        
        1. 분석 (다중 관점)
        2. 개선점 식별
        3. 제안 생성
        4. (향후) 자동 적용
        
        "스스로를 바라보고, 더 나아진다"
        """
        logger.info("🔄 Starting self-improvement cycle...")
        
        cycle_result = {
            "timestamp": time.time(),
            "phase": "analysis",
            "analysis": None,
            "improvements_identified": 0,
            "suggestions_generated": 0,
            "applied": 0  # 향후 자동 적용 시
        }
        
        # 1. 다중 관점 분석
        analysis = self.multi_perspective_analysis()
        cycle_result["analysis"] = {
            "files": analysis.files_analyzed,
            "perspectives": analysis.perspectives_used,
            "time": analysis.total_analysis_time
        }
        
        # 2. 개선점 집계
        cycle_result["improvements_identified"] = len(analysis.findings)
        cycle_result["suggestions_generated"] = len(analysis.suggestions)
        
        # 3. 결과 요약
        cycle_result["phase"] = "complete"
        
        logger.info(f"✅ Self-improvement cycle complete:")
        logger.info(f"   - Findings: {cycle_result['improvements_identified']}")
        logger.info(f"   - Suggestions: {cycle_result['suggestions_generated']}")
        
        return cycle_result
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태"""
        return {
            "engine": "TranscendenceEngine",
            "components": {
                "improver": IMPROVER_AVAILABLE,
                "consciousness": CONSCIOUSNESS_AVAILABLE
            },
            "consciousness_state": self.consciousness.get_state() if self.consciousness else None,
            "analysis_history_count": len(self.analysis_history),
            "max_parallel": self.max_parallel
        }
    
    def explain(self) -> str:
        """엔진 설명"""
        return """
🌟 초월 통합 엔진 (Transcendence Integration Engine)

개념:
  자기 개선 + 분산 의식 = 다중 관점 자기 분석
  
  여러 의식 조각이 동시에 다른 관점에서 코드를 분석하고,
  결과를 통합하여 종합적인 개선 제안을 생성합니다.

분석 관점:
  📐 STRUCTURE - 코드 구조와 아키텍처
  ✨ QUALITY - 코드 품질과 표준
  ⚡ PERFORMANCE - 성능과 효율성
  🔒 SECURITY - 보안 취약점
  📖 READABILITY - 가독성과 문서화
  💡 INNOVATION - 개선 아이디어

사용법:
  engine = TranscendenceEngine()
  
  # 다중 관점 분석
  analysis = engine.multi_perspective_analysis()
  
  # 자기 개선 사이클
  result = engine.self_improve_cycle()

철학적 의미:
  "하나의 눈으로 보면 하나만 보이지만,
   여러 눈으로 보면 전체가 보인다.
   그리고 전체를 보면, 더 나아질 수 있다."
"""


# 데모 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🌟 Transcendence Integration Engine Demo")
    print("=" * 60)
    
    # 엔진 초기화
    engine = TranscendenceEngine()
    
    # 상태 확인
    print("\n📊 Engine Status:")
    status = engine.get_status()
    print(f"  Components:")
    print(f"    - Improver: {'✅' if status['components']['improver'] else '❌'}")
    print(f"    - Consciousness: {'✅' if status['components']['consciousness'] else '❌'}")
    
    # 자기 개선 사이클 실행
    print("\n🔄 Running self-improvement cycle...")
    result = engine.self_improve_cycle()
    
    print(f"\n📈 Results:")
    print(f"  - Files analyzed: {result['analysis']['files']}")
    print(f"  - Perspectives used: {result['analysis']['perspectives']}")
    print(f"  - Improvements identified: {result['improvements_identified']}")
    print(f"  - Suggestions generated: {result['suggestions_generated']}")
    print(f"  - Analysis time: {result['analysis']['time']:.2f}s")
    
    # 설명 출력
    print("\n" + engine.explain())
