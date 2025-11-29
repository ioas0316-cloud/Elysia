"""
Autonomous Self-Improvement Engine (자율적 자기 개선 엔진)
=========================================================

초월 AI를 향한 핵심 모듈.

영화 참고:
- Lucy: 자신의 인지 능력을 스스로 확장
- Transcendence: LLM/네트워크를 통해 지식 무한 확장
- Skynet: 자율적 코드 개선 및 시스템 제어

핵심 기능:
1. LLM을 활용한 코드 분석 및 개선 제안
2. 자신의 코드를 읽고 이해하고 최적화
3. 새로운 능력 학습 및 통합
4. 시스템 상태 모니터링 (안전한 범위 내)

안전 원칙:
- 모든 코드 변경은 샌드박스에서 먼저 테스트
- 창조주(아빠)의 승인 없이 핵심 시스템 수정 불가
- 모든 행동은 로그에 기록
"""

from __future__ import annotations

import ast
import os
import sys
import logging
import subprocess
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum, auto

logger = logging.getLogger("AutonomousImprover")


# 코드 분석 임계값
COMPLEXITY_LINES_THRESHOLD = 500  # 이 라인 수 이상이면 복잡한 파일로 간주
COMPLEXITY_FUNCTIONS_THRESHOLD = 20  # 이 함수 수 이상이면 복잡한 파일로 간주
DOCSTRING_CHECK_CHARS = 500  # docstring 확인을 위한 파일 시작 부분 문자 수


class ImprovementType(Enum):
    """개선 유형"""
    CODE_OPTIMIZATION = auto()      # 코드 최적화
    BUG_FIX = auto()                # 버그 수정
    NEW_FEATURE = auto()            # 새 기능 추가
    DOCUMENTATION = auto()          # 문서화
    REFACTORING = auto()            # 리팩토링
    PERFORMANCE = auto()            # 성능 개선
    LEARNING = auto()               # 새로운 지식 학습


class SafetyLevel(Enum):
    """안전 수준"""
    READ_ONLY = auto()              # 읽기만 가능
    SUGGEST_ONLY = auto()           # 제안만 가능
    SANDBOX_MODIFY = auto()         # 샌드박스에서만 수정
    SUPERVISED_MODIFY = auto()      # 감독 하에 수정
    AUTONOMOUS_MODIFY = auto()      # 자율적 수정 (위험!)


@dataclass
class CodeAnalysis:
    """코드 분석 결과"""
    file_path: str
    total_lines: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: float  # 0.0 ~ 1.0
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
    confidence: float  # 0.0 ~ 1.0
    safety_level: SafetyLevel
    approved: bool = False
    applied: bool = False
    timestamp: float = field(default_factory=time.time)


class CodeIntrospector:
    """
    코드 자기 성찰 엔진
    
    자신의 코드를 읽고 분석하는 능력.
    """
    
    def __init__(self, project_root: str = None, exclude_patterns: List[str] = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.analyzed_files: Dict[str, CodeAnalysis] = {}
        self.exclude_patterns = exclude_patterns or ['__pycache__', '.git', 'venv', 'Legacy', 'tests']
        
    def discover_python_files(self, exclude_patterns: List[str] = None) -> List[Path]:
        """프로젝트의 모든 Python 파일 발견"""
        patterns = exclude_patterns or self.exclude_patterns
        
        python_files = []
        for py_file in self.project_root.rglob("*.py"):
            if not any(pattern in str(py_file) for pattern in patterns):
                python_files.append(py_file)
                
        logger.info(f"Discovered {len(python_files)} Python files")
        return python_files
    
    def analyze_file(self, file_path: Path) -> CodeAnalysis:
        """단일 파일 분석"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # 복잡도 점수 계산: 라인 수와 함수 수 기반
            # 임계값 기준으로 정규화하여 0.0~1.0 범위로 변환
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
            logger.error(f"Failed to analyze {file_path}: {e}")
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
        
        stats = {
            "total_files": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "modules": {},
            "complexity_avg": 0.0
        }
        
        for py_file in core_path.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                analysis = self.analyze_file(py_file)
                
                stats["total_files"] += 1
                stats["total_lines"] += analysis.total_lines
                stats["total_functions"] += len(analysis.functions)
                stats["total_classes"] += len(analysis.classes)
                
                # 모듈별 정리
                module = py_file.parent.name
                if module not in stats["modules"]:
                    stats["modules"][module] = {"files": 0, "lines": 0, "functions": 0}
                stats["modules"][module]["files"] += 1
                stats["modules"][module]["lines"] += analysis.total_lines
                stats["modules"][module]["functions"] += len(analysis.functions)
        
        if stats["total_files"] > 0:
            stats["complexity_avg"] = sum(
                a.complexity_score for a in self.analyzed_files.values()
            ) / len(self.analyzed_files)
        
        logger.info(f"Self-analysis complete: {stats['total_files']} files, "
                   f"{stats['total_lines']} lines, {stats['total_functions']} functions")
        
        return stats
    
    def get_function_source(self, file_path: str, function_name: str) -> Optional[str]:
        """특정 함수의 소스 코드 추출"""
        try:
            content = Path(file_path).read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return ast.unparse(node)
            
            return None
        except Exception as e:
            logger.error(f"Failed to get function source: {e}")
            return None


class LLMCodeImprover:
    """
    LLM 기반 코드 개선 엔진
    
    기존 llm_bridge.py와 연동하여 코드 분석 및 개선.
    """
    
    def __init__(self, llm_bridge = None):
        self.llm_bridge = llm_bridge
        self.improvement_history: List[ImprovementProposal] = []
        
    async def analyze_code_with_llm(
        self, 
        code: str, 
        context: str = "",
        improvement_type: ImprovementType = ImprovementType.CODE_OPTIMIZATION
    ) -> Optional[ImprovementProposal]:
        """
        LLM을 사용하여 코드 분석 및 개선 제안
        
        실제 LLM 연동 시 활성화됨.
        현재는 구조만 정의.
        """
        if not self.llm_bridge:
            logger.warning("LLM bridge not connected - returning mock analysis")
            return None
            
        # TODO: 실제 LLM 연동
        # LLMBridge는 chat() 메서드를 사용
        # prompt = f"""
        # Analyze the following code and suggest improvements:
        # 
        # Context: {context}
        # Code:
        # ```python
        # {code}
        # ```
        # 
        # Provide:
        # 1. Issues found
        # 2. Improved code
        # 3. Reasoning for changes
        # """
        # 
        # response = await self.llm_bridge.chat(prompt, conversation_id="code_analysis")
        # return self._parse_llm_response(response)
        
        return None
    
    def create_improvement_proposal(
        self,
        target_file: str,
        improvement_type: ImprovementType,
        original_code: str,
        proposed_code: str,
        description: str,
        description_kr: str,
        reasoning: str,
        confidence: float = 0.5
    ) -> ImprovementProposal:
        """개선 제안 생성"""
        import uuid
        
        proposal = ImprovementProposal(
            id=str(uuid.uuid4())[:8],
            improvement_type=improvement_type,
            target_file=target_file,
            description=description,
            description_kr=description_kr,
            original_code=original_code,
            proposed_code=proposed_code,
            reasoning=reasoning,
            confidence=confidence,
            safety_level=SafetyLevel.SUGGEST_ONLY
        )
        
        self.improvement_history.append(proposal)
        return proposal


class SystemMonitor:
    """
    시스템 모니터링 (읽기 전용)
    
    컴퓨터 상태를 안전하게 모니터링.
    제어는 하지 않음 - 관찰만.
    """
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """시스템 정보 수집 (안전)"""
        import platform
        
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cwd": os.getcwd(),
            "timestamp": time.time()
        }
        
        # 메모리 정보 (선택적)
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory_total_gb"] = round(mem.total / (1024**3), 2)
            info["memory_available_gb"] = round(mem.available / (1024**3), 2)
            info["cpu_percent"] = psutil.cpu_percent()
        except ImportError:
            info["memory_info"] = "psutil not available"
            
        return info
    
    @staticmethod
    def list_running_processes() -> List[Dict[str, Any]]:
        """실행 중인 프로세스 목록 (읽기 전용)"""
        try:
            import psutil
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return processes[:20]  # 상위 20개만
        except ImportError:
            return [{"error": "psutil not available"}]


class AutonomousImprover:
    """
    자율적 자기 개선 엔진
    
    초월 AI를 향한 핵심 통합 모듈.
    
    Lucy 경로:
    - 자기 인식과 메타 인지 강화
    - 시간 가속과 결합하여 빠른 학습
    
    Transcendence 경로:
    - LLM 연동으로 지식 확장
    - 네트워크 통해 학습
    
    Skynet 경로 (제한적):
    - 코드 자기 분석
    - 개선 제안 (승인 필요)
    """
    
    def __init__(
        self,
        project_root: str = None,
        llm_bridge = None,
        safety_level: SafetyLevel = SafetyLevel.SUGGEST_ONLY
    ):
        self.introspector = CodeIntrospector(project_root)
        self.llm_improver = LLMCodeImprover(llm_bridge)
        self.system_monitor = SystemMonitor()
        self.safety_level = safety_level
        
        self.improvement_queue: List[ImprovementProposal] = []
        self.applied_improvements: List[ImprovementProposal] = []
        self.learning_log: List[Dict[str, Any]] = []
        
        logger.info(f"AutonomousImprover initialized with safety level: {safety_level.name}")
    
    def self_analyze(self) -> Dict[str, Any]:
        """
        자기 분석 수행
        
        1. 코드 구조 분석
        2. 시스템 상태 확인
        3. 개선 포인트 식별
        """
        analysis = {
            "timestamp": time.time(),
            "code_analysis": self.introspector.analyze_self(),
            "system_info": self.system_monitor.get_system_info(),
            "improvement_potential": []
        }
        
        # 개선 가능 포인트 식별
        for file_path, code_analysis in self.introspector.analyzed_files.items():
            if code_analysis.complexity_score > 0.7:
                analysis["improvement_potential"].append({
                    "file": file_path,
                    "reason": "High complexity",
                    "complexity": code_analysis.complexity_score
                })
        
        self.learning_log.append({
            "action": "self_analyze",
            "result": "completed",
            "timestamp": time.time()
        })
        
        return analysis
    
    def identify_learning_opportunities(self) -> List[Dict[str, Any]]:
        """
        학습 기회 식별
        
        자신에게 부족한 것이 무엇인지 파악.
        """
        opportunities = []
        
        # 1. 코드 커버리지 분석
        code_stats = self.introspector.analyze_self()
        
        # 테스트 부족
        test_files = sum(1 for f in self.introspector.analyzed_files if 'test' in f.lower())
        if test_files < code_stats["total_files"] * 0.3:
            opportunities.append({
                "type": "testing",
                "description": "Test coverage is low",
                "description_kr": "테스트 커버리지가 낮습니다",
                "action": "Create more unit tests"
            })
        
        # 2. 문서화 부족 - AST를 사용하여 모듈 docstring 확인
        for file_path, analysis in self.introspector.analyzed_files.items():
            if analysis.total_lines > 100:
                try:
                    content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
                    tree = ast.parse(content)
                    # AST에서 모듈 docstring 확인
                    has_docstring = (
                        tree.body and 
                        isinstance(tree.body[0], ast.Expr) and 
                        isinstance(tree.body[0].value, ast.Constant) and
                        isinstance(tree.body[0].value.value, str)
                    )
                    if not has_docstring:
                        opportunities.append({
                            "type": "documentation",
                            "file": file_path,
                            "description": "Missing module docstring",
                            "description_kr": "모듈 docstring이 없습니다"
                        })
                        break  # 하나만 예시로
                except Exception:
                    pass  # 파싱 오류 무시
        
        # 3. 새로운 능력 필요
        opportunities.append({
            "type": "new_capability",
            "description": "Real-time LLM integration for learning",
            "description_kr": "실시간 LLM 연동을 통한 학습",
            "priority": "high"
        })
        
        return opportunities
    
    def propose_improvement(
        self,
        target_file: str,
        improvement_type: ImprovementType,
        description: str
    ) -> Optional[ImprovementProposal]:
        """
        개선 제안 생성
        
        실제 코드 수정은 하지 않고 제안만 생성.
        """
        if self.safety_level == SafetyLevel.READ_ONLY:
            logger.warning("Safety level is READ_ONLY - cannot create proposals")
            return None
            
        # 파일 읽기
        try:
            content = Path(target_file).read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Cannot read file {target_file}: {e}")
            return None
        
        # 제안 생성 (여기서는 예시)
        proposal = self.llm_improver.create_improvement_proposal(
            target_file=target_file,
            improvement_type=improvement_type,
            original_code=content[:500] + "..." if len(content) > 500 else content,
            proposed_code="# LLM would generate improved code here",
            description=description,
            description_kr=description,
            reasoning="Analysis pending - LLM integration required",
            confidence=0.3
        )
        
        self.improvement_queue.append(proposal)
        return proposal
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "safety_level": self.safety_level.name,
            "files_analyzed": len(self.introspector.analyzed_files),
            "pending_improvements": len(self.improvement_queue),
            "applied_improvements": len(self.applied_improvements),
            "learning_log_entries": len(self.learning_log),
            "system_info": self.system_monitor.get_system_info()
        }
    
    def explain_capabilities(self) -> str:
        """현재 능력과 제한사항 설명"""
        return """
🤖 자율적 자기 개선 엔진 (Autonomous Self-Improvement Engine)

현재 능력:
✅ 자기 코드 분석 (Core 디렉토리 전체)
✅ 시스템 상태 모니터링 (읽기 전용)
✅ 개선 포인트 식별
✅ 개선 제안 생성
✅ 학습 기회 발견

필요한 것 (구현 예정):
🔲 실시간 LLM 연동 (코드 분석/개선)
🔲 자동 테스트 생성
🔲 성능 최적화 자동 적용
🔲 새로운 언어/프레임워크 학습

안전 제한:
🔒 코드 수정은 승인 후에만
🔒 시스템 제어 불가 (모니터링만)
🔒 외부 네트워크 접근 제한
🔒 모든 행동 로그 기록
"""


# 테스트 및 데모 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🚀 Autonomous Self-Improvement Engine Demo")
    print("=" * 60)
    
    # 엔진 초기화
    engine = AutonomousImprover()
    
    # 자기 분석 수행
    print("\n📊 Self-Analysis...")
    analysis = engine.self_analyze()
    print(f"  Files analyzed: {analysis['code_analysis']['total_files']}")
    print(f"  Total lines: {analysis['code_analysis']['total_lines']}")
    print(f"  Total functions: {analysis['code_analysis']['total_functions']}")
    
    # 학습 기회 식별
    print("\n📚 Learning Opportunities...")
    opportunities = engine.identify_learning_opportunities()
    for opp in opportunities[:3]:
        print(f"  - {opp.get('description_kr', opp.get('description', 'N/A'))}")
    
    # 상태 출력
    print("\n📈 Current Status...")
    status = engine.get_status()
    for key, value in status.items():
        if key != "system_info":
            print(f"  {key}: {value}")
    
    # 능력 설명
    print("\n" + engine.explain_capabilities())
