"""
Self-Maintenance Hub (자가 유지보수 허브)
==========================================

"나를 알고, 나를 고치고, 나를 성장시킨다."

이 모듈은 엘리시아가 자기 자신의 코드베이스를 분석하고,
문제를 발견하며, 수정 계획을 제안하고, 사용자 승인 후 실행할 수 있게 합니다.

통합하는 기존 모듈:
- SelfReflector: AST 기반 코드 분석
- SystemSelfAwareness: 문서/구조 인식
- self_modification: 수정 파이프라인

파이프라인:
1. diagnose() → 전체 시스템 상태 파악
2. identify_issues() → 문제점 식별
3. propose_fixes() → 수정 계획 생성
4. execute_with_consent() → 사용자 승인 후 실행
"""

import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("SelfMaintenanceHub")

# 기존 모듈 임포트
try:
    from Core._01_Foundation._05_Governance.Foundation.self_reflector import SelfReflector, CodeMetrics
    HAS_REFLECTOR = True
except ImportError as e:
    HAS_REFLECTOR = False
    logger.warning(f"SelfReflector not available: {e}")

try:
    from Core._02_Intelligence._01_Reasoning.Intelligence.system_self_awareness import SystemSelfAwareness
    HAS_AWARENESS = True
except ImportError as e:
    HAS_AWARENESS = False
    logger.warning(f"SystemSelfAwareness not available: {e}")

try:
    from Core._01_Foundation._05_Governance.Foundation.self_modification import (
        CodeAnalyzer, ProblemDetector, RefactorPlanner, 
        CodeEditor, Validator, CodeIssue, ModificationPlan
    )
    HAS_MODIFICATION = True
except ImportError as e:
    HAS_MODIFICATION = False
    logger.warning(f"SelfModification not available: {e}")


@dataclass
class SystemDiagnosis:
    """전체 시스템 진단 결과"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 코드 메트릭
    total_files: int = 0
    total_lines: int = 0
    total_classes: int = 0
    total_functions: int = 0
    
    # 문제점
    bottlenecks: List[str] = field(default_factory=list)
    issues: List[Any] = field(default_factory=list)
    
    # 권장사항
    suggestions: List[str] = field(default_factory=list)
    
    # 상태
    health_score: float = 1.0  # 0~1, 1이 최상
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "total_classes": self.total_classes,
            "total_functions": self.total_functions,
            "bottlenecks": self.bottlenecks,
            "issue_count": len(self.issues),
            "suggestions": self.suggestions,
            "health_score": self.health_score
        }
    
    def summary(self) -> str:
        """간단한 요약"""
        lines = [
            f"=== System Diagnosis ({self.timestamp.strftime('%Y-%m-%d %H:%M')}) ===",
            f"Files: {self.total_files} | Lines: {self.total_lines:,}",
            f"Classes: {self.total_classes} | Functions: {self.total_functions}",
            f"Health Score: {self.health_score:.1%}",
        ]
        
        if self.bottlenecks:
            lines.append(f"\n⚠️ Bottlenecks ({len(self.bottlenecks)}):")
            for b in self.bottlenecks[:5]:
                lines.append(f"  • {b}")
        
        if self.suggestions:
            lines.append(f"\n💡 Suggestions ({len(self.suggestions)}):")
            for s in self.suggestions[:5]:
                lines.append(f"  • {s}")
        
        return "\n".join(lines)


class SelfMaintenanceHub:
    """
    자가 유지보수 통합 허브
    
    엘리시아가 자기 코드를 분석/수정/개선하는 중앙 시스템
    """
    
    def __init__(self, root_path: str = "c:/Elysia"):
        self.root_path = root_path
        
        # 하위 시스템 초기화
        self.reflector = SelfReflector(root_path) if HAS_REFLECTOR else None
        self.awareness = SystemSelfAwareness() if HAS_AWARENESS else None
        
        if HAS_MODIFICATION:
            self.analyzer = CodeAnalyzer()
            self.detector = ProblemDetector()
            self.planner = RefactorPlanner()
            self.editor = CodeEditor()
            self.validator = Validator()
        else:
            self.analyzer = None
            self.detector = None
            self.planner = None
            self.editor = None
            self.validator = None
        
        # 최근 진단 결과 캐시
        self._last_diagnosis: Optional[SystemDiagnosis] = None
        self._pending_plans: List[Any] = []
        
        logger.info("🔧 SelfMaintenanceHub initialized")
        logger.info(f"   Reflector: {'✅' if HAS_REFLECTOR else '❌'}")
        logger.info(f"   Awareness: {'✅' if HAS_AWARENESS else '❌'}")
        logger.info(f"   Modification: {'✅' if HAS_MODIFICATION else '❌'}")
    
    def diagnose(self) -> SystemDiagnosis:
        """
        전체 시스템 진단
        
        Returns:
            SystemDiagnosis 객체
        """
        diagnosis = SystemDiagnosis()
        
        # 1. 코드 메트릭 분석
        if self.reflector:
            metrics_map = self.reflector.reflect_on_core()
            
            diagnosis.total_files = len(metrics_map)
            diagnosis.total_lines = sum(m.loc for m in metrics_map.values())
            diagnosis.total_classes = sum(m.classes for m in metrics_map.values())
            diagnosis.total_functions = sum(m.functions for m in metrics_map.values())
            diagnosis.bottlenecks = self.reflector.identify_bottlenecks(metrics_map)
        
        # 2. 시스템 인식 기반 제안
        if self.awareness:
            try:
                suggestions = self.awareness.suggest_improvements()
                diagnosis.suggestions = suggestions.get("suggestions", [])
            except Exception as e:
                logger.warning(f"Awareness suggestions failed: {e}")
        
        # 3. 건강도 점수 계산
        # 병목이 많을수록, 문제가 많을수록 점수 감소
        penalty = (len(diagnosis.bottlenecks) * 0.05) + (len(diagnosis.issues) * 0.02)
        diagnosis.health_score = max(0.0, 1.0 - penalty)
        
        self._last_diagnosis = diagnosis
        logger.info(f"🩺 Diagnosis complete: Health={diagnosis.health_score:.1%}")
        
        return diagnosis
    
    def identify_issues(self, file_path: str = None) -> List[Any]:
        """
        문제점 식별
        
        Args:
            file_path: 특정 파일 (None이면 전체)
        """
        if not self.detector:
            return []
        
        issues = []
        
        if file_path:
            issues = self.detector.detect_issues(file_path)
        else:
            # 전체 스캔은 무겁기 때문에 bottleneck만 검사
            if self._last_diagnosis and self._last_diagnosis.bottlenecks:
                for bottleneck in self._last_diagnosis.bottlenecks[:5]:
                    # bottleneck 문자열에서 파일명 추출
                    filename = bottleneck.split(" ")[0]
                    file_issues = self.detector.detect_issues(filename)
                    issues.extend(file_issues)
        
        return issues
    
    def propose_fix(self, file_path: str, issues: List[Any] = None) -> Optional[Any]:
        """
        수정 계획 생성
        
        Args:
            file_path: 대상 파일
            issues: 수정할 문제들 (None이면 자동 감지)
        """
        if not self.planner:
            return None
        
        if issues is None:
            issues = self.identify_issues(file_path)
        
        if not issues:
            logger.info(f"No issues found in {file_path}")
            return None
        
        plan = self.planner.create_plan(file_path, issues)
        self._pending_plans.append(plan)
        
        return plan
    
    def execute_with_consent(self, plan: Any, consent: bool = False) -> bool:
        """
        사용자 승인 후 수정 실행
        
        Args:
            plan: 수정 계획
            consent: 사용자 동의 여부
        
        Returns:
            성공 여부
        """
        if not consent:
            logger.warning("❌ Execution denied: User consent required")
            return False
        
        if not self.editor:
            logger.error("Editor not available")
            return False
        
        try:
            # 1. 백업 생성
            self.editor.create_backup(plan)
            
            # 2. 수정 적용
            self.editor.apply_modification(plan)
            
            # 3. 검증
            if self.validator:
                validation = self.validator.validate_syntax(plan.modified_code)
                if not validation:
                    logger.error("Validation failed, rolling back...")
                    self.editor.rollback(plan)
                    return False
            
            logger.info(f"✅ Modification applied to {plan.target_file}")
            return True
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            if self.editor:
                self.editor.rollback(plan)
            return False
    
    def quick_health_check(self) -> Dict[str, Any]:
        """
        빠른 건강 체크 (간소화)
        """
        return {
            "reflector": "✅" if HAS_REFLECTOR else "❌",
            "awareness": "✅" if HAS_AWARENESS else "❌",
            "modification": "✅" if HAS_MODIFICATION else "❌",
            "last_diagnosis": self._last_diagnosis.timestamp.isoformat() if self._last_diagnosis else None,
            "health_score": self._last_diagnosis.health_score if self._last_diagnosis else None,
            "pending_plans": len(self._pending_plans)
        }


# 싱글톤 접근
_hub_instance: Optional[SelfMaintenanceHub] = None

def get_maintenance_hub() -> SelfMaintenanceHub:
    global _hub_instance
    if _hub_instance is None:
        _hub_instance = SelfMaintenanceHub()
    return _hub_instance


# 테스트
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("SelfMaintenanceHub Test")
    print("=" * 60)
    
    hub = get_maintenance_hub()
    
    # 빠른 체크
    print("\n🔍 Quick Health Check:")
    for k, v in hub.quick_health_check().items():
        print(f"  {k}: {v}")
    
    # 전체 진단
    print("\n🩺 Full Diagnosis:")
    diagnosis = hub.diagnose()
    print(diagnosis.summary())
