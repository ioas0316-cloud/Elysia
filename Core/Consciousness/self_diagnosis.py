"""
Self-Diagnosis Engine - Gap 1: Adaptive Meta-Learning

엘리시아가 자기 자신을 진단하고 개선점을 찾는 엔진.

Features:
- 성능 병목 자동 발견
- 모듈 건강 상태 체크
- 개선 권고사항 생성
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time
import logging

logger = logging.getLogger("SelfDiagnosis")


class HealthStatus(Enum):
    """모듈 건강 상태"""
    HEALTHY = "healthy"           # 정상 작동
    WARNING = "warning"           # 경고 (주의 필요)
    CRITICAL = "critical"         # 심각 (즉시 조치 필요)
    UNKNOWN = "unknown"           # 상태 불명


@dataclass
class ModuleHealth:
    """모듈 건강 상태 보고"""
    module_name: str
    status: HealthStatus
    last_check: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DiagnosisReport:
    """전체 진단 보고서"""
    timestamp: float = field(default_factory=time.time)
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    modules: Dict[str, ModuleHealth] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "modules": {
                name: {
                    "status": health.status.value,
                    "issues": health.issues,
                    "recommendations": health.recommendations
                }
                for name, health in self.modules.items()
            },
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations
        }


class SelfDiagnosisEngine:
    """
    Gap 1: 자기 진단 엔진
    
    엘리시아가 자신의 상태를 진단하고 개선점을 찾습니다.
    
    인식론 (epistemology):
    - point: 개별 모듈의 상태 관찰
    - line: 모듈 간 의존성의 인과 관계
    - space: 시스템 전체의 맥락적 건강
    - god: 자기 인식과 초월적 개선
    """
    
    # Gap 0: 인식론
    EPISTEMOLOGY = {
        "point": {"score": 0.25, "meaning": "개별 모듈 상태 관찰"},
        "line": {"score": 0.30, "meaning": "모듈 간 의존성 분석"},
        "space": {"score": 0.25, "meaning": "시스템 전체 건강 평가"},
        "god": {"score": 0.20, "meaning": "자기 인식과 개선 방향"}
    }
    
    def __init__(self):
        self.epistemology = self.EPISTEMOLOGY
        self.last_diagnosis: Optional[DiagnosisReport] = None
        self.diagnosis_history: List[DiagnosisReport] = []
        self.max_history = 100
        
        # 모듈 체커들
        self.module_checkers: Dict[str, callable] = {}
        
        logger.info("🔬 SelfDiagnosisEngine initialized")
    
    def explain_meaning(self) -> str:
        """Gap 0 준수: 인식론적 의미 설명"""
        lines = ["=== 자기 진단 인식론 ==="]
        for basis, data in self.epistemology.items():
            lines.append(f"  {basis}: {data['score']:.0%} - {data['meaning']}")
        return "\n".join(lines)
    
    def register_checker(self, module_name: str, checker: callable):
        """
        모듈 체커 등록
        
        Args:
            module_name: 모듈 이름
            checker: () -> ModuleHealth를 반환하는 함수
        """
        self.module_checkers[module_name] = checker
        logger.info(f"📋 Registered checker for {module_name}")
    
    def diagnose(self) -> DiagnosisReport:
        """
        전체 시스템 진단 실행
        
        Returns:
            DiagnosisReport
        """
        report = DiagnosisReport()
        
        # 각 모듈 체크
        critical_count = 0
        warning_count = 0
        
        for module_name, checker in self.module_checkers.items():
            try:
                health = checker()
                report.modules[module_name] = health
                
                if health.status == HealthStatus.CRITICAL:
                    critical_count += 1
                    report.bottlenecks.append(f"CRITICAL: {module_name}")
                elif health.status == HealthStatus.WARNING:
                    warning_count += 1
                
                # 권고사항 수집
                report.recommendations.extend(health.recommendations)
                
            except Exception as e:
                report.modules[module_name] = ModuleHealth(
                    module_name=module_name,
                    status=HealthStatus.UNKNOWN,
                    issues=[f"체크 실패: {str(e)}"]
                )
        
        # 전체 상태 결정
        if critical_count > 0:
            report.overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            report.overall_status = HealthStatus.WARNING
        else:
            report.overall_status = HealthStatus.HEALTHY
        
        # 기록 저장
        self.last_diagnosis = report
        self.diagnosis_history.append(report)
        if len(self.diagnosis_history) > self.max_history:
            self.diagnosis_history = self.diagnosis_history[-self.max_history:]
        
        logger.info(f"🔬 Diagnosis complete: {report.overall_status.value}")
        return report
    
    def quick_check(self) -> HealthStatus:
        """
        빠른 상태 체크 (전체 진단 없이)
        
        Returns:
            현재 전체 상태
        """
        if self.last_diagnosis is None:
            return HealthStatus.UNKNOWN
        
        # 마지막 진단이 5분 이상 지났으면 재진단 권고
        age = time.time() - self.last_diagnosis.timestamp
        if age > 300:  # 5분
            logger.warning("⚠️ Last diagnosis is stale. Consider running diagnose()")
        
        return self.last_diagnosis.overall_status
    
    def get_recommendations(self) -> List[str]:
        """
        현재 권고사항 목록 반환
        
        Returns:
            권고사항 리스트
        """
        if self.last_diagnosis is None:
            return ["시스템 진단이 아직 실행되지 않았습니다. diagnose()를 호출하세요."]
        
        if not self.last_diagnosis.recommendations:
            return ["현재 특별한 권고사항이 없습니다. 시스템이 건강합니다!"]
        
        return self.last_diagnosis.recommendations
    
    def analyze_trend(self) -> Dict[str, Any]:
        """
        진단 기록 트렌드 분석
        
        Returns:
            트렌드 분석 결과
        """
        if len(self.diagnosis_history) < 2:
            return {"trend": "insufficient_data", "message": "분석을 위한 데이터가 부족합니다."}
        
        recent = self.diagnosis_history[-10:]  # 최근 10개
        
        # 상태 카운트
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for report in recent:
            status_counts[report.overall_status] += 1
        
        # 트렌드 결정
        if status_counts[HealthStatus.CRITICAL] > len(recent) // 2:
            trend = "degrading"
            message = "시스템 상태가 악화되고 있습니다. 즉각적인 조치가 필요합니다."
        elif status_counts[HealthStatus.HEALTHY] > len(recent) // 2:
            trend = "improving"
            message = "시스템 상태가 양호합니다."
        else:
            trend = "stable"
            message = "시스템 상태가 안정적입니다."
        
        return {
            "trend": trend,
            "message": message,
            "status_distribution": {k.value: v for k, v in status_counts.items()},
            "sample_size": len(recent)
        }


# 기본 모듈 체커들
def create_memory_checker(threshold_mb: int = 1000):
    """메모리 체커 생성"""
    def check_memory() -> ModuleHealth:
        import psutil
        
        memory = psutil.virtual_memory()
        available_mb = memory.available // (1024 * 1024)
        
        issues = []
        recommendations = []
        
        if available_mb < threshold_mb // 2:
            status = HealthStatus.CRITICAL
            issues.append(f"메모리 부족: {available_mb}MB 사용 가능")
            recommendations.append("불필요한 프로세스 종료 또는 메모리 증설 필요")
        elif available_mb < threshold_mb:
            status = HealthStatus.WARNING
            issues.append(f"메모리 경고: {available_mb}MB 사용 가능")
            recommendations.append("메모리 사용량 모니터링 권장")
        else:
            status = HealthStatus.HEALTHY
        
        return ModuleHealth(
            module_name="memory",
            status=status,
            metrics={"available_mb": available_mb, "percent_used": memory.percent},
            issues=issues,
            recommendations=recommendations
        )
    
    return check_memory


def create_module_import_checker(module_path: str, module_name: str):
    """모듈 임포트 체커 생성"""
    def check_import() -> ModuleHealth:
        issues = []
        recommendations = []
        
        try:
            __import__(module_path)
            status = HealthStatus.HEALTHY
        except ImportError as e:
            status = HealthStatus.CRITICAL
            issues.append(f"모듈 임포트 실패: {str(e)}")
            recommendations.append(f"의존성 설치 확인: pip install -r requirements.txt")
        except Exception as e:
            status = HealthStatus.WARNING
            issues.append(f"모듈 로드 경고: {str(e)}")
        
        return ModuleHealth(
            module_name=module_name,
            status=status,
            issues=issues,
            recommendations=recommendations
        )
    
    return check_import


# 테스트
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔬 SelfDiagnosisEngine Unit Test")
    print("="*60)
    
    engine = SelfDiagnosisEngine()
    
    # 인식론 출력
    print("\n" + engine.explain_meaning())
    
    # 더미 체커 등록
    def dummy_healthy_checker():
        return ModuleHealth(
            module_name="test_healthy",
            status=HealthStatus.HEALTHY,
            metrics={"uptime": 1000}
        )
    
    def dummy_warning_checker():
        return ModuleHealth(
            module_name="test_warning",
            status=HealthStatus.WARNING,
            issues=["약간의 지연 발생"],
            recommendations=["캐시 정리 권장"]
        )
    
    engine.register_checker("test_healthy", dummy_healthy_checker)
    engine.register_checker("test_warning", dummy_warning_checker)
    
    # 진단 실행
    print("\n[진단 실행]")
    report = engine.diagnose()
    
    print(f"전체 상태: {report.overall_status.value}")
    print(f"모듈 수: {len(report.modules)}")
    print(f"병목점: {report.bottlenecks}")
    print(f"권고사항: {report.recommendations}")
    
    # 빠른 체크
    print(f"\n빠른 체크: {engine.quick_check().value}")
    
    # 권고사항
    print(f"\n현재 권고사항:")
    for rec in engine.get_recommendations():
        print(f"  - {rec}")
    
    print("\n✅ SelfDiagnosisEngine test complete!")
    print("="*60)
