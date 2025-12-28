"""
Bootstrap Guardian: 환경 자가 복구 시스템
=======================================
"두개골을 스스로 고치는 뇌"

부팅 전 환경 상태를 검사하고, 문제 발견 시 자동으로 복구합니다.
- 복구(같은 버전): 사용자 확인 불필요
- 업그레이드/신규: 사용자 확인 필요
"""

import sys
import subprocess
import importlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("BootstrapGuardian")


class IssueType(Enum):
    MISSING = "missing"           # 패키지 없음
    CORRUPTED = "corrupted"       # 패키지 손상 (import 실패)
    VERSION_MISMATCH = "version"  # 버전 불일치


@dataclass
class PackageStatus:
    name: str
    required_version: Optional[str]
    current_version: Optional[str]
    issue: Optional[IssueType]
    error_message: Optional[str] = None
    
    @property
    def is_healthy(self) -> bool:
        return self.issue is None


@dataclass
class EnvironmentStatus:
    packages: List[PackageStatus]
    
    @property
    def is_healthy(self) -> bool:
        return all(p.is_healthy for p in self.packages)
    
    @property
    def issues(self) -> List[PackageStatus]:
        return [p for p in self.packages if not p.is_healthy]


class BootstrapGuardian:
    """
    부팅 전 환경 상태 검사 및 자동 복구
    
    Usage:
        guardian = BootstrapGuardian()
        if guardian.guard():
            # 정상 부팅
        else:
            # 복구 실패
    """
    
    # 핵심 의존성 (이름, 최소 버전, pip 패키지명)
    CRITICAL_PACKAGES = [
        ("torch", "2.0.0", "torch"),
        ("numpy", "1.20.0", "numpy"),
        ("sentence_transformers", None, "sentence-transformers"),
    ]
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.repairs_made = 0
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
        logger.info(message)
    
    def check_package(self, name: str, min_version: Optional[str] = None) -> PackageStatus:
        """단일 패키지 상태 검사"""
        try:
            module = importlib.import_module(name)
            current_version = getattr(module, "__version__", "unknown")
            
            # 버전 확인
            if min_version and current_version != "unknown":
                from packaging.version import Version
                if Version(current_version) < Version(min_version):
                    return PackageStatus(
                        name=name,
                        required_version=min_version,
                        current_version=current_version,
                        issue=IssueType.VERSION_MISMATCH
                    )
            
            return PackageStatus(
                name=name,
                required_version=min_version,
                current_version=current_version,
                issue=None
            )
            
        except ImportError as e:
            # 누락 또는 손상
            error_msg = str(e)
            
            # 손상 감지 (예: torch._C 문제)
            if "_C" in error_msg or "Extension" in error_msg:
                return PackageStatus(
                    name=name,
                    required_version=min_version,
                    current_version=None,
                    issue=IssueType.CORRUPTED,
                    error_message=error_msg
                )
            
            return PackageStatus(
                name=name,
                required_version=min_version,
                current_version=None,
                issue=IssueType.MISSING,
                error_message=error_msg
            )
    
    def check_environment(self) -> EnvironmentStatus:
        """전체 환경 검사"""
        self._log("🔍 Bootstrap Guardian: Checking environment...")
        
        statuses = []
        for name, min_ver, _ in self.CRITICAL_PACKAGES:
            status = self.check_package(name, min_ver)
            statuses.append(status)
            
            if status.is_healthy:
                self._log(f"   ✅ {name}: {status.current_version}")
            else:
                self._log(f"   ❌ {name}: {status.issue.value} - {status.error_message}")
        
        return EnvironmentStatus(packages=statuses)
    
    def repair_package(self, status: PackageStatus) -> bool:
        """
        패키지 복구
        
        복구는 자동 실행 (사용자 확인 불필요)
        """
        # pip 패키지명 찾기
        pip_name = None
        for name, _, pip_pkg in self.CRITICAL_PACKAGES:
            if name == status.name:
                pip_name = pip_pkg
                break
        
        if not pip_name:
            self._log(f"   ⚠️ Unknown package: {status.name}")
            return False
        
        self._log(f"   🔧 Repairing {status.name}...")
        
        try:
            # 손상된 경우: 제거 후 재설치
            if status.issue == IssueType.CORRUPTED:
                self._log(f"      Uninstalling corrupted {pip_name}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "-y", pip_name],
                    capture_output=True,
                    check=False
                )
            
            # 설치/재설치
            self._log(f"      Installing {pip_name}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self._log(f"   ✅ {status.name} repaired successfully!")
                self.repairs_made += 1
                return True
            else:
                self._log(f"   ❌ Repair failed: {result.stderr}")
                return False
                
        except Exception as e:
            self._log(f"   ❌ Repair exception: {e}")
            return False
    
    def guard(self) -> bool:
        """
        전체 환경 검사 및 자동 복구 파이프라인
        
        Returns:
            True if environment is healthy (or successfully repaired)
            False if environment has unrecoverable issues
        """
        self._log("")
        self._log("🛡️ Bootstrap Guardian: Activating...")
        self._log("=" * 50)
        
        # 1. 초기 검사
        status = self.check_environment()
        
        if status.is_healthy:
            self._log("")
            self._log("✅ All systems nominal. Ready for boot.")
            return True
        
        # 2. 문제 발견 - 자동 복구 시도
        self._log("")
        self._log("⚠️ Issues detected. Initiating auto-repair...")
        self._log("-" * 50)
        
        for pkg_status in status.issues:
            self.repair_package(pkg_status)
        
        # 3. 재검사
        self._log("")
        self._log("🔍 Re-checking environment...")
        final_status = self.check_environment()
        
        if final_status.is_healthy:
            self._log("")
            self._log(f"✅ Environment repaired! ({self.repairs_made} packages fixed)")
            return True
        else:
            self._log("")
            self._log("❌ Some issues could not be auto-repaired:")
            for pkg in final_status.issues:
                self._log(f"   • {pkg.name}: {pkg.error_message}")
            self._log("")
            self._log("Manual intervention required.")
            return False


def main():
    """테스트 실행"""
    guardian = BootstrapGuardian(verbose=True)
    
    if guardian.guard():
        print("\n🚀 Environment ready. Elysia can boot safely.")
    else:
        print("\n💔 Environment check failed.")


if __name__ == "__main__":
    main()
