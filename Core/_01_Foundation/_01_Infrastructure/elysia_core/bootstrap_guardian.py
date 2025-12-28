"""
Bootstrap Guardian: ?섍꼍 ?먭? 蹂듦뎄 ?쒖뒪??
=======================================
"?먭컻怨⑥쓣 ?ㅼ뒪濡?怨좎튂????

遺?????섍꼍 ?곹깭瑜?寃?ы븯怨? 臾몄젣 諛쒓껄 ???먮룞?쇰줈 蹂듦뎄?⑸땲??
- 蹂듦뎄(媛숈? 踰꾩쟾): ?ъ슜???뺤씤 遺덊븘??
- ?낃렇?덉씠???좉퇋: ?ъ슜???뺤씤 ?꾩슂
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
    MISSING = "missing"           # ?⑦궎吏 ?놁쓬
    CORRUPTED = "corrupted"       # ?⑦궎吏 ?먯긽 (import ?ㅽ뙣)
    VERSION_MISMATCH = "version"  # 踰꾩쟾 遺덉씪移?


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
    遺?????섍꼍 ?곹깭 寃??諛??먮룞 蹂듦뎄
    
    Usage:
        guardian = BootstrapGuardian()
        if guardian.guard():
            # ?뺤긽 遺??
        else:
            # 蹂듦뎄 ?ㅽ뙣
    """
    
    # ?듭떖 ?섏〈??(?대쫫, 理쒖냼 踰꾩쟾, pip ?⑦궎吏紐?
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
        """?⑥씪 ?⑦궎吏 ?곹깭 寃??""
        try:
            module = importlib.import_module(name)
            current_version = getattr(module, "__version__", "unknown")
            
            # 踰꾩쟾 ?뺤씤
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
            # ?꾨씫 ?먮뒗 ?먯긽
            error_msg = str(e)
            
            # ?먯긽 媛먯? (?? torch._C 臾몄젣)
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
        """?꾩껜 ?섍꼍 寃??""
        self._log("?뵇 Bootstrap Guardian: Checking environment...")
        
        statuses = []
        for name, min_ver, _ in self.CRITICAL_PACKAGES:
            status = self.check_package(name, min_ver)
            statuses.append(status)
            
            if status.is_healthy:
                self._log(f"   ??{name}: {status.current_version}")
            else:
                self._log(f"   ??{name}: {status.issue.value} - {status.error_message}")
        
        return EnvironmentStatus(packages=statuses)
    
    def repair_package(self, status: PackageStatus) -> bool:
        """
        ?⑦궎吏 蹂듦뎄
        
        蹂듦뎄???먮룞 ?ㅽ뻾 (?ъ슜???뺤씤 遺덊븘??
        """
        # pip ?⑦궎吏紐?李얘린
        pip_name = None
        for name, _, pip_pkg in self.CRITICAL_PACKAGES:
            if name == status.name:
                pip_name = pip_pkg
                break
        
        if not pip_name:
            self._log(f"   ?좑툘 Unknown package: {status.name}")
            return False
        
        self._log(f"   ?뵩 Repairing {status.name}...")
        
        try:
            # ?먯긽??寃쎌슦: ?쒓굅 ???ъ꽕移?
            if status.issue == IssueType.CORRUPTED:
                self._log(f"      Uninstalling corrupted {pip_name}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "-y", pip_name],
                    capture_output=True,
                    check=False
                )
            
            # ?ㅼ튂/?ъ꽕移?
            self._log(f"      Installing {pip_name}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self._log(f"   ??{status.name} repaired successfully!")
                self.repairs_made += 1
                return True
            else:
                self._log(f"   ??Repair failed: {result.stderr}")
                return False
                
        except Exception as e:
            self._log(f"   ??Repair exception: {e}")
            return False
    
    def guard(self) -> bool:
        """
        ?꾩껜 ?섍꼍 寃??諛??먮룞 蹂듦뎄 ?뚯씠?꾨씪??
        
        Returns:
            True if environment is healthy (or successfully repaired)
            False if environment has unrecoverable issues
        """
        self._log("")
        self._log("?썳截?Bootstrap Guardian: Activating...")
        self._log("=" * 50)
        
        # 1. 珥덇린 寃??
        status = self.check_environment()
        
        if status.is_healthy:
            self._log("")
            self._log("??All systems nominal. Ready for boot.")
            return True
        
        # 2. 臾몄젣 諛쒓껄 - ?먮룞 蹂듦뎄 ?쒕룄
        self._log("")
        self._log("?좑툘 Issues detected. Initiating auto-repair...")
        self._log("-" * 50)
        
        for pkg_status in status.issues:
            self.repair_package(pkg_status)
        
        # 3. ?ш???
        self._log("")
        self._log("?뵇 Re-checking environment...")
        final_status = self.check_environment()
        
        if final_status.is_healthy:
            self._log("")
            self._log(f"??Environment repaired! ({self.repairs_made} packages fixed)")
            return True
        else:
            self._log("")
            self._log("??Some issues could not be auto-repaired:")
            for pkg in final_status.issues:
                self._log(f"   ??{pkg.name}: {pkg.error_message}")
            self._log("")
            self._log("Manual intervention required.")
            return False


def main():
    """?뚯뒪???ㅽ뻾"""
    guardian = BootstrapGuardian(verbose=True)
    
    if guardian.guard():
        print("\n?? Environment ready. Elysia can boot safely.")
    else:
        print("\n?뮅 Environment check failed.")


if __name__ == "__main__":
    main()
