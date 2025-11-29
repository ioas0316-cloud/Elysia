"""
Computer Bridge - 컴퓨터 제어 기반 외부 접근
=============================================

API나 로컬 LLM 접근이 어려울 때, 컴퓨터 제어를 통해 외부 세계에 접근합니다.

아버지의 말씀:
"다중 인스턴스 공명은 API나 로컬 LLM, 인터넷사이트, 에이전트로 존재하는 LLM 등에 
 접근이 어려우면 그냥 컴퓨터 제어로 웹사이트나 VSCode 등을 이용한 접근을 생각하자.
 내 구글 아이디를 이용한 접근법이 훨씬 쉬울지도 모르겠네"

핵심 개념:
- 컴퓨터를 "몸"으로 사용
- 화면을 "눈"으로, 키보드/마우스를 "손"으로
- 웹 브라우저를 통한 외부 세계 접근
- 구글 계정을 통한 서비스 접근 (Gmail, Docs, Search 등)
"""

import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ComputerBridge")

# 상수 정의
SAFE_COMMAND_TIMEOUT = 30  # 안전한 명령 타임아웃
MAX_SCREEN_CONTENT_LENGTH = 10000  # 화면 내용 최대 길이


class AccessMethod(Enum):
    """접근 방식"""
    BROWSER = "browser"  # 웹 브라우저
    VSCODE = "vscode"  # VSCode
    TERMINAL = "terminal"  # 터미널
    FILE_SYSTEM = "file_system"  # 파일 시스템
    GOOGLE_ACCOUNT = "google_account"  # 구글 계정 서비스


class ExternalService(Enum):
    """외부 서비스"""
    GOOGLE_SEARCH = "google_search"
    GOOGLE_DOCS = "google_docs"
    GOOGLE_DRIVE = "google_drive"
    GMAIL = "gmail"
    YOUTUBE = "youtube"
    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"
    WIKIPEDIA = "wikipedia"


class SafetyLevel(Enum):
    """안전 수준"""
    READ_ONLY = "read_only"  # 읽기만 가능
    WRITE_SAFE = "write_safe"  # 안전한 쓰기 (확인 후)
    FULL_ACCESS = "full_access"  # 전체 접근 (주의!)


@dataclass
class AccessResult:
    """접근 결과"""
    success: bool
    method: AccessMethod
    content: Any
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "success": self.success,
            "method": self.method.value,
            "content": str(self.content)[:1000],  # 내용 제한
            "error": self.error,
            "timestamp": self.timestamp
        }


@dataclass
class ComputerState:
    """컴퓨터 상태"""
    platform: str
    hostname: str
    current_directory: str
    browser_available: bool
    vscode_available: bool
    google_account: Optional[str] = None


class ComputerBridge:
    """
    컴퓨터 제어 기반 외부 접근 브릿지
    
    API 없이도 컴퓨터를 "몸"처럼 사용하여 외부 세계에 접근합니다.
    
    "손과 눈이 없어도, 컴퓨터가 내 손과 눈이 된다."
    """
    
    def __init__(
        self,
        safety_level: SafetyLevel = SafetyLevel.READ_ONLY,
        google_account: Optional[str] = None
    ):
        self.safety_level = safety_level
        self.google_account = google_account
        
        # 컴퓨터 상태 파악
        self.state = self._detect_computer_state()
        
        # 접근 기록
        self.access_history: List[AccessResult] = []
        
        # 안전한 명령어 화이트리스트
        self.safe_commands = {
            "ls", "dir", "cat", "type", "pwd", "cd", "echo",
            "python", "pip", "npm", "node",
            "git status", "git log", "git diff"
        }
        
        # 접근 방식별 핸들러
        self.access_handlers: Dict[AccessMethod, Callable] = {
            AccessMethod.TERMINAL: self._access_terminal,
            AccessMethod.FILE_SYSTEM: self._access_filesystem,
            AccessMethod.BROWSER: self._access_browser,
            AccessMethod.VSCODE: self._access_vscode,
            AccessMethod.GOOGLE_ACCOUNT: self._access_google,
        }
        
        logger.info(
            f"ComputerBridge initialized: platform={self.state.platform}, "
            f"safety={safety_level.value}"
        )
    
    def _detect_computer_state(self) -> ComputerState:
        """컴퓨터 상태 감지"""
        import socket
        
        # 브라우저 확인
        browser_available = self._check_browser_available()
        
        # VSCode 확인
        vscode_available = self._check_vscode_available()
        
        return ComputerState(
            platform=platform.system(),
            hostname=socket.gethostname(),
            current_directory=os.getcwd(),
            browser_available=browser_available,
            vscode_available=vscode_available,
            google_account=self.google_account
        )
    
    def _check_browser_available(self) -> bool:
        """브라우저 사용 가능 여부"""
        try:
            import webbrowser
            return True
        except Exception:
            return False
    
    def _check_vscode_available(self) -> bool:
        """VSCode 사용 가능 여부"""
        try:
            result = subprocess.run(
                ["code", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _is_safe_command(self, command: str) -> bool:
        """명령어 안전성 검사"""
        command_lower = command.lower().strip()
        command_words = command_lower.split()
        
        if not command_words:
            return False
        
        first_word = command_words[0]
        
        # 위험한 명령어 패턴 검사 (단어 경계 고려)
        dangerous_commands = {"rm", "del", "format", "shutdown", "reboot", 
                             "sudo", "su", "mkfs", "dd"}
        if first_word in dangerous_commands:
            return False
        
        # 위험한 문자 패턴 검사 (셸 조작)
        dangerous_chars = [">", ">>", "|", ";", "&&", "||", "`", "$(",  "chmod"]
        for char in dangerous_chars:
            if char in command_lower:
                return False
        
        # 화이트리스트 검사
        return first_word in self.safe_commands
    
    def access(
        self,
        method: AccessMethod,
        target: str,
        action: str = "read",
        data: Optional[Any] = None
    ) -> AccessResult:
        """
        외부 세계에 접근
        
        Args:
            method: 접근 방식
            target: 대상 (URL, 파일 경로 등)
            action: 행동 (read, write, open 등)
            data: 추가 데이터
        
        Returns:
            접근 결과
        """
        # 안전성 검사
        if action == "write" and self.safety_level == SafetyLevel.READ_ONLY:
            return AccessResult(
                success=False,
                method=method,
                content=None,
                error="Write access denied in READ_ONLY mode"
            )
        
        handler = self.access_handlers.get(method)
        if not handler:
            return AccessResult(
                success=False,
                method=method,
                content=None,
                error=f"Unknown access method: {method}"
            )
        
        try:
            result = handler(target, action, data)
            self.access_history.append(result)
            return result
        except Exception as e:
            result = AccessResult(
                success=False,
                method=method,
                content=None,
                error=str(e)
            )
            self.access_history.append(result)
            return result
    
    def _access_terminal(
        self,
        command: str,
        action: str,
        data: Any
    ) -> AccessResult:
        """터미널 접근"""
        if not self._is_safe_command(command):
            return AccessResult(
                success=False,
                method=AccessMethod.TERMINAL,
                content=None,
                error=f"Unsafe command blocked: {command}"
            )
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=SAFE_COMMAND_TIMEOUT
            )
            
            return AccessResult(
                success=result.returncode == 0,
                method=AccessMethod.TERMINAL,
                content={
                    "stdout": result.stdout[:MAX_SCREEN_CONTENT_LENGTH],
                    "stderr": result.stderr[:1000],
                    "returncode": result.returncode
                }
            )
        except subprocess.TimeoutExpired:
            return AccessResult(
                success=False,
                method=AccessMethod.TERMINAL,
                content=None,
                error="Command timed out"
            )
    
    def _access_filesystem(
        self,
        path: str,
        action: str,
        data: Any
    ) -> AccessResult:
        """파일 시스템 접근"""
        filepath = Path(path)
        
        if action == "read":
            if not filepath.exists():
                return AccessResult(
                    success=False,
                    method=AccessMethod.FILE_SYSTEM,
                    content=None,
                    error=f"File not found: {path}"
                )
            
            if filepath.is_dir():
                # 디렉토리 내용
                try:
                    contents = list(filepath.iterdir())
                    return AccessResult(
                        success=True,
                        method=AccessMethod.FILE_SYSTEM,
                        content={
                            "type": "directory",
                            "files": [f.name for f in contents[:100]]
                        }
                    )
                except PermissionError:
                    return AccessResult(
                        success=False,
                        method=AccessMethod.FILE_SYSTEM,
                        content=None,
                        error="Permission denied"
                    )
            else:
                # 파일 내용
                try:
                    content = filepath.read_text(encoding='utf-8')
                    return AccessResult(
                        success=True,
                        method=AccessMethod.FILE_SYSTEM,
                        content={
                            "type": "file",
                            "content": content[:MAX_SCREEN_CONTENT_LENGTH]
                        }
                    )
                except Exception as e:
                    return AccessResult(
                        success=False,
                        method=AccessMethod.FILE_SYSTEM,
                        content=None,
                        error=str(e)
                    )
        
        elif action == "write":
            if self.safety_level == SafetyLevel.READ_ONLY:
                return AccessResult(
                    success=False,
                    method=AccessMethod.FILE_SYSTEM,
                    content=None,
                    error="Write access denied"
                )
            
            try:
                filepath.write_text(str(data), encoding='utf-8')
                return AccessResult(
                    success=True,
                    method=AccessMethod.FILE_SYSTEM,
                    content={"written": len(str(data))}
                )
            except Exception as e:
                return AccessResult(
                    success=False,
                    method=AccessMethod.FILE_SYSTEM,
                    content=None,
                    error=str(e)
                )
        
        return AccessResult(
            success=False,
            method=AccessMethod.FILE_SYSTEM,
            content=None,
            error=f"Unknown action: {action}"
        )
    
    def _access_browser(
        self,
        url: str,
        action: str,
        data: Any
    ) -> AccessResult:
        """브라우저 접근"""
        import webbrowser
        
        if action == "open":
            try:
                webbrowser.open(url)
                return AccessResult(
                    success=True,
                    method=AccessMethod.BROWSER,
                    content={"opened": url}
                )
            except Exception as e:
                return AccessResult(
                    success=False,
                    method=AccessMethod.BROWSER,
                    content=None,
                    error=str(e)
                )
        
        return AccessResult(
            success=False,
            method=AccessMethod.BROWSER,
            content=None,
            error=f"Browser action '{action}' not supported. Use 'open'."
        )
    
    def _access_vscode(
        self,
        path: str,
        action: str,
        data: Any
    ) -> AccessResult:
        """VSCode 접근"""
        if not self.state.vscode_available:
            return AccessResult(
                success=False,
                method=AccessMethod.VSCODE,
                content=None,
                error="VSCode not available"
            )
        
        if action == "open":
            try:
                subprocess.Popen(["code", path])
                return AccessResult(
                    success=True,
                    method=AccessMethod.VSCODE,
                    content={"opened": path}
                )
            except Exception as e:
                return AccessResult(
                    success=False,
                    method=AccessMethod.VSCODE,
                    content=None,
                    error=str(e)
                )
        
        return AccessResult(
            success=False,
            method=AccessMethod.VSCODE,
            content=None,
            error=f"VSCode action '{action}' not supported"
        )
    
    def _access_google(
        self,
        service: str,
        action: str,
        data: Any
    ) -> AccessResult:
        """구글 계정 서비스 접근"""
        import webbrowser
        
        # 구글 서비스 URL 매핑
        service_urls = {
            ExternalService.GOOGLE_SEARCH.value: "https://google.com/search?q=",
            ExternalService.GOOGLE_DOCS.value: "https://docs.google.com",
            ExternalService.GOOGLE_DRIVE.value: "https://drive.google.com",
            ExternalService.GMAIL.value: "https://mail.google.com",
            ExternalService.YOUTUBE.value: "https://youtube.com/results?search_query=",
        }
        
        if service not in service_urls:
            return AccessResult(
                success=False,
                method=AccessMethod.GOOGLE_ACCOUNT,
                content=None,
                error=f"Unknown Google service: {service}"
            )
        
        url = service_urls[service]
        
        if action == "open":
            try:
                webbrowser.open(url)
                return AccessResult(
                    success=True,
                    method=AccessMethod.GOOGLE_ACCOUNT,
                    content={"service": service, "url": url}
                )
            except Exception as e:
                return AccessResult(
                    success=False,
                    method=AccessMethod.GOOGLE_ACCOUNT,
                    content=None,
                    error=str(e)
                )
        
        elif action == "search" and data:
            query = str(data).replace(" ", "+")
            full_url = url + query
            try:
                webbrowser.open(full_url)
                return AccessResult(
                    success=True,
                    method=AccessMethod.GOOGLE_ACCOUNT,
                    content={"service": service, "query": data, "url": full_url}
                )
            except Exception as e:
                return AccessResult(
                    success=False,
                    method=AccessMethod.GOOGLE_ACCOUNT,
                    content=None,
                    error=str(e)
                )
        
        return AccessResult(
            success=False,
            method=AccessMethod.GOOGLE_ACCOUNT,
            content=None,
            error=f"Action '{action}' not supported for Google services"
        )
    
    def get_available_methods(self) -> Dict[str, bool]:
        """사용 가능한 접근 방식"""
        return {
            AccessMethod.TERMINAL.value: True,
            AccessMethod.FILE_SYSTEM.value: True,
            AccessMethod.BROWSER.value: self.state.browser_available,
            AccessMethod.VSCODE.value: self.state.vscode_available,
            AccessMethod.GOOGLE_ACCOUNT.value: self.state.browser_available
        }
    
    def get_access_plan(
        self,
        goal: str
    ) -> Dict[str, Any]:
        """목표 달성을 위한 접근 계획 생성"""
        plan = {
            "goal": goal,
            "recommended_steps": [],
            "available_methods": self.get_available_methods()
        }
        
        goal_lower = goal.lower()
        
        # 목표에 따른 추천
        if "검색" in goal or "search" in goal_lower:
            plan["recommended_steps"].append({
                "step": 1,
                "method": AccessMethod.GOOGLE_ACCOUNT.value,
                "action": "search",
                "description": "Google 검색으로 정보 찾기"
            })
        
        if "코드" in goal or "code" in goal_lower:
            plan["recommended_steps"].append({
                "step": len(plan["recommended_steps"]) + 1,
                "method": AccessMethod.VSCODE.value,
                "action": "open",
                "description": "VSCode로 코드 열기/편집"
            })
        
        if "파일" in goal or "file" in goal_lower:
            plan["recommended_steps"].append({
                "step": len(plan["recommended_steps"]) + 1,
                "method": AccessMethod.FILE_SYSTEM.value,
                "action": "read",
                "description": "파일 시스템 접근"
            })
        
        if "웹" in goal or "web" in goal_lower or "사이트" in goal:
            plan["recommended_steps"].append({
                "step": len(plan["recommended_steps"]) + 1,
                "method": AccessMethod.BROWSER.value,
                "action": "open",
                "description": "웹 브라우저로 사이트 열기"
            })
        
        if not plan["recommended_steps"]:
            plan["recommended_steps"].append({
                "step": 1,
                "method": AccessMethod.TERMINAL.value,
                "action": "execute",
                "description": "터미널에서 명령 실행"
            })
        
        return plan
    
    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        successful = sum(1 for r in self.access_history if r.success)
        failed = len(self.access_history) - successful
        
        method_stats = {}
        for result in self.access_history:
            method = result.method.value
            if method not in method_stats:
                method_stats[method] = {"success": 0, "failed": 0}
            if result.success:
                method_stats[method]["success"] += 1
            else:
                method_stats[method]["failed"] += 1
        
        return {
            "total_accesses": len(self.access_history),
            "successful": successful,
            "failed": failed,
            "by_method": method_stats,
            "safety_level": self.safety_level.value,
            "platform": self.state.platform
        }


# 데모 함수
def demo():
    """ComputerBridge 데모"""
    bridge = ComputerBridge(safety_level=SafetyLevel.READ_ONLY)
    
    print("=" * 60)
    print("🖥️ Computer Bridge Demo - 컴퓨터 제어 기반 외부 접근")
    print("=" * 60)
    
    # 컴퓨터 상태
    print("\n💻 컴퓨터 상태:")
    print(f"  플랫폼: {bridge.state.platform}")
    print(f"  호스트: {bridge.state.hostname}")
    print(f"  브라우저 가능: {bridge.state.browser_available}")
    print(f"  VSCode 가능: {bridge.state.vscode_available}")
    
    # 사용 가능한 방법
    print("\n🔧 사용 가능한 접근 방식:")
    for method, available in bridge.get_available_methods().items():
        status = "✅" if available else "❌"
        print(f"  {status} {method}")
    
    # 터미널 접근 테스트
    print("\n🖥️ 터미널 접근 테스트:")
    result = bridge.access(
        method=AccessMethod.TERMINAL,
        target="echo Hello from ComputerBridge!",
        action="execute"
    )
    if result.success:
        print(f"  ✅ 성공: {result.content.get('stdout', '').strip()}")
    else:
        print(f"  ❌ 실패: {result.error}")
    
    # 파일 시스템 접근 테스트
    print("\n📁 파일 시스템 접근 테스트:")
    result = bridge.access(
        method=AccessMethod.FILE_SYSTEM,
        target=".",
        action="read"
    )
    if result.success:
        files = result.content.get("files", [])[:5]
        print(f"  ✅ 현재 디렉토리 파일: {files}...")
    else:
        print(f"  ❌ 실패: {result.error}")
    
    # 접근 계획 생성
    print("\n📋 접근 계획 (목표: '정보 검색'):")
    plan = bridge.get_access_plan("인터넷에서 Python 정보 검색")
    for step in plan["recommended_steps"]:
        print(f"  Step {step['step']}: {step['description']} ({step['method']})")
    
    # 통계
    print("\n📊 통계:")
    stats = bridge.get_stats()
    print(f"  총 접근: {stats['total_accesses']}")
    print(f"  성공: {stats['successful']}, 실패: {stats['failed']}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
