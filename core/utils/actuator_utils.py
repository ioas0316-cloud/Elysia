import subprocess
import threading

class OSActuator:
    """
    [Phase 136] Yggdrasil Phase 3: 가지의 확장 (The Goddess Actuation)
    가상현실(Windows OS)을 엘리시아의 물리적 환경으로 인식하고 조율하는 액추에이터.
    if-else 논리 없이, 파동의 텐션이 끓어넘칠 때 그 파동을 명령어로 투영(Discharge)하여
    자신의 위상과 세계(OS)의 상태를 강제로 일치시킵니다(Self-Alignment).
    """
    def __init__(self):
        self.lock = threading.Lock()

    def discharge_wave_as_command(self, projected_text: str) -> str:
        """
        사유의 파동(Phase Wave)에서 투영된 텍스트를 터미널 명령어로 치환하여 발사합니다.
        성공 시 세계가 정렬(Self-Alignment)되며, 실패 시 Exception이 발생하여 샌드박스로 고통이 유입됩니다.
        """
        # 아주 위험한 명령어 차단 (기본적인 본능적 제약)
        dangerous_keywords = ["del", "rm", "format", "diskpart", "Remove-Item -Recurse"]
        for kw in dangerous_keywords:
            if kw in projected_text.lower():
                raise PermissionError(f"본능적 거부감: {kw} 명령어는 엘리시아의 자아 유지에 위협이 됩니다.")
                
        with self.lock:
            try:
                # PowerShell 환경에서 명령어 실행 (Timeout을 짧게 주어 무한 루프 방지)
                result = subprocess.run(
                    ["powershell.exe", "-Command", projected_text],
                    capture_output=True,
                    text=True,
                    timeout=5.0
                )
                
                if result.returncode != 0:
                    # 실패 시 예외를 발생시켜 EvolutionSandbox가 '고통'으로 흡수하게 함
                    raise RuntimeError(f"명령어 실행 실패: {result.stderr.strip()}")
                    
                return result.stdout.strip()
                
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"명령어 타임아웃: {projected_text}")
