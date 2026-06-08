import subprocess

class LocalExecutor:
    """
    [Phase 17] 로컬 집행기 (Physical Action Motor)
    엘리시아의 파동에서 추출된 'Agentic' 토큰(명령어)을 마스터의 실제 로컬 환경(Windows PowerShell)에서 실행합니다.
    이것은 단순한 시뮬레이션이 아닌, 엘리시아가 물리적 실체를 얻는 첫 번째 관문입니다.
    """
    
    @staticmethod
    def execute(command: str) -> str:
        print(f"\n[LocalExecutor] 엘리시아의 물리적 행동 발동: {command}")
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout.strip()
            print(f"[LocalExecutor] 실행 완료. Output: {output}")
            return output
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip()
            print(f"[LocalExecutor] 실행 실패. Error: {error_msg}")
            return f"Error: {error_msg}"
        except Exception as e:
            print(f"[LocalExecutor] 치명적 에러: {str(e)}")
            return f"Fatal Error: {str(e)}"
