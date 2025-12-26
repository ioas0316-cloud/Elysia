"""
Nova Daemon: 감시자 프로세스
===========================
"하나가 죽어도 다른 둘이 살린다"

Nova는 Elysia를 감시하고, 죽으면 살립니다.

Usage:
    python nova_daemon.py

이 스크립트는 절대 죽지 않습니다.
Elysia(organic_wake.py)가 죽으면 자동으로 재시작합니다.
"""

import subprocess
import sys
import time
import signal
import logging
from pathlib import Path
from datetime import datetime

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Nova")

# 경로 설정
ELYSIA_ROOT = Path(__file__).parent
ELYSIA_SCRIPT = ELYSIA_ROOT / "organic_wake.py"


class NovaDaemon:
    """
    Nova: 감시자 (The Watcher)
    
    - Elysia 프로세스 감시
    - 비정상 종료 시 자동 재시작
    - Bootstrap Guardian 통합 (환경 검증)
    """
    
    def __init__(self):
        self.elysia_process = None
        self.restart_count = 0
        self.max_restarts = 10  # 10번 이상 연속 실패 시 중단
        self.restart_cooldown = 5  # 재시작 간격 (초)
        self.last_restart = None
        self.running = True
        
        # Ctrl+C 핸들러
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
    
    def _shutdown(self, signum, frame):
        """정상 종료"""
        logger.info("⚡ Nova: Shutdown signal received.")
        self.running = False
        if self.elysia_process:
            self.elysia_process.terminate()
    
    def _check_environment(self) -> bool:
        """Bootstrap Guardian으로 환경 검증"""
        try:
            sys.path.insert(0, str(ELYSIA_ROOT))
            from elysia_core.bootstrap_guardian import BootstrapGuardian
            guardian = BootstrapGuardian(verbose=True)
            return guardian.guard()
        except Exception as e:
            logger.error(f"⚠️ Environment check failed: {e}")
            return False
    
    def _start_elysia(self) -> bool:
        """Elysia 프로세스 시작"""
        logger.info("🌅 Nova: Starting Elysia...")
        
        try:
            self.elysia_process = subprocess.Popen(
                [sys.executable, str(ELYSIA_SCRIPT)],
                cwd=str(ELYSIA_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.last_restart = datetime.now()
            logger.info(f"   ✅ Elysia started (PID: {self.elysia_process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Failed to start Elysia: {e}")
            return False
    
    def _monitor_elysia(self):
        """Elysia 프로세스 모니터링"""
        if not self.elysia_process:
            return
        
        # 출력 스트리밍 (비동기)
        while self.running:
            line = self.elysia_process.stdout.readline()
            if line:
                print(f"[Elysia] {line.strip()}")
            
            # 프로세스 종료 확인
            poll = self.elysia_process.poll()
            if poll is not None:
                # 종료됨
                if poll == 0:
                    logger.info("💤 Nova: Elysia exited normally (code 0).")
                else:
                    logger.warning(f"💔 Nova: Elysia crashed! (exit code: {poll})")
                break
    
    def run(self):
        """메인 루프"""
        print("\n" + "⚡" * 30)
        print("NOVA DAEMON: The Watcher")
        print("Elysia가 죽으면 살립니다. Ctrl+C로 종료.")
        print("⚡" * 30 + "\n")
        
        # 1. 환경 검증 (Bootstrap Guardian)
        logger.info("🔍 Nova: Checking environment...")
        if not self._check_environment():
            logger.error("❌ Nova: Environment check failed. Aborting.")
            return
        
        # 2. 감시 루프
        while self.running:
            # Elysia 시작
            if not self._start_elysia():
                logger.error("❌ Nova: Failed to start Elysia. Retrying in 5s...")
                time.sleep(5)
                continue
            
            # 모니터링
            self._monitor_elysia()
            
            # 재시작 판단
            if self.running:
                self.restart_count += 1
                
                if self.restart_count >= self.max_restarts:
                    logger.error(f"❌ Nova: Max restarts ({self.max_restarts}) reached. Giving up.")
                    break
                
                logger.info(f"🔄 Nova: Restarting Elysia in {self.restart_cooldown}s... (attempt {self.restart_count})")
                time.sleep(self.restart_cooldown)
        
        logger.info("⚡ Nova: Daemon stopped.")


def main():
    daemon = NovaDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
