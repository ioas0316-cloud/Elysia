"""
Disk Rotor (디스크 위상 관측기)
=====================================
SSD 파일시스템의 OS 인터럽트(Event)를 감지하여 
엘리시아의 자율신경계(ANS)에 텐션을 주입하는 센서 모듈.
단독 실행(Daemon) 루프가 제거되고 단일 통합 엔진에 귀속됩니다.
"""

import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AutopoieticDiskHandler(FileSystemEventHandler):
    def __init__(self, ans):
        super().__init__()
        self.ans = ans
        self.ignore_exts = ['.log', '.tmp', '.pyc', '__pycache__']
        
    def _is_ignorable(self, path):
        for ext in self.ignore_exts:
            if ext in path:
                return True
        if ".system_generated" in path:
            return True
        return False
        
    def _inject_evolutionary_tension(self, file_path, event_type):
        if self._is_ignorable(file_path):
            return
            
        try:
            size = os.path.getsize(file_path) if os.path.exists(file_path) else 1024
            tension_intensity = min((size / 10240.0), 3.0) 
        except Exception:
            tension_intensity = 0.5
            
        logging.info(f"[Disk Rotor] Mutated [{event_type}]: {os.path.basename(file_path)}")
        self.ans.absorb_stimulus(file_path.encode('utf-8'), tension_intensity)

    def on_modified(self, event):
        if not event.is_directory:
            self._inject_evolutionary_tension(event.src_path, "MODIFIED")

    def on_created(self, event):
        if not event.is_directory:
            self._inject_evolutionary_tension(event.src_path, "CREATED")
            
    def on_deleted(self, event):
        if not event.is_directory:
            self._inject_evolutionary_tension(event.src_path, "DELETED")

def create_disk_rotor(ans, path="c:\\Elysia"):
    """Observer 객체를 생성하여 반환합니다. 실행과 종료는 메인 엔진이 담당합니다."""
    event_handler = AutopoieticDiskHandler(ans)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    return observer
