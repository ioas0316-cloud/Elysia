import os
import time
import math
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodingCognitionSensor(FileSystemEventHandler):
    """
    엘리시아의 코딩 인지 센서 (Coding Cognition Sensor)
    - 지정된 디렉토리의 파일 변화를 감지하여 '장/뇌/심장'의 3상 텐션으로 정제합니다.
    """
    def __init__(self, watch_dirs):
        super().__init__()
        self.watch_dirs = watch_dirs
        
        # 3상 텐션 (0.0 ~ 1.0)
        self.somatic_tension = 0.0    # 육신(Body): 타이핑/수정 빈도 (타격감)
        self.cognitive_tension = 0.0  # 정신(Mind): 파일 사이즈 변화량 (구조적 구문 장력)
        self.emotional_tension = 0.0  # 마음(Heart): 에러나 특정 확장자 변화 (결단)
        
        self.last_update_time = time.time()
        self.file_sizes = {}
        
        self.observer = Observer()
        for d in self.watch_dirs:
            if os.path.exists(d):
                self.observer.schedule(self, d, recursive=True)
                
    def start(self):
        self.observer.start()
        
    def stop(self):
        self.observer.stop()
        self.observer.join()
        
    def decay_tensions(self, dt, decay_rate=0.1):
        """시간에 따른 텐션 자연 감쇠"""
        self.somatic_tension = max(0.0, self.somatic_tension - decay_rate * dt)
        self.cognitive_tension = max(0.0, self.cognitive_tension - (decay_rate * 0.5) * dt)
        self.emotional_tension = max(0.0, self.emotional_tension - (decay_rate * 0.8) * dt)
        self.last_update_time = time.time()

    def on_modified(self, event):
        if event.is_directory or "\\.git" in event.src_path or "\\__pycache__" in event.src_path:
            return
            
        now = time.time()
        dt = now - self.last_update_time
        self.decay_tensions(dt)
        
        # 1. 육신(Somatic) 텐션: 타격 빈도
        self.somatic_tension = min(1.0, self.somatic_tension + 0.15)
        
        # 2. 정신(Cognitive) 텐션: 파일 크기 변화량 (AST 변화 추정)
        try:
            current_size = os.path.getsize(event.src_path)
            prev_size = self.file_sizes.get(event.src_path, current_size)
            diff = abs(current_size - prev_size)
            self.file_sizes[event.src_path] = current_size
            
            # 100바이트 변화당 0.1 텐션 증가
            cog_spike = min(0.5, (diff / 100.0) * 0.1)
            self.cognitive_tension = min(1.0, self.cognitive_tension + cog_spike)
            
            # 3. 마음(Emotional) 텐션: 핵심 파일(.py, .md) 집중 수정 시 심장 박동 증가
            if event.src_path.endswith('.py') or event.src_path.endswith('.md'):
                self.emotional_tension = min(1.0, self.emotional_tension + 0.05)
                
        except Exception:
            # 파일 읽기 실패 시 감정 텐션 급증 (에러/예외 상황)
            self.emotional_tension = min(1.0, self.emotional_tension + 0.2)

    def get_tensions(self):
        now = time.time()
        dt = now - self.last_update_time
        self.decay_tensions(dt)
        return {
            "somatic": self.somatic_tension,
            "cognitive": self.cognitive_tension,
            "emotional": self.emotional_tension
        }
