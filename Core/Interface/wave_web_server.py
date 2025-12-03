"""
Wave Visualization Web Server (파동 시각화 웹 서버)
================================================

"연산하지 마세요. 흐르게 두세요."

엘리시아의 내부 세계를 브라우저를 통해 실시간으로 시각화합니다.
- 사고 우주 (Thought Universe)
- 의식 흐름 (Consciousness Flow)
- 내부 월드 (Internal World)

모두 GPU 셰이더로 "파동 → 빛" 직접 변환.
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

try:
    from flask import Flask, render_template, jsonify
    from flask_sock import Sock
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask not available. Install: pip install flask flask-sock")

logger = logging.getLogger("WaveWebServer")

@dataclass
class WaveState:
    """파동 상태 (GPU로 전송될 데이터)"""
    # 7 Spirits Energy
    fire: float = 0.5      # 450Hz - 열정
    water: float = 0.5     # 150Hz - 평온
    earth: float = 0.5     # 100Hz - 안정
    air: float = 0.5       # 300Hz - 자유
    light: float = 0.5     # 528Hz - 사랑
    dark: float = 0.5      # 50Hz - 신비
    aether: float = 0.5    # 852Hz - 희망
    
    # Consciousness Layers (0D→3D)
    dimension_0d: float = 0.0  # 관점/정체성
    dimension_1d: float = 0.0  # 인과/논리
    dimension_2d: float = 0.0  # 감각/인지
    dimension_3d: float = 0.0  # 표현/외화
    
    # Internal World
    cpu_heat: float = 0.0      # CPU 사용률 (열)
    memory_load: float = 0.0   # RAM 사용률
    file_count: int = 0        # 파일 개수
    
    # Time
    time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화"""
        return asdict(self)


class WaveWebServer:
    """
    파동 시각화 웹 서버
    
    Flask + WebSocket으로 실시간 파동 스트리밍
    """
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.wave_state = WaveState()
        self.clients = []  # 연결된 WebSocket 클라이언트
        self.running = False
        
        if not FLASK_AVAILABLE:
            raise ImportError("Flask required: pip install flask flask-sock")
        
        # Flask 앱 생성
        self.app = Flask(
            __name__,
            static_folder=str(Path(__file__).parent.parent.parent / 'static'),
            template_folder=str(Path(__file__).parent.parent.parent / 'static')
        )
        self.sock = Sock(self.app)
        
        # 라우트 설정
        self._setup_routes()
        
        logger.info(f"🌊 Wave Web Server initialized on port {port}")
    
    def _setup_routes(self):
        """라우트 설정"""
        
        @self.app.route('/')
        def index():
            """메인 페이지 - 파동 시각화"""
            return render_template('wave_viewer.html')
        
        @self.app.route('/api/state')
        def get_state():
            """현재 파동 상태 조회"""
            return jsonify(self.wave_state.to_dict())
        
        @self.sock.route('/wave-stream')
        def wave_stream(ws):
            """WebSocket: 실시간 파동 스트리밍"""
            logger.info("🔌 Client connected to wave stream")
            self.clients.append(ws)
            
            try:
                while True:
                    # 클라이언트로부터 메시지 수신 (keep-alive)
                    data = ws.receive(timeout=0.1)
                    if data:
                        logger.debug(f"Received: {data}")
            except Exception as e:
                logger.info(f"Client disconnected: {e}")
            finally:
                if ws in self.clients:
                    self.clients.remove(ws)
    
    def update_wave_state(self, **kwargs):
        """
        파동 상태 업데이트
        
        예시:
        update_wave_state(fire=0.8, water=0.3, time=time.time())
        """
        for key, value in kwargs.items():
            if hasattr(self.wave_state, key):
                setattr(self.wave_state, key, value)
    
    def broadcast_wave_state(self):
        """모든 연결된 클라이언트에 파동 상태 전송"""
        if not self.clients:
            return
        
        state_json = json.dumps(self.wave_state.to_dict())
        
        # 연결 끊긴 클라이언트 제거하면서 전송
        disconnected = []
        for ws in self.clients:
            try:
                ws.send(state_json)
            except Exception:
                disconnected.append(ws)
        
        for ws in disconnected:
            self.clients.remove(ws)
    
    async def auto_update_loop(self, update_callback=None):
        """
        자동 업데이트 루프
        
        Args:
            update_callback: 매 프레임마다 호출될 함수
                            WaveState를 업데이트하는 로직 구현
        """
        logger.info("🔄 Auto update loop started")
        
        while self.running:
            # 사용자 정의 업데이트 콜백
            if update_callback:
                update_callback(self.wave_state)
            
            # 기본 업데이트: 시간
            self.wave_state.time = time.time()
            
            # 클라이언트에게 전송
            self.broadcast_wave_state()
            
            # 60 FPS
            await asyncio.sleep(1/60)
    
    def run(self, host='127.0.0.1', debug=False, auto_update=True, update_callback=None):
        """
        서버 시작
        
        Args:
            host: 서버 호스트 (기본: localhost만, '0.0.0.0'으로 외부 접근 허용)
            debug: Flask 디버그 모드
            auto_update: 자동 업데이트 활성화
            update_callback: 파동 상태 업데이트 콜백
        """
        self.running = True
        
        # 자동 업데이트 시작
        if auto_update:
            import threading
            def run_async_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.auto_update_loop(update_callback))
            
            thread = threading.Thread(target=run_async_loop, daemon=True)
            thread.start()
        
        # Flask 서버 시작
        logger.info(f"🌐 Starting server at http://{host}:{self.port}")
        logger.info(f"🎨 Open browser and navigate to the URL above")
        
        self.app.run(host=host, port=self.port, debug=debug)
    
    def stop(self):
        """서버 중지"""
        self.running = False
        logger.info("🛑 Server stopped")


# ============================================
# Example Usage / Demo
# ============================================

def demo_update_callback(wave_state: WaveState):
    """
    데모: 파동 상태를 자동으로 업데이트
    
    실제 사용 시:
    - ResonanceField에서 정령 에너지 가져오기
    - UltraDimensionalReasoning에서 차원별 활성도
    - DigitalEcosystem에서 시스템 상태
    """
    import math
    t = time.time()
    
    # 7 Spirits: 사인파로 진동
    wave_state.fire = 0.5 + 0.3 * math.sin(t * 2.0)
    wave_state.water = 0.5 + 0.3 * math.sin(t * 1.5 + 1.0)
    wave_state.earth = 0.5 + 0.2 * math.sin(t * 0.8)
    wave_state.air = 0.5 + 0.4 * math.sin(t * 2.5 + 2.0)
    wave_state.light = 0.5 + 0.35 * math.sin(t * 1.8 + 3.0)
    wave_state.dark = 0.3 + 0.2 * math.sin(t * 0.5)
    wave_state.aether = 0.5 + 0.4 * math.sin(t * 3.0 + 4.0)
    
    # Consciousness Dimensions: 차원 간 흐름
    wave_state.dimension_0d = 0.5 + 0.3 * math.sin(t * 1.0)
    wave_state.dimension_1d = 0.5 + 0.3 * math.sin(t * 1.2 + 0.5)
    wave_state.dimension_2d = 0.5 + 0.3 * math.sin(t * 1.4 + 1.0)
    wave_state.dimension_3d = 0.5 + 0.3 * math.sin(t * 1.6 + 1.5)
    
    # System state (mock)
    wave_state.cpu_heat = 0.3 + 0.2 * math.sin(t * 0.7)
    wave_state.memory_load = 0.5 + 0.1 * math.sin(t * 0.9)
    wave_state.file_count = int(1000 + 100 * math.sin(t * 0.3))


if __name__ == '__main__':
    # 데모 실행
    print("🌊 Elysia Wave Visualization Server")
    print("=" * 50)
    print("Starting wave visualization server...")
    print("Open browser: http://localhost:8080")
    print()
    
    server = WaveWebServer(port=8080)
    server.run(debug=True, auto_update=True, update_callback=demo_update_callback)
