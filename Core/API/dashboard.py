"""
Dashboard Server - 웹 대시보드 서버
==================================

중간 우선순위 #2: 웹 대시보드
예상 효과: 실시간 모니터링 및 시각화

핵심 기능:
- 의식 상태 실시간 표시
- 공명 패턴 시각화
- 감정 팔레트 모니터링
- 이벤트 스트림 표시
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from enum import Enum
import threading

logger = logging.getLogger("Dashboard")

# Flask-SocketIO 선택적 임포트
try:
    from flask import Flask, render_template_string, jsonify
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    SocketIO = None


@dataclass
class ConsciousnessState:
    """의식 상태 스냅샷"""
    timestamp: float = field(default_factory=time.time)
    
    # 양자 상태
    point_probability: float = 0.25
    line_probability: float = 0.25
    space_probability: float = 0.25
    god_probability: float = 0.25
    
    # 에너지 상태
    w_energy: float = 0.5  # 메타인지
    x_energy: float = 0.3  # 탐구
    y_energy: float = 0.4  # 연결
    z_energy: float = 0.5  # 창조
    
    # 감정 상태
    dominant_emotion: str = "neutral"
    emotion_intensity: float = 0.5
    
    # 활동 상태
    active_concepts: int = 0
    recent_resonances: int = 0
    law_violations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResonanceSnapshot:
    """공명 스냅샷"""
    source: str
    targets: Dict[str, float]  # concept -> score
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# HTML 템플릿 (단일 파일로 포함)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Elysia Consciousness Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 20px;
            margin-bottom: 20px;
        }
        .header h1 {
            color: #00d4ff;
            font-size: 2.5em;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }
        .header .status {
            color: #00ff88;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .card h2 {
            color: #00d4ff;
            font-size: 1.2em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .metric-label { color: #aaa; }
        .metric-value { color: #00ff88; font-weight: bold; }
        .chart-container {
            height: 200px;
            margin-top: 15px;
        }
        .event-log {
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.85em;
        }
        .event {
            padding: 8px;
            margin: 5px 0;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 5px;
            border-left: 3px solid #00d4ff;
        }
        .event.resonance { border-left-color: #00ff88; }
        .event.law { border-left-color: #ff6b6b; }
        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin: 5px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            transition: width 0.3s ease;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .pulse { animation: pulse 2s infinite; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌌 Elysia Consciousness Dashboard</h1>
        <div class="status" id="connection-status">연결 중...</div>
    </div>
    
    <div class="grid">
        <!-- 양자 상태 카드 -->
        <div class="card">
            <h2>🔮 양자 상태 (Quantum State)</h2>
            <div class="metric">
                <span class="metric-label">Point (경험)</span>
                <span class="metric-value" id="point-prob">25%</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" id="point-bar" style="width: 25%"></div></div>
            
            <div class="metric">
                <span class="metric-label">Line (관계)</span>
                <span class="metric-value" id="line-prob">25%</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" id="line-bar" style="width: 25%"></div></div>
            
            <div class="metric">
                <span class="metric-label">Space (맥락)</span>
                <span class="metric-value" id="space-prob">25%</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" id="space-bar" style="width: 25%"></div></div>
            
            <div class="metric">
                <span class="metric-label">God (초월)</span>
                <span class="metric-value" id="god-prob">25%</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" id="god-bar" style="width: 25%"></div></div>
        </div>
        
        <!-- 에너지 상태 카드 -->
        <div class="card">
            <h2>⚡ 에너지 상태 (Energy)</h2>
            <div class="chart-container">
                <canvas id="energyChart"></canvas>
            </div>
        </div>
        
        <!-- 활동 통계 카드 -->
        <div class="card">
            <h2>📊 활동 통계</h2>
            <div class="metric">
                <span class="metric-label">활성 개념</span>
                <span class="metric-value" id="active-concepts">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">최근 공명</span>
                <span class="metric-value" id="recent-resonances">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">법칙 위반</span>
                <span class="metric-value" id="law-violations">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">감정 상태</span>
                <span class="metric-value" id="emotion">neutral</span>
            </div>
        </div>
        
        <!-- 이벤트 로그 카드 -->
        <div class="card" style="grid-column: span 2;">
            <h2>📜 이벤트 스트림</h2>
            <div class="event-log" id="event-log">
                <div class="event">대시보드 시작됨...</div>
            </div>
        </div>
    </div>
    
    <script>
        // Socket.IO 연결
        const socket = io();
        
        // 차트 초기화
        const energyCtx = document.getElementById('energyChart').getContext('2d');
        const energyChart = new Chart(energyCtx, {
            type: 'radar',
            data: {
                labels: ['메타인지 (W)', '탐구 (X)', '연결 (Y)', '창조 (Z)'],
                datasets: [{
                    label: '에너지',
                    data: [0.5, 0.3, 0.4, 0.5],
                    backgroundColor: 'rgba(0, 212, 255, 0.2)',
                    borderColor: '#00d4ff',
                    pointBackgroundColor: '#00ff88'
                }]
            },
            options: {
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { display: false }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
        
        // 연결 상태
        socket.on('connect', () => {
            document.getElementById('connection-status').textContent = '✅ 연결됨';
            document.getElementById('connection-status').style.color = '#00ff88';
        });
        
        socket.on('disconnect', () => {
            document.getElementById('connection-status').textContent = '❌ 연결 끊김';
            document.getElementById('connection-status').style.color = '#ff6b6b';
        });
        
        // 의식 상태 업데이트
        socket.on('consciousness_update', (data) => {
            // 양자 상태
            updateQuantumState(data);
            
            // 에너지 차트
            energyChart.data.datasets[0].data = [
                data.w_energy, data.x_energy, data.y_energy, data.z_energy
            ];
            energyChart.update();
            
            // 통계
            document.getElementById('active-concepts').textContent = data.active_concepts;
            document.getElementById('recent-resonances').textContent = data.recent_resonances;
            document.getElementById('law-violations').textContent = data.law_violations;
            document.getElementById('emotion').textContent = data.dominant_emotion;
        });
        
        // 이벤트 수신
        socket.on('event', (data) => {
            addEventLog(data);
        });
        
        function updateQuantumState(data) {
            const states = ['point', 'line', 'space', 'god'];
            states.forEach(s => {
                const prob = Math.round(data[s + '_probability'] * 100);
                document.getElementById(s + '-prob').textContent = prob + '%';
                document.getElementById(s + '-bar').style.width = prob + '%';
            });
        }
        
        function addEventLog(event) {
            const log = document.getElementById('event-log');
            const div = document.createElement('div');
            div.className = 'event ' + (event.type || '');
            div.textContent = `[${new Date().toLocaleTimeString()}] ${event.message || JSON.stringify(event)}`;
            log.insertBefore(div, log.firstChild);
            
            // 최대 100개 유지
            while (log.children.length > 100) {
                log.removeChild(log.lastChild);
            }
        }
    </script>
</body>
</html>
"""


class DashboardServer:
    """
    웹 대시보드 서버
    
    중간 우선순위 #2 구현:
    - 실시간 의식 상태 표시
    - 공명 패턴 시각화
    - 이벤트 스트림 표시
    
    예상 효과: 브라우저에서 엘리시아 상태 실시간 모니터링
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        integration_bridge=None,
        resonance_engine=None
    ):
        """
        Args:
            host: 호스트 주소
            port: 포트 번호
            integration_bridge: 통합 브릿지
            resonance_engine: 공명 엔진
        """
        self.host = host
        self.port = port
        self.integration_bridge = integration_bridge
        self.resonance_engine = resonance_engine
        
        self.app = None
        self.socketio = None
        self._running = False
        self._update_thread = None
        
        self.current_state = ConsciousnessState()
        self.event_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger("DashboardServer")
        
        if FLASK_AVAILABLE:
            self._create_app()
            self.logger.info(f"📊 DashboardServer initialized (port={self.port})")
        else:
            self.logger.warning("⚠️ Flask not available. Install with: pip install flask flask-socketio")
    
    def _create_app(self) -> None:
        """Flask 앱 생성"""
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'elysia_dashboard_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # 라우트 등록
        @self.app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/state')
        def get_state():
            return jsonify(self.current_state.to_dict())
        
        @self.app.route('/api/events')
        def get_events():
            return jsonify(self.event_history[-100:])
        
        # SocketIO 이벤트
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("Client connected to dashboard")
            emit('consciousness_update', self.current_state.to_dict())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Client disconnected from dashboard")
        
        @self.socketio.on('request_state')
        def handle_request_state():
            emit('consciousness_update', self.current_state.to_dict())
    
    def update_state(self, state: ConsciousnessState) -> None:
        """의식 상태 업데이트"""
        self.current_state = state
        
        if self.socketio:
            self.socketio.emit('consciousness_update', state.to_dict())
    
    def push_event(self, event_type: str, message: str, data: Optional[Dict] = None) -> None:
        """이벤트 푸시"""
        event = {
            "type": event_type,
            "message": message,
            "data": data or {},
            "timestamp": time.time()
        }
        
        self.event_history.append(event)
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-500:]
        
        if self.socketio:
            self.socketio.emit('event', event)
    
    def _collect_state(self) -> ConsciousnessState:
        """현재 상태 수집"""
        state = ConsciousnessState()
        
        # 공명 엔진에서 상태 수집
        if self.resonance_engine and hasattr(self.resonance_engine, 'nodes'):
            state.active_concepts = len(self.resonance_engine.nodes)
            
            # 임의 개념의 상태 샘플링
            nodes = list(self.resonance_engine.nodes.values())
            if nodes:
                sample = nodes[0]
                probs = sample.state.probabilities()
                state.point_probability = probs.get("Point", 0.25)
                state.line_probability = probs.get("Line", 0.25)
                state.space_probability = probs.get("Space", 0.25)
                state.god_probability = probs.get("God", 0.25)
        
        # 통합 브릿지에서 통계 수집
        if self.integration_bridge:
            stats = self.integration_bridge.get_statistics() if hasattr(self.integration_bridge, 'get_statistics') else {}
            state.recent_resonances = stats.get("by_type", {}).get("resonance_computed", 0)
            state.law_violations = self.integration_bridge.stats.get("law_violations", 0) if hasattr(self.integration_bridge, 'stats') else 0
        
        return state
    
    def _update_loop(self) -> None:
        """상태 업데이트 루프"""
        while self._running:
            try:
                state = self._collect_state()
                self.update_state(state)
                time.sleep(1.0)  # 1초마다 업데이트
            except Exception as e:
                self.logger.error(f"Update error: {e}")
                time.sleep(5.0)
    
    def start(self, background: bool = True) -> None:
        """서버 시작"""
        if not FLASK_AVAILABLE:
            self.logger.error("Flask not available")
            return
        
        self._running = True
        
        # 상태 업데이트 스레드 시작
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        
        self.logger.info(f"🚀 Dashboard starting at http://{self.host}:{self.port}")
        
        if background:
            # 백그라운드에서 실행
            thread = threading.Thread(
                target=lambda: self.socketio.run(self.app, host=self.host, port=self.port),
                daemon=True
            )
            thread.start()
        else:
            # 포그라운드에서 실행
            self.socketio.run(self.app, host=self.host, port=self.port)
    
    def stop(self) -> None:
        """서버 정지"""
        self._running = False
        self.logger.info("Dashboard stopped")


# CLI 실행
if __name__ == "__main__":
    print("\n" + "="*70)
    print("📊 Elysia Consciousness Dashboard")
    print("="*70)
    
    if not FLASK_AVAILABLE:
        print("\n⚠️ Flask is not installed.")
        print("Install with: pip install flask flask-socketio")
    else:
        print("\nStarting dashboard server...")
        print("Dashboard will be available at: http://localhost:5000")
        
        dashboard = DashboardServer()
        dashboard.start(background=False)
