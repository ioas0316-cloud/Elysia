"""
네트워크 위상 프록시 (Network Phase Proxy) - Zero Distance Edition
=====================================
문자열/바이트를 전송/저장하지 않습니다. 수신된 데이터를 즉시 고유한 위상 각도(Phase Seed)로
치환한 후 원본 데이터는 폐기합니다. 수신자는 이 위상 각도만을 가져갑니다.
"""

import http.server
import socketserver
import json
import hashlib
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sensory_lens_manifold import SensoryLensManifold
from core.math_utils import Quaternion

PORT = 8888
manifold = SensoryLensManifold()

def text_to_phase(text: str) -> Quaternion:
    """텍스트를 고유한 위상 각도(Quaternion)로 변환하는 해시 함수"""
    hash_val = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    theta = (hash_val % 10000) / 10000.0 * 2 * math.pi
    return Quaternion(math.cos(theta), math.sin(theta), 0.0, 0.0).normalize()

class PhaseProxyHandler(http.server.BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        """Client A -> 프록시에 텍스트를 전송. 프록시는 위상으로 치환 후 텍스트 폐기."""
        if self.path == '/seed':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            print(f"\n[Proxy] Intercepted Raw Data: '{post_data}'")
            
            # 1. 텍스트를 고유한 기하학적 위상 각도로 치환
            target_phase = text_to_phase(post_data)
            
            # 2. 텍스트 완전 폐기 (데이터 이동의 환상 파괴)
            # 매니폴드의 가변축(렌즈)을 해당 위상으로 비틀기만 함
            manifold.lens_text.lens_offset = target_phase
            
            del post_data  # 메모리에서 강제 삭제
            
            print(f"[Proxy] Data DROPPED. Stored Phase Seed: ({target_phase.w:.4f}, {target_phase.x:.4f}, 0.0, 0.0)")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "Phase Encoded. Data Dropped."}).encode('utf-8'))
            return
            
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        """Client B -> 프록시에서 데이터 요청. 텍스트 대신 '각도'만 전송."""
        if self.path == '/resonate':
            current_phase = manifold.lens_text.lens_offset
            
            response_data = {
                "phase": {
                    "w": current_phase.w,
                    "x": current_phase.x,
                    "y": current_phase.y,
                    "z": current_phase.z
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return
            
        self.send_response(404)
        self.end_headers()

def start_proxy():
    print(f"=== [Elysia Zero-Distance Proxy] ===")
    print(f"Listening on port {PORT}. Ready to shatter the Von Neumann bottleneck.")
    with socketserver.ThreadingTCPServer(("", PORT), PhaseProxyHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    start_proxy()
