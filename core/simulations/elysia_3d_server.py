"""
Elysia 4D Hyper-Accelerated Server
======================================
GPU 주관 시간 초가속을 위해 물리 엔진을 분리하고,
4차원 원본 위상 데이터(w,x,y,z)를 전송합니다.
"""

import asyncio
import websockets
import json
import torch
import math
import logging
import threading
import http.server
import socketserver
import webbrowser
import os
import numpy as np

from core.brain.fractal_multiverse import FractalMultiverse
from core.brain.topological_language_mapper import TopologicalLanguageMapper
from core.brain.causal_phase_mapper import CausalPhaseMapper
from core.utils.math_utils import Quaternion
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 다중 우주 초기화
multiverse = FractalMultiverse(num_nodes=2000)
# 인과 위상 매퍼 (메인 우주용)
causal_mapper = multiverse.causal_mapper
# 언어 우주 매핑 (메인 우주 대상)
lang_mapper = TopologicalLanguageMapper(multiverse.lenses['Language'], start_idx=100)
num_seeded = lang_mapper.seed_vocabulary()
logging.info(f"위상 언어 매퍼 초기화 완료: {num_seeded}개의 단어 앵커가 심어졌습니다.")
knowledge_labels = {}
next_anchor_idx = 1
connected_clients = set()
view_offset = 0
current_joy = 0.0

async def handler(websocket):
    global next_anchor_idx, view_offset
    logging.info("웹 4D 클라이언트 연결됨. 주관 시간 초가속 돌입.")
    connected_clients.add(websocket)
    
    try:
        for idx, text in knowledge_labels.items():
            await websocket.send(json.dumps({
                'type': 'knowledge_label',
                'node_idx': idx,
                'text': text
            }))
            
        async for message in websocket:
            data = json.loads(message)
            if data.get('type') == 'inject_knowledge' and next_anchor_idx < 2000:
                text = data.get('text', '')
                q_w, q_x, q_y, q_z = data['q_w'], data['q_x'], data['q_y'], data['q_z']
                q_tensor = torch.tensor([q_w, q_x, q_y, q_z], dtype=torch.float32, device=multiverse.device)
                
                multiverse.lenses['Language'].rotor_field.anchor_knowledge(next_anchor_idx, q_tensor)
                knowledge_labels[next_anchor_idx] = text
                
                for client in list(connected_clients):
                    try:
                        await client.send(json.dumps({
                            'type': 'knowledge_label',
                            'node_idx': next_anchor_idx,
                            'text': text
                        }))
                    except:
                        pass
                next_anchor_idx += 1
                
            elif data.get('type') == 'set_time_offset':
                view_offset = int(data.get('offset', 0))
                # 버퍼 크기에 맞게 클램핑
                if view_offset < -len(multiverse.history_buffer) + 1:
                    view_offset = -len(multiverse.history_buffer) + 1
                if view_offset > 0:
                    view_offset = 0
                
            elif data.get('type') == 'stimulus':
                # 사용자의 입력을 철저한 기하학적 인과를 가진 4D 파동으로 변환
                text = data.get('text', '')
                q_tensor = causal_mapper.text_to_phase(text)
                
                # 의식의 감각수용기(0번 노드)에 자극 주입
                focus_idx = 0
                main_universe = multiverse.lenses['Language'].rotor_field
                main_universe.phases[focus_idx] = q_tensor
                
                # 자극 직후의 기쁨(Resonance) 즉시 측정 (엘리시아의 주권적 판단)
                adj = main_universe.adjacency[focus_idx]
                phases = main_universe.phases
                local_resonance = (phases[focus_idx] * phases).sum(dim=1) * adj
                avg_resonance = local_resonance.sum().item() / (adj.sum().item() + 1e-9)
                
                # 의도와 다른 공백(여백)이 발생했는지 관측
                if avg_resonance > 0.4:
                    action = "수용(Resonance)"
                    color = "#00ff00"
                else:
                    action = "거부/여백생성(Void)"
                    color = "#ff007f"
                    
                for client in list(connected_clients):
                    try:
                        await client.send(json.dumps({
                            'type': 'observatory_log',
                            'message': f"[{action}] '{text}' (공명도: {avg_resonance:.2f})",
                            'color': color
                        }))
                    except:
                        pass
                
                # 0번 노드 주변으로 자가 정렬된 단어들을 즉시 추출
                adj_np = adj.cpu().numpy()
                connected_indices = np.argsort(adj_np)[-5:][::-1] # 상위 5개만 옹알이
                
                babble_words = []
                for idx in connected_indices:
                    if idx in lang_mapper.node_to_word and adj[idx] > 0.6:
                        babble_words.append(lang_mapper.node_to_word[idx])
                
                if babble_words:
                    for client in list(connected_clients):
                        try:
                            await client.send(json.dumps({
                                'type': 'babble',
                                'words': babble_words,
                                'joy': current_joy
                            }))
                        except:
                            pass
                            
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)

async def physics_loop():
    """GPU 주관 시간 초가속 (Hyper-Acceleration) 루프"""
    global current_joy
    while True:
        current_joy = multiverse.step_physics()
        await asyncio.sleep(0.001)

async def broadcast_loop():
    """60FPS 관측 렌더링 루프"""
    global view_offset
    while True:
        await asyncio.sleep(1/40.0) # 40 FPS
        if not connected_clients: continue
        if not multiverse.history_buffer: continue
        
        idx = view_offset - 1 if view_offset < 0 else -1
        if idx < -len(multiverse.history_buffer):
            idx = -len(multiverse.history_buffer)
            
        frame = multiverse.history_buffer[idx]
        phases_np = frame['phases']
        adj_matrix = frame['adj']
        
        # 4D 위상 원본 추출 (w, x, y, z)
        pw = phases_np[:, 0] * 5.0
        px = phases_np[:, 1] * 5.0
        py = phases_np[:, 2] * 5.0
        pz = phases_np[:, 3] * 5.0
        
        pos_4d = np.column_stack((pw, px, py, pz)).flatten().tolist()
        
        rows, cols = np.where(adj_matrix > 0.8) # 강한 연결만
        links = []
        max_links = 20000
        for r, c in zip(rows, cols):
            if r < c:
                links.append(int(r))
                links.append(int(c))
                if len(links) >= max_links:
                    break
        
        state_msg = json.dumps({
            "type": "state",
            "joy": current_joy,
            "pos_4d": pos_4d,
            "links": links
        })
        
        for client in list(connected_clients):
            try:
                await client.send(state_msg)
            except:
                pass

async def main():
    def start_http_server():
        web_dir = os.path.join(os.path.dirname(__file__), 'web_3d')
        os.chdir(web_dir)
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", 8000), handler)
        logging.info("HTTP Server 시작: http://localhost:8000")
        httpd.serve_forever()
        
    threading.Thread(target=start_http_server, daemon=True).start()
    
    logging.info("WebSocket 4D 하이퍼 스트리밍 대기 중 (ws://localhost:8765)")
    webbrowser.open('http://localhost:8000')
    
    server = await websockets.serve(handler, "localhost", 8765, max_size=16777216)
    
    # 초가속 루프와 브로드캐스트 루프를 동시에 실행
    await asyncio.gather(
        physics_loop(),
        broadcast_loop()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
