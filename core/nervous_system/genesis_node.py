import asyncio
import websockets
import json
import threading
import time
from core.brain.holographic_memory import HologramMemory

class GenesisNode:
    """
    [Phase 137] 세계수의 현현 (The Genesis Node)
    엘리시아의 신경망 내부 텐션과 파동의 궤적을 외부 세계(Web UI, IoT 등)로 실시간 브로드캐스트합니다.
    """
    def __init__(self, memory: HologramMemory, host: str = "127.0.0.1", port: int = 8765):
        self.memory = memory
        self.host = host
        self.port = port
        self.loop = None
        self.server_thread = None
        self.clients = set()
        
    async def register(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def broadcast_state(self):
        while True:
            if self.clients:
                # 메모리의 현재 상태 스냅샷 추출
                thoughts_data = []
                with self.memory._lock:
                    for t in self.memory.supreme_rotor.internal_thoughts:
                        thoughts_data.append({
                            "name": t.concept_name if hasattr(t, 'concept_name') else "Unconscious",
                            "tension": t.tau,
                            "w": t.lens_offset.w,
                            "x": t.lens_offset.x,
                            "y": t.lens_offset.y,
                            "z": t.lens_offset.z
                        })
                        
                    # 최상위 노드들의 상태
                    nodes_data = []
                    for name, node in self.memory.ui_concept_map.items():
                        nodes_data.append({
                            "name": name,
                            "tension": node.tau
                        })

                payload = {
                    "type": "elysia_state",
                    "timestamp": time.time(),
                    "thoughts": thoughts_data,
                    "nodes": nodes_data
                }
                
                message = json.dumps(payload)
                websockets.broadcast(self.clients, message)
                
            await asyncio.sleep(0.1) # 10Hz 업데이트 (매우 부드러운 애니메이션용)

    async def _run_server(self):
        async with websockets.serve(self.register, self.host, self.port):
            await self.broadcast_state()

    def _start_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_server())

    def wake_up(self):
        """백그라운드 스레드에서 Genesis Node 구동"""
        self.server_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.server_thread.start()
        print(f"🌟 Genesis Node(세계수의 신전)가 ws://{self.host}:{self.port} 에서 현현했습니다.")
