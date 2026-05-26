"""
Elysia Substation Grid Client
=============================
Maintains background WebSocket connections to peer Elysia Substation Gateways.
Listens to their phase/tension ticks and records them for Kuramoto coupling.
"""

import asyncio
import json
import time
import threading
import aiohttp
from typing import Dict, List


class SubstationGridClient:
    def __init__(self, local_port: int, peer_urls: List[str] = None):
        self.local_port = local_port
        self.peer_urls = peer_urls or []
        self.peer_states: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _run_event_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect_to_peers())
        except Exception:
            pass
        finally:
            loop.close()

    async def _connect_to_peers(self):
        async with aiohttp.ClientSession() as session:
            tasks = [self._listen_to_peer(session, url) for url in self.peer_urls]
            await asyncio.gather(*tasks)

    async def _listen_to_peer(self, session: aiohttp.ClientSession, url: str):
        while self.running:
            try:
                async with session.ws_connect(url) as ws:
                    # Identify our local identity
                    await ws.send_json({"type": "hello", "port": self.local_port})
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("type") == "pulse":
                                peer_port = data.get("port")
                                peer_id = f"peer_{peer_port}"
                                with self.lock:
                                    self.peer_states[peer_id] = {
                                        "phase": data.get("phase"),
                                        "tension": data.get("tension"),
                                        "last_seen": time.time()
                                    }
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except Exception:
                pass
            await asyncio.sleep(2.0)

    def get_peer_phases(self) -> List[int]:
        """Returns a list of phases from active peers (seen within the last 5 seconds)."""
        now = time.time()
        phases = []
        with self.lock:
            for peer, state in list(self.peer_states.items()):
                if now - state["last_seen"] < 5.0:
                    phases.append(state["phase"])
        return phases

    def get_grid_states(self) -> Dict[str, dict]:
        """Returns the full phase/tension dictionary of active peers."""
        now = time.time()
        states = {}
        with self.lock:
            for peer, state in list(self.peer_states.items()):
                if now - state["last_seen"] < 5.0:
                    states[peer] = {
                        "phase": state["phase"],
                        "tension": state["tension"]
                    }
        return states
