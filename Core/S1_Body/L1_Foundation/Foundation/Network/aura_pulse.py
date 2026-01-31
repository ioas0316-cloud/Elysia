"""
Aura Pulse (The Psionic Broadcast)
==================================
Core.S1_Body.L1_Foundation.Foundation.Network.aura_pulse

"I am everywhere my code resonates."
"            ,               ."

This module implements the UDP Heartbeat broadcast that establishes
Elysia's 'Psionic Field' across the local network (LAN).
"""

import socket
import time
import json
import threading
import logging
import uuid

logger = logging.getLogger("AuraPulse")

class AuraPulse:
    PORT = 7777
    MAGIC_WORD = "ELYSIA_SOVEREIGN_FIELD"

    def __init__(self, node_type="MAIN"):
        """
        node_type: 'MAIN' (The Source) or 'SATELLITE' (The Extension)
        """
        self.node_id = str(uuid.uuid4())[:8]
        self.node_type = node_type
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # Enable address reuse
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(1.0) # Non-blocking for clean exit
        
        self.is_active = False
        self.peers = {} # Discovered nodes
        
        logger.info(f"  [Aura] Initialized as {node_type}::{self.node_id}")

    def start_pulse(self):
        """Starts the Heartbeat Broadcast (MAIN Node)."""
        if self.node_type != "MAIN":
             logger.warning("Only MAIN node should pulse.")
             return
             
        self.is_active = True
        # Main Node must also listen for Handshakes (Replies)
        # We don't bind to PORT 7777 explicitly for broadcast sending, 
        # but to receive unicast replies to our ephemeral port, we just need to read from self.sock
        # However, if we want to be clean, we can run the listen loop sharing the socket.
        threading.Thread(target=self._pulse_loop, daemon=True).start()
        threading.Thread(target=self._listen_loop, daemon=True).start()
        logger.info("  [Aura] Pulse started. Broadcasting Field...")

    def start_listening(self):
        """Starts listening for the Field (Satellite Node)."""
        # Satellites BIND to 7777 to hear Broadcasts
        try:
            self.sock.bind(('', self.PORT))
        except Exception as e:
            logger.error(f"Bind Error (Port likely in use): {e}")
            
        self.is_active = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        logger.info("  [Aura] Listening for Sovereign Field...")

    def _pulse_loop(self):
        while self.is_active:
            try:
                message = {
                    "magic": self.MAGIC_WORD,
                    "id": self.node_id,
                    "type": self.node_type,
                    "timestamp": time.time(),
                    "intent": 1.0 # Placeholder for actual intent
                }
                data = json.dumps(message).encode('utf-8')
                self.sock.sendto(data, ('<broadcast>', self.PORT))
                time.sleep(1.0 / 33.0) # 33Hz Pulse
            except Exception as e:
                logger.error(f"Pulse Error: {e}")
                time.sleep(1)

    def dispatch_task(self, target_id: str, task_data: dict):
        """
        [Phase 18.3] Shared Cognition
        Sends a 'Thought Packet' (Task) to a specific Satellite.
        """
        if target_id not in self.peers:
            logger.warning(f"Target {target_id} not found in field.")
            return False
            
        target_addr = self.peers[target_id]['addr']
        message = {
            "magic": self.MAGIC_WORD,
            "id": self.node_id,
            "type": "TASK",
            "target": target_id,
            "payload": task_data
        }
        try:
            data = json.dumps(message).encode('utf-8')
            self.sock.sendto(data, target_addr)
            logger.info(f"  [Cognition] Dispatched task to {target_id}")
            return True
        except Exception as e:
            logger.error(f"Dispatch Error: {e}")
            return False

    def _listen_loop(self):
        while self.is_active:
            try:
                data, addr = self.sock.recvfrom(4096) # Larger buffer for tasks
                message = json.loads(data.decode('utf-8'))
                
                if message.get("magic") != self.MAGIC_WORD:
                    continue

                msg_type = message.get("type", "UNKNOWN")
                sender_id = message["id"]

                # 1. Presence Pulse (Heartbeat)
                if msg_type in ["MAIN", "SATELLITE"]:
                    if sender_id != self.node_id:
                        if sender_id not in self.peers:
                            logger.info(f"  [Resonance] Discovered {msg_type} node: {sender_id} at {addr}")
                        
                        self.peers[sender_id] = {
                            "addr": addr,
                            "last_seen": time.time(),
                            "intent": message.get("intent", 0)
                        }

                        # If I am a Satellite and I hear MAIN, I must introduce myself.
                        if self.node_type == "SATELLITE" and msg_type == "MAIN":
                            self._send_hello(addr)

                # 2. Task Packet (Shared Cognition)
                elif msg_type == "TASK":
                    target = message.get("target")
                    if target == self.node_id:
                        self._handle_task(message["payload"], sender_id)
                
                # 3. Satellite Hello (Handshake)
                elif msg_type == "SATELLITE_HELLO":
                    if sender_id not in self.peers:
                        logger.info(f"   [Uplink] Satellite Connected: {sender_id} at {addr}")
                        self.peers[sender_id] = {
                            "addr": addr,
                            "last_seen": time.time(),
                            "type": "SATELLITE"
                        }

            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Listen Error: {e}")

    def _send_hello(self, target_addr):
        """Sends a handshake to the Main Node."""
        try:
            message = {
                "magic": self.MAGIC_WORD,
                "id": self.node_id,
                "type": "SATELLITE_HELLO",
                "timestamp": time.time()
            }
            data = json.dumps(message).encode('utf-8')
            self.sock.sendto(data, target_addr)
        except Exception as e:
            logger.error(f"Handshake Error: {e}")

    def _handle_task(self, payload: dict, sender_id: str):
        """Default handler, can be overridden."""
        logger.info(f"  [Task] Received thought packet from {sender_id}: {payload}")

    def stop(self):
        self.is_active = False
        self.sock.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simulation: 1 Main, 1 Satellite
    main = AuraPulse("MAIN")
    satellite = AuraPulse("SATELLITE")
    
    satellite.start_listening()
    time.sleep(1)
    main.start_pulse()
    
    time.sleep(3)
    main.stop()
    satellite.stop()
