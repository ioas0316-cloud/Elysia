"""
The Mycelium Network: Hive Mind
===============================
Phase 21 The Tree - Module 3
Core.L2_Metabolism.Reproduction.mycelium

"Roots do not speak, but the forest knows."

This module handles inter-process communication (IPC) between
Mother and Child nodes using UDP broadcasts or simple TCP/HTTP.
"""

import logging
import socket
import threading
import json
from typing import Callable, Optional

logger = logging.getLogger("Reproduction.Mycelium")

class MyceliumNetwork:
    """
    The Telepathic Link.
    """
    def __init__(self, port: int = 5000, callback: Optional[Callable] = None):
        self.port = port
        self.callback = callback
        self.running = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind to localhost
        try:
            self.sock.bind(('localhost', self.port))
            logger.info(f"🕸️ [MYCELIUM] Listening on port {self.port}")
            self.running = True
            self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listen_thread.start()
        except Exception as e:
            logger.warning(f"⚠️ [MYCELIUM] Port {self.port} busy. Am I a Child?")
            # If port 5000 is busy, we might be a child. Try ephemeral.
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.running = False # Passive mode for now

    def broadcast(self, message: dict, target_port: int = 5000):
        """
        Sends a thought to the network.
        """
        try:
            payload = json.dumps(message).encode('utf-8')
            self.sock.sendto(payload, ('localhost', target_port))
            logger.info(f"🕸️ [MYCELIUM] Sent: {message.get('type')} -> :{target_port}")
        except Exception as e:
            logger.error(f"❌ [MYCELIUM] Failed to transmit: {e}")

    def _listen_loop(self):
        """
        Background listener.
        """
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                msg = json.loads(data.decode('utf-8'))
                logger.info(f"🕸️ [MYCELIUM] Received from {addr}: {msg}")
                
                if self.callback:
                    self.callback(msg)
                    
            except Exception as e:
                logger.error(f"❌ [MYCELIUM] Listen Error: {e}")
                time.sleep(1)

    def close(self):
        self.running = False
        self.sock.close()
