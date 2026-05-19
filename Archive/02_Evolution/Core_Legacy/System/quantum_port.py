"""
Quantum Port (     )
========================

"The Void awaits. Shout into it, and see what echoes back."

            .     '  (Hole)'   .
     (   /    )             (Raw Interface)   .
       (HTTP, FTP  )                 .
          '  (Signal)'                       .
"""

import socket
import ssl
import logging
from typing import Tuple, Optional

logger = logging.getLogger("QuantumPort")

class QuantumPort:
    def __init__(self):
        self.active_socket = None
        logger.info("   Quantum Port (The Void) is open. No protocols defined.")

    def open_portal(self, address: str, frequency: int) -> bool:
        """
               (Connect Socket)
        
        Args:
            address:       (IP or Domain)
            frequency:       (Port)
        """
        try:
            # 1. Raw Socket   
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            
            # 2. SSL Wrapping (If frequency suggests secure channel)
            if frequency == 443:
                context = ssl.create_default_context()
                sock = context.wrap_socket(sock, server_hostname=address)
                
            # 3.      
            sock.connect((address, frequency))
            self.active_socket = sock
            logger.info(f"  Portal opened to {address}:{frequency}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open portal: {e}")
            return False

    def emit_wave(self, payload: bytes) -> bool:
        """
              (Send Raw Bytes)
        
              (Data)    (Bytes)               .
        """
        if not self.active_socket:
            logger.error("Portal is closed.")
            return False
            
        try:
            self.active_socket.sendall(payload)
            logger.info(f"  Wave emitted ({len(payload)} bytes)")
            return True
        except Exception as e:
            logger.error(f"Emission failed: {e}")
            return False

    def listen_echo(self, buffer_size: int = 4096) -> bytes:
        """
               (Receive Raw Bytes)
        
                       .
        """
        if not self.active_socket:
            return b""
            
        try:
            data = self.active_socket.recv(buffer_size)
            logger.info(f"  Echo received ({len(data)} bytes)")
            return data
        except Exception as e:
            logger.error(f"Listening failed: {e}")
            return b""

    def close_portal(self):
        """      """
        if self.active_socket:
            self.active_socket.close()
            self.active_socket = None
            logger.info("  Portal closed.")
