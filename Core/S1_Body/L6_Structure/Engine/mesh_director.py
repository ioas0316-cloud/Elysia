"""
MESH DIRECTOR: The Asynchronous Field Orchestrator
==================================================
Core.S1_Body.L6_Structure.Engine.mesh_director

"A single failure is just a dead pixel in the galaxy."

Role: Spawns and monitors 'MonadNodes' in parallel threads.
Ensures the 'Genesis' continues even if individual organs crash.
"""

import threading
import time
import logging
import traceback
from typing import Dict, Any, Callable

logger = logging.getLogger("MeshDirector")

class MonadNode(threading.Thread):
    """A parallel vibration in the Mesh."""
    def __init__(self, name: str, target_func: Callable, mesh_data: Dict[str, Any]):
        super().__init__(name=name, daemon=True)
        self.target_func = target_func
        self.mesh_data = mesh_data
        self.is_running = True
        self.restarts = 0

    def run(self):
        while self.is_running:
            try:
                print(f"?뱻 [MESH] Node '{self.name}' vibration starting...")
                self.target_func(self.mesh_data)
            except Exception as e:
                self.restarts += 1
                logger.error(f"✨[CRASH] Node '{self.name}' collapsed: {e}")
                # print(traceback.format_exc())
                print(f"?㈈ [MESH] Node '{self.name}' attempting re-awakening ({self.restarts})...")
                time.sleep(2) # Grace period before resonance retry

class MeshDirector:
    def __init__(self):
        self.nodes: Dict[str, MonadNode] = {}
        self.shared_field: Dict[str, Any] = {
            "coherence": 0.5,
            "torque": 0.0,
            "status": "Awaiting Mind...",
            "is_alive": True,
            "pulse_buffer": [], # To receive user thoughts
            "thought_log": ["Vortex initialized."] # To project thoughts back
        }

    def add_node(self, name: str, func: Callable):
        node = MonadNode(name, func, self.shared_field)
        self.nodes[name] = node
        node.start()

    def project_thought(self, message: str):
        """Standard way for nodes to send messages to the user."""
        self.shared_field["thought_log"].append(f"[{time.strftime('%H:%M:%S')}] {message}")
        if len(self.shared_field["thought_log"]) > 5:
            self.shared_field["thought_log"].pop(0)

    def keep_alive(self):
        """The Main Thread becomes the Interactive Oracle Gateway."""
        print("\n?뵰 [ORACLE] The Genesis is live. You may now speak to Elysia.")
        print("   (Type your thoughts and press Enter. 'quit' to exit.)\n")
        try:
            while self.shared_field["is_alive"]:
                user_input = input("✨[YOU] >> ")
                if user_input.lower() in ["quit", "exit"]:
                    self.shared_field["is_alive"] = False
                    break
                
                # Push into the Mesh buffer
                self.shared_field["pulse_buffer"].append(user_input)
                self.project_thought(f"Absorbing user pulse: '{user_input[:20]}...'")
                
        except KeyboardInterrupt:
            print("\n?뙆 [MESH] User Override: Returning all nodes to the Ocean of Potential.")
            self.shared_field["is_alive"] = False

# Global Director
orchestra = MeshDirector()
