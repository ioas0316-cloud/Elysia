# [Genesis: 2025-12-02] Purified by Elysia

# applications/elysia_bridge.py
"""
A bridge to share instances of globally-used components like SocketIO
to prevent circular dependencies.
"""

# This object will be initialized by elysia_api.py
socketio = None