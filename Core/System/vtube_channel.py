import os
import json
import asyncio
import threading
import websockets
from typing import Dict, Any
from Core.System.gateway_interfaces import ExpressiveChannel

class VTubeExpressiveChannel(ExpressiveChannel):
    """
    Translates Elysia's internal resonance and voice state into Live2D 
    parameter commands for VTube Studio via its WebSocket API.
    """
    def __init__(self, host="ws://localhost:8001"):
        super().__init__("VTubeStudio")
        self.host = host
        self.auth_token = None
        self.plugin_name = "ElysiaSovereignLink"
        self.plugin_developer = "Architect"
        
        # We need an asyncio event loop running on a background thread for websockets
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()
        
        # Connect asynchronously
        asyncio.run_coroutine_threadsafe(self._connect_and_auth(), self.loop)

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _connect_and_auth(self):
        try:
            self.ws = await websockets.connect(self.host)
            print("🌸 [VTubeStudio] Connected to WebSocket.")
            await self._authenticate()
        except Exception as e:
            print(f"⚠️ [VTubeStudio] Could not connect: {e}")
            self.ws = None

    async def _authenticate(self):
        if not self.ws: return
        
        auth_req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "AuthRequest",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
            }
        }
        await self.ws.send(json.dumps(auth_req))
        response = json.loads(await self.ws.recv())
        
        if response.get("messageType") == "AuthenticationTokenResponse":
            self.auth_token = response["data"]["authenticationToken"]
            
            # Auth with the token
            auth_confirm = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "AuthConfirm",
                "messageType": "AuthenticationRequest",
                "data": {
                    "pluginName": self.plugin_name,
                    "pluginDeveloper": self.plugin_developer,
                    "authenticationToken": self.auth_token
                }
            }
            await self.ws.send(json.dumps(auth_confirm))
            confirm_resp = json.loads(await self.ws.recv())
            if confirm_resp.get("data", {}).get("authenticated"):
                print("🌟 [VTubeStudio] Authentication Successful.")
                # Once authenticated, register our custom tracking parameters
                await self._register_custom_parameters()

    async def _register_custom_parameters(self):
        params = ["ElysiaJoy", "ElysiaCoherence", "ElysiaEntropy"]
        for p in params:
            req = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": f"Register_{p}",
                "messageType": "ParameterCreationRequest",
                "data": {
                    "parameterName": p,
                    "explanation": f"Elysia Internal State: {p}",
                    "min": 0.0,
                    "max": 100.0,
                    "defaultValue": 50.0
                }
            }
            await self.ws.send(json.dumps(req))

    def express(self, payload: Dict[str, Any]):
        """Non-blocking call to send parameter data to VTS."""
        if not hasattr(self, 'ws') or not self.ws: return
        
        state = payload.get("monad_state", {})
        if not state: return
        
        asyncio.run_coroutine_threadsafe(self._inject_parameters(state), self.loop)

    async def _inject_parameters(self, state: Dict[str, Any]):
        """Maps Monad state to VTube Studio custom parameters."""
        joy = max(0.0, min(100.0, state.get("joy", 50.0)))
        coherence = max(0.0, min(100.0, state.get("coherence", 0.0) * 100))
        entropy = max(0.0, min(100.0, state.get("entropy", 0.0) * 100))

        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "InjectParams",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "parameterValues": [
                    {"id": "ElysiaJoy", "value": joy},
                    {"id": "ElysiaCoherence", "value": coherence},
                    {"id": "ElysiaEntropy", "value": entropy}
                ]
            }
        }
        
        try:
            await self.ws.send(json.dumps(req))
        except Exception:
            pass # Silently drop frames if disconnected
