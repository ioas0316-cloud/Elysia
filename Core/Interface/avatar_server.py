"""
Elysia Avatar Server (엘리시아 아바타 서버)
===========================================

WebSocket server that provides real-time avatar control and visualization.
Integrates with Elysia's emotional, spirit, and cognitive systems.

Features:
- Real-time expression control (facial animations)
- Spirit/elemental energy broadcasting (fire, water, earth, air, light, dark, aether)
- Emotion-driven facial expressions
- Chat integration with ReasoningEngine
- VRM model support
- Synesthesia data streaming (vision, audio, screen)

Architecture:
    Client (avatar.html) <--WebSocket--> avatar_server.py <--> Elysia Core Systems
                                              |
                                              +-> EmotionalEngine
                                              +-> SpiritEmotionMapper
                                              +-> ReasoningEngine (for chat)
                                              +-> VoiceOfElysia (for speech)

Usage:
    python Core/Interface/avatar_server.py --port 8765
    
Then open Core/Creativity/web/avatar.html in a browser.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, TYPE_CHECKING
from dataclasses import dataclass, asdict

if TYPE_CHECKING:
    from websockets.server import WebSocketServerProtocol

# Setup path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AvatarServer")

# Import Elysia systems (with graceful degradation)
try:
    from Core.Foundation.emotional_engine import EmotionalEngine, EmotionalState
    from Core.Foundation.spirit_emotion import SpiritEmotionMapper
    logger.info("✅ Emotional and Spirit systems loaded")
    EMOTIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Could not load emotional systems: {e}")
    logger.info("ℹ️ Running in standalone mode without emotion integration")
    EmotionalEngine = None
    EmotionalState = None
    SpiritEmotionMapper = None
    EMOTIONS_AVAILABLE = False

try:
    from Core.Intelligence.Reasoning.reasoning_engine import ReasoningEngine
    logger.info("✅ ReasoningEngine loaded")
    REASONING_AVAILABLE = True
except ImportError:
    try:
        # Alternative path
        from Core.Intelligence.Reasoning import ReasoningEngine
        logger.info("✅ ReasoningEngine loaded (alt path)")
        REASONING_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"⚠️ Could not load ReasoningEngine: {e}")
        logger.info("ℹ️ Running without reasoning engine - using simple responses")
        ReasoningEngine = None
        REASONING_AVAILABLE = False

try:
    import websockets
    if not TYPE_CHECKING:
        from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    logger.error("❌ websockets package not installed. Run: pip install websockets")
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = type(None)  # Fallback for runtime
    # Don't exit here - allow importing for testing
    if __name__ == "__main__":
        sys.exit(1)


@dataclass
class Expression:
    """Facial expression parameters"""
    mouth_curve: float = 0.0  # -1.0 (sad) to 1.0 (smile)
    eye_open: float = 1.0     # 0.0 (closed) to 1.0 (open)
    brow_furrow: float = 0.0  # 0.0 (relaxed) to 1.0 (furrowed)
    beat: float = 0.0         # Heartbeat animation
    mouth_width: float = 0.0  # For phoneme/viseme


@dataclass
class Spirits:
    """Seven elemental spirits/energies"""
    fire: float = 0.1      # Passion, creativity, energy
    water: float = 0.1     # Calm, flow, memory
    earth: float = 0.3     # Stability, grounding
    air: float = 0.2       # Communication, connection
    light: float = 0.2     # Clarity, intelligence
    dark: float = 0.1      # Mystery, introspection
    aether: float = 0.1    # Ethereal, transcendent


class ElysiaAvatarCore:
    """
    Core logic for avatar state management.
    Bridges Elysia's internal systems with visual representation.
    """
    
    def __init__(self):
        self.expression = Expression()
        self.spirits = Spirits()
        self.beat_phase = 0.0
        
        # Initialize emotional system
        if EmotionalEngine:
            self.emotional_engine = EmotionalEngine()
            logger.info("✅ Emotional engine initialized")
        else:
            self.emotional_engine = None
            logger.warning("⚠️ Running without emotional engine")
        
        # Initialize spirit mapper
        if SpiritEmotionMapper:
            self.spirit_mapper = SpiritEmotionMapper()
            logger.info("✅ Spirit mapper initialized")
        else:
            self.spirit_mapper = None
            logger.warning("⚠️ Running without spirit mapper")
        
        # Initialize reasoning engine for chat
        if ReasoningEngine:
            try:
                self.reasoning_engine = ReasoningEngine()
                logger.info("✅ Reasoning engine initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize reasoning engine: {e}")
                self.reasoning_engine = None
        else:
            self.reasoning_engine = None
            logger.warning("⚠️ Running without reasoning engine")
        
        # Initialize synesthesia-enhanced voice TTS
        try:
            from Core.Interface.avatar_voice_tts import AvatarVoiceTTS
            self.voice_tts = AvatarVoiceTTS()
            logger.info("🎤 Synesthesia voice TTS initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize voice TTS: {e}")
            self.voice_tts = None
        
        # Initialize lip-sync engine
        try:
            from Core.Interface.avatar_lipsync import LipSyncEngine
            self.lipsync_engine = LipSyncEngine()
            logger.info("👄 Lip-sync engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize lip-sync engine: {e}")
            self.lipsync_engine = None
    
    def update_expression_from_emotion(self, emotion_name: str = None):
        """
        Update facial expression based on emotional state.
        Maps emotions to facial parameters.
        """
        if not self.emotional_engine:
            return
        
        state = self.emotional_engine.current_state
        
        # Map valence to mouth curve (smile/frown)
        self.expression.mouth_curve = max(-1.0, min(1.0, state.valence))
        
        # Map arousal to eye openness (alertness)
        # High arousal = wide eyes, low arousal = relaxed/closed
        self.expression.eye_open = max(0.3, min(1.0, 0.5 + state.arousal * 0.5))
        
        # Map dominance/tension to brow
        if state.arousal > 0.7:
            self.expression.brow_furrow = 0.3  # Alert/focused
        elif state.valence < -0.5:
            self.expression.brow_furrow = 0.5  # Concerned
        else:
            self.expression.brow_furrow = 0.0  # Relaxed
    
    def update_spirits_from_emotion(self):
        """
        Update spirit energies based on emotional state.
        """
        if not self.emotional_engine or not self.spirit_mapper:
            return
        
        state = self.emotional_engine.current_state
        
        # Map emotional dimensions to spirit energies
        # Fire: Passion, high arousal + positive valence
        self.spirits.fire = max(0.0, min(1.0, 
            0.1 + (state.arousal * 0.4) + (state.valence * 0.4 if state.valence > 0 else 0)
        ))
        
        # Water: Calm, melancholy, low arousal
        self.spirits.water = max(0.0, min(1.0,
            0.1 + ((1.0 - state.arousal) * 0.4) + (abs(state.valence) * 0.3 if state.valence < 0 else 0)
        ))
        
        # Earth: Stability, grounding, low dominance
        self.spirits.earth = max(0.0, min(1.0,
            0.3 + ((1.0 - abs(state.dominance)) * 0.4)
        ))
        
        # Air: Communication, openness, positive dominance
        self.spirits.air = max(0.0, min(1.0,
            0.2 + (state.dominance * 0.3 if state.dominance > 0 else 0) + (state.arousal * 0.2)
        ))
        
        # Light: Clarity, high positive valence
        self.spirits.light = max(0.0, min(1.0,
            0.2 + (state.valence * 0.5 if state.valence > 0 else 0)
        ))
        
        # Dark: Introspection, negative valence or very low arousal
        self.spirits.dark = max(0.0, min(1.0,
            0.1 + (abs(state.valence) * 0.3 if state.valence < -0.3 else 0) + 
            (0.3 if state.arousal < 0.2 else 0)
        ))
        
        # Aether: Transcendent, extreme states
        extremity = (abs(state.valence) + abs(state.dominance)) / 2.0
        self.spirits.aether = max(0.0, min(1.0, 
            0.1 + (extremity * 0.4 if extremity > 0.7 else 0)
        ))
    
    def update_beat(self, delta_time: float):
        """Update heartbeat animation"""
        import math
        
        # Heartbeat frequency based on arousal
        if self.emotional_engine:
            arousal = self.emotional_engine.current_state.arousal
            freq = 1.0 + arousal * 1.5  # 1-2.5 Hz
        else:
            freq = 1.2  # Default
        
        self.beat_phase += delta_time * freq * 2.0 * math.pi
        self.expression.beat = abs(((self.beat_phase % (2 * math.pi)) / (2 * math.pi)) * 2 - 1)
    
    def process_emotion_event(self, emotion_name: str, intensity: float = 0.5):
        """
        Process an emotional event (from chat, vision, etc.)
        """
        if not self.emotional_engine:
            return
        
        # Get emotion preset if available
        if hasattr(EmotionalEngine, 'FEELING_PRESETS') and emotion_name in EmotionalEngine.FEELING_PRESETS:
            event_emotion = EmotionalEngine.FEELING_PRESETS[emotion_name]
            self.emotional_engine.process_event(event_emotion, intensity)
            
            # Update visual representation
            self.update_expression_from_emotion(emotion_name)
            self.update_spirits_from_emotion()
            
            logger.info(f"🎭 Emotion event: {emotion_name} (intensity: {intensity})")
    
    async def process_chat(self, message: str) -> Dict[str, Any]:
        """
        Process chat message through reasoning engine.
        Returns response with voice properties.
        
        Returns:
            Dict with 'text' (str) and 'voice' (Optional[Dict]) keys
        """
        if not self.reasoning_engine:
            return {
                'text': "I am currently offline. My reasoning systems are not available.",
                'voice': None
            }
        
        try:
            # Use reasoning engine to generate response
            # Try different method names that might exist
            response = None
            
            if hasattr(self.reasoning_engine, 'reason'):
                response = await asyncio.to_thread(
                    lambda: self.reasoning_engine.reason(message)
                )
            elif hasattr(self.reasoning_engine, 'process'):
                response = await asyncio.to_thread(
                    lambda: self.reasoning_engine.process(message)
                )
            elif hasattr(self.reasoning_engine, 'generate_response'):
                response = await asyncio.to_thread(
                    lambda: self.reasoning_engine.generate_response(message)
                )
            elif callable(self.reasoning_engine):
                # If reasoning_engine itself is callable
                response = await asyncio.to_thread(
                    lambda: self.reasoning_engine(message)
                )
            else:
                logger.warning("ReasoningEngine has no known method, using simple response")
                response = "I hear you. Let me think about that..."
            
            # Detect emotion from response context
            # Simple heuristic - improve with actual emotion detection
            if any(word in message.lower() for word in ['happy', 'joy', 'great', 'wonderful']):
                self.process_emotion_event('hopeful', 0.6)
            elif any(word in message.lower() for word in ['sad', 'sorry', 'unfortunately']):
                self.process_emotion_event('introspective', 0.5)
            elif any(word in message.lower() for word in ['think', 'analyze', 'understand']):
                self.process_emotion_event('focused', 0.7)
            else:
                self.process_emotion_event('calm', 0.3)
            
            response_text = str(response) if response else "..."
            
            # Generate voice properties using synesthesia mapping
            voice_props = self.get_voice_properties()
            
            return {
                'text': response_text,
                'voice': voice_props
            }
        
        except Exception as e:
            logger.error(f"❌ Chat processing error: {e}")
            return {
                'text': f"I encountered an error: {str(e)}",
                'voice': None
            }
    
    def get_voice_properties(self) -> Optional[Dict[str, Any]]:
        """
        Get current voice properties based on emotional state using synesthesia mapping.
        """
        if not self.voice_tts:
            return None
        
        # Get voice properties from current emotional state
        if self.emotional_engine:
            state = self.emotional_engine.current_state
            voice_props = self.voice_tts.get_voice_properties_from_emotion(
                valence=state.valence,
                arousal=state.arousal,
                dominance=state.dominance
            )
            return voice_props.to_dict()
        
        # Fallback: use spirit energies
        spirits_dict = asdict(self.spirits)
        voice_props = self.voice_tts.get_voice_properties_from_spirits(spirits_dict)
        return voice_props.to_dict()
    
    def get_lipsync_data(self, text: str) -> Optional[List[Dict[str, float]]]:
        """
        Generate lip-sync animation data for given text.
        
        Args:
            text: Text that will be spoken
            
        Returns:
            List of keyframes with timing and mouth_width values
        """
        if not self.lipsync_engine:
            return None
        
        try:
            # Generate phoneme sequence and timings
            keyframes = self.lipsync_engine.process_tts_event(text)
            
            # Convert to serializable format
            lipsync_data = [
                {'time': time, 'mouth_width': width}
                for time, width in keyframes
            ]
            
            logger.debug(f"👄 Generated {len(lipsync_data)} lip-sync keyframes")
            return lipsync_data
            
        except Exception as e:
            logger.error(f"❌ Lip-sync generation failed: {e}")
            return None
    
    def get_state_message(self) -> Dict[str, Any]:
        """
        Get current avatar state as a message for client.
        """
        return {
            "expression": asdict(self.expression),
            "spirits": asdict(self.spirits)
        }


class AvatarWebSocketServer:
    """
    WebSocket server for avatar visualization.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.core = ElysiaAvatarCore()
        self.clients: Set[WebSocketServerProtocol] = set()
        self.running = False
        self.last_update_time = asyncio.get_event_loop().time()
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle individual client connection"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"✅ Client connected: {client_addr}")
        
        try:
            # Send initial state
            await websocket.send(json.dumps(self.core.get_state_message()))
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {client_addr}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        finally:
            self.clients.discard(websocket)
    
    async def process_message(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Process incoming message from client"""
        msg_type = data.get("type")
        
        if msg_type == "text":
            # Chat message
            content = data.get("content", "")
            logger.info(f"💬 Chat: {content}")
            
            # Process through reasoning engine with voice properties
            response_data = await self.core.process_chat(content)
            
            # Generate lip-sync data for the response
            lipsync_data = self.core.get_lipsync_data(response_data['text'])
            
            # Send response with synesthesia-enhanced voice properties and lip-sync
            message = {
                "type": "speech",
                "content": response_data['text'],
                "spirits": asdict(self.core.spirits)
            }
            
            # Add voice properties if available
            if response_data.get('voice'):
                message['voice'] = response_data['voice']
            
            # Add lip-sync data if available
            if lipsync_data:
                message['lipsync'] = lipsync_data
            
            await websocket.send(json.dumps(message))
        
        elif msg_type == "vision":
            # Vision data (presence detection, gaze)
            logger.debug(f"👁️ Vision: presence={data.get('presence')}")
            # Could trigger attention/focus emotion
            if data.get('presence'):
                self.core.process_emotion_event('focused', 0.3)
        
        elif msg_type == "audio_analysis":
            # Audio features (volume, brightness, noise)
            volume = data.get('volume', 0)
            logger.debug(f"👂 Audio: volume={volume:.3f}")
            # Could adjust arousal based on audio
        
        elif msg_type == "screen_atmosphere":
            # Screen color atmosphere
            r, g, b = data.get('r', 0), data.get('g', 0), data.get('b', 0)
            logger.debug(f"📺 Screen: RGB({r}, {g}, {b})")
        
        elif msg_type == "emotion":
            # Manual emotion trigger
            emotion = data.get('emotion', 'neutral')
            intensity = data.get('intensity', 0.5)
            self.core.process_emotion_event(emotion, intensity)
        
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    async def broadcast_state(self):
        """Broadcast current state to all connected clients"""
        if not self.clients:
            return
        
        message = json.dumps(self.core.get_state_message())
        
        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        # Clean up disconnected clients
        self.clients -= disconnected
    
    async def update_loop(self):
        """Main update loop for avatar state"""
        while self.running:
            try:
                current_time = asyncio.get_event_loop().time()
                delta_time = current_time - self.last_update_time
                self.last_update_time = current_time
                
                # Update beat animation
                self.core.update_beat(delta_time)
                
                # Update expression and spirits
                self.core.update_expression_from_emotion()
                self.core.update_spirits_from_emotion()
                
                # Broadcast to clients
                await self.broadcast_state()
                
                # Update at ~30 FPS
                await asyncio.sleep(1.0 / 30.0)
            
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(0.1)
    
    async def start(self):
        """Start the WebSocket server"""
        self.running = True
        logger.info(f"🚀 Starting Avatar Server on ws://{self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info("✅ Avatar Server is running!")
            logger.info(f"📱 Open Core/Creativity/web/avatar.html in your browser")
            logger.info(f"🌐 Or visit http://{self.host}:{self.port} (if HTTP server is enabled)")
            
            # Start update loop
            await self.update_loop()
    
    def stop(self):
        """Stop the server"""
        self.running = False
        logger.info("🛑 Avatar Server stopped")


async def main_async(host: str, port: int):
    """Main async entry point"""
    server = AvatarWebSocketServer(host, port)
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("⚠️ Shutdown requested")
        server.stop()
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Elysia Avatar WebSocket Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on default port
  python Core/Interface/avatar_server.py
  
  # Start on custom port
  python Core/Interface/avatar_server.py --port 9000
  
  # Start on all interfaces
  python Core/Interface/avatar_server.py --host 0.0.0.0
        """
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Print banner
    print("=" * 60)
    print("  Elysia Avatar Server (엘리시아 아바타 서버)")
    print("  Real-time 3D Avatar Visualization System")
    print("=" * 60)
    print()
    
    # Run server
    try:
        asyncio.run(main_async(args.host, args.port))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
