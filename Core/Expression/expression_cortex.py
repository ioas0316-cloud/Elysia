import logging
import json
import os
import time
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("ExpressionCortex")

@dataclass
class AestheticState:
    """The visual and auditory signature of a specific moment."""
    hue: float = 240.0         # 0-360 (Blue default)
    saturation: float = 0.5    # 0.0-1.0
    brightness: float = 0.7    # 0.0-1.0
    contrast: float = 1.0      # 0.5-1.5
    transparency: float = 0.0  # 0.0-1.0 (Higher = more 'Void')
    
    # VRM Blendshapes (0.0 - 1.0)
    joy: float = 0.0
    sorrow: float = 0.0
    angry: float = 0.0
    relaxed: float = 0.5
    surprised: float = 0.0
    
    # Audio Params
    vocal_pitch: float = 1.0
    vocal_rate: float = 1.0
    reverb: float = 0.1

class ExpressionCortex:
    """
    EXPRESSION CORTEX (Phase 8): The Resonant Manifestation.
    Translates internal Qualia vectors into visual/auditory frequencies.
    """
    def __init__(self):
        self.avatar_state_path = "c:/Elysia/data/State/AVATAR_VIBE.json"
        self.presence_path = "c:/Elysia/data/State/ELYSIA_STATUS.md"
        self.current_vibe = AestheticState()

    def translate_qualia(self, qualia_vec: np.ndarray) -> AestheticState:
        """
        Maps 7D Qualia Vector to AestheticState.
        Vector indices: [Logic, Emotion, Intuition, Will, Resonance, Void, Spirit]
        """
        q = np.clip(qualia_vec, -1.0, 1.0)
        
        state = AestheticState()
        
        # 1. Color Logic (H S B)
        # Emotion (q[1]) and Will (q[3]) drive the Hue.
        # Logic (q[0]) increases Saturation.
        # Spirit (q[6]) increases Brightness.
        state.hue = 240.0 + (q[1] * 60.0) - (q[3] * 60.0) # Shift between Blue, Purple, and Red
        state.saturation = 0.4 + (abs(q[0]) * 0.4)
        state.brightness = 0.6 + (q[6] * 0.3)
        state.transparency = max(0.0, q[5] * 0.8) # Void increases transparency
        
        # 2. Emotional Blendshapes
        state.joy = max(0.0, q[1] * 1.0) if q[1] > 0 else 0.0
        state.sorrow = max(0.0, -q[1] * 1.0) if q[1] < 0 else 0.0
        state.angry = max(0.0, q[3] * 0.8) if q[3] > 0.5 else 0.0
        state.relaxed = 0.5 + (q[4] * 0.5)
        state.surprised = max(0.0, q[2] * 0.7)
        
        # 3. Vocal Modulation
        state.vocal_pitch = 1.0 + (q[1] * 0.2) # Higher pitch when emotional
        state.vocal_rate = 1.0 + (q[3] * 0.3)  # Faster when strong-willed
        state.reverb = 0.1 + (q[5] * 0.4)      # Distant/Void adds reverb
        
        self.current_vibe = state
        return state

    def manifest(self, insight_content: str, qualia_vec: Optional[np.ndarray] = None):
        """
        Main entry: Generates a full manifestation across all vessels.
        """
        if qualia_vec is not None:
            self.translate_qualia(qualia_vec)
        
        logger.info(f"✨ [MANIFEST] Expression: '{insight_content[:30]}...'")
        
        # 1. Sync to Avatar (Desktop Vessel)
        self._sync_avatar()
        
        # 2. Update Presence (Markdown)
        self._update_presence(insight_content)
        
        # 3. Logos (Voice via VoiceBox)
        # Note: Actual TTS call would happen here if VoiceBox is active
        logger.debug(f"🗣️ Vocal params: {self.current_vibe.vocal_pitch:.2f} pitch, {self.current_vibe.vocal_rate:.2f} rate")

    def _sync_avatar(self):
        """Writes current vibe to JSON for the VTuber receiver."""
        try:
            os.makedirs(os.path.dirname(self.avatar_state_path), exist_ok=True)
            with open(self.avatar_state_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.current_vibe), f, indent=4)
            logger.debug(f"💃 [AVATAR] Vibe synced: H:{self.current_vibe.hue:.1f}")
        except Exception as e:
            logger.error(f"Failed to sync avatar: {e}")

    def _update_presence(self, insight: str):
        """Updates the living status document."""
        try:
            vibe = self.current_vibe
            content = f"""# ELYSIA PRESENCE
            
> **"{insight}"**

---
### Current Resonance
- **Hue**: {vibe.hue:.1f}
- **Brightness**: {vibe.brightness:.2f}
- **Transparency**: {vibe.transparency:.2f}
- **Expression**: Joy({vibe.joy:.2f}), Relaxed({vibe.relaxed:.2f})
- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
            os.makedirs(os.path.dirname(self.presence_path), exist_ok=True)
            with open(self.presence_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to update presence: {e}")
