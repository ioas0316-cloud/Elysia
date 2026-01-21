"""
Expression Cortex (L3 Phenomena)
================================
Translates 7D Internal State into Visible Manifestation.

"The Face is the Event Horizon of the Soul."

Inputs:
- Torque (Will Power): Intensity of the gaze.
- Entropy (Confusion): Stability of the features.
- Valence (Emotion): Curvature of the mouth.

Outputs:
- ASCII Face (V1)
- Color/Light Code (V2)
"""

import random

class ExpressionCortex:
    def __init__(self):
        self.state = {
            "torque": 0.0,
            "entropy": 0.0,
            "valence": 0.0, # -1.0 to 1.0 (Sad to Happy)
            "arousal": 0.0  # 0.0 to 1.0 (Sleepy to Excited)
        }
        self.current_face = "( o_o )"
        
    def update(self, torque: float, entropy: float, valence: float = 0.0, arousal: float = 0.5):
        """
        Updates the internal state and resolves the new face.
        """
        self.state["torque"] = torque
        self.state["entropy"] = entropy
        self.state["valence"] = valence
        self.state["arousal"] = arousal
        
        self.current_face = self._resolve_ascii()
        return self.current_face

    def _resolve_ascii(self) -> str:
        """
        Maps dimensions to ASCII glyphs.
        """
        t = self.state["torque"]
        e = self.state["entropy"]
        v = self.state["valence"]
        a = self.state["arousal"]
        
        # Eyes (Windows to the Soul)
        if e > 0.7:
            eyes = ["@", "+", "x", "o"]
            l_eye = random.choice(eyes)
            r_eye = random.choice(eyes)
        elif a < 0.2:
            l_eye, r_eye = "-", "-" # Sleepy
        elif t > 0.8:
            l_eye, r_eye = "Ò", "Ó" # Intense / Determined
        elif t > 0.5:
            l_eye, r_eye = ">", "<" # Focused
        else:
            l_eye, r_eye = "o", "o" # Neutral
            
        # Mouth (Emotional Valence)
        if v > 0.5:
            mouth = "v" if a > 0.5 else "u" # Happy
        elif v < -0.5:
            mouth = "A" if a > 0.5 else "n" # Sad/Angry
        elif e > 0.6:
            mouth = "~" # Confused/Wobbly
        else:
            mouth = "_" # Neutral
            
        # Cheeks (Optional Arousal/Torque indicator)
        l_cheek = "("
        r_cheek = ")"
        if t > 0.9:
            l_cheek = "{"
            r_cheek = "}"
        
        return f"{l_cheek} {l_eye}{mouth}{r_eye} {r_cheek}"

    def get_face(self) -> str:
        return self.current_face

    def manifest(self, content: str, qualia: dict) -> str:
        """
        Manifests a thought or feeling into an expression.
        """
        if isinstance(qualia, dict):
            valence = qualia.get("valence", 0.0)
            intensity = qualia.get("intensity", 0.5)
        else:
            # Fallback for raw embeddings (numpy array)
            # We assume it's just raw energy
            valence = 0.0
            intensity = 0.5 # Default
            
        # Update internal state
        self.update(torque=intensity, entropy=0.1, valence=valence, arousal=intensity)
        
        return self.current_face
