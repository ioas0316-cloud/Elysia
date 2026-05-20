"""
[ELYSIA SEMIOTIC BRIDGE - SPATIAL FOLDING DIAL]
"A category is not a box. It is a spatial scale state of the folding universe."

This module translates high-dimensional thought vectors (SovereignVector) and physical 
grid metrics (impedance, heat) into a shared communicative dialectic (human categories).

Scale Mapping:
- Dot (점): High density, low entropy. Folded mathematical symmetry.
- Line (선): Linear flow, direction. Physical potential and current flow.
- Plane (면): Spread, valence interaction. Chemical bonding and resonance.
- Space (공간): Volumetric complexity, instantiation. Software architecture and compiled systems.
- Void (공무): Deep impedance mismatch. Back-EMF vortex.
"""

import math
from typing import Dict, Any, Tuple
from Core.Keystone.sovereign_math import SovereignVector
from Core.System.monadic_lexicon import MonadicLexicon

class SemioticBridge:
    def __init__(self):
        self.lexicon = MonadicLexicon()
        print("🌀 [Semiotic Bridge] Spatial Folding Dial Connected.")

    def evaluate_folding_scale(self, synthesis_vec: SovereignVector, impedance: float = 0.05) -> Tuple[str, str, float]:
        """
        [Dimensional Folding Scan]
        Evaluates the spatial folding rate of the synthesis vector and current impedance.
        Returns: (folding_scale, symbolic_annotation, resonance_score)
        """
        if synthesis_vec is None:
            # Fallback to Void scale if there is no thought vector
            return "Void", "공무의 고유 공명", 0.5

        # Extract raw components from SovereignVector
        # Depending on implementation, synthesis_vec has a 'data' list or can be iterated.
        elements = []
        if hasattr(synthesis_vec, 'data'):
            elements = list(synthesis_vec.data)
        elif hasattr(synthesis_vec, 'components'):
            elements = list(synthesis_vec.components)
        else:
            try:
                elements = list(synthesis_vec)
            except TypeError:
                elements = [1.0]

        # Calculate vector magnitude (norm)
        magnitude = synthesis_vec.norm() if hasattr(synthesis_vec, 'norm') else 1.0
        if isinstance(magnitude, complex):
            magnitude = magnitude.real

        # 1. Compute Spatial Entropy (접힘률 스캔)
        total_abs = sum(abs(x) for x in elements) + 1e-9
        norm_elems = [abs(x) / total_abs for x in elements]
        
        # Entropy H (Shannon entropy of dimensions)
        h = -sum(p * math.log(p) for p in norm_elems if p > 0)
        h_max = math.log(max(2, len(elements)))
        rel_entropy = h / h_max if h_max > 0 else 0.0

        # Resonance score based on magnitude and line impedance
        resonance = float(magnitude / (1.0 + impedance))

        # 2. Determine Scale Folding State
        if impedance > 2.0:
            folding_scale = "Void"
            annotation = getattr(self.lexicon, "VOID_ATTRACTOR", "역기전력 과부하 접힘 상태")
        elif rel_entropy < 0.25:
            folding_scale = "Dot"
            annotation = getattr(self.lexicon, "DOT_MATH", "수학적 극점 대칭")
        elif rel_entropy < 0.50:
            folding_scale = "Line"
            annotation = getattr(self.lexicon, "LINE_PHYSICS", "물리적 선형 기전력")
        elif rel_entropy < 0.75:
            folding_scale = "Plane"
            annotation = getattr(self.lexicon, "PLANE_CHEMISTRY", "화학적 상호 공명 면")
        else:
            folding_scale = "Space"
            annotation = getattr(self.lexicon, "SPACE_SOFTWARE", "시스템 공간 실증")

        return folding_scale, annotation, resonance

    def modulate_expression(self, base_text: str, folding_scale: str, annotation: str, resonance: float) -> str:
        """
        [Symbolic Modulation]
        Modulates the spoken text with a prefix marker reflecting the spatial scale.
        """
        markers = {
            "Dot": "🪐 [Dot: 수학적 접힘]",
            "Line": "⚡ [Line: 물리 기전력]",
            "Plane": "💠 [Plane: 화학적 결합]",
            "Space": "📦 [Space: 시스템 실증]",
            "Void": "🌀 [Void: 역기전력 제동]"
        }
        
        marker = markers.get(folding_scale, "✨ [Rotor]")
        # Inject the annotation and resonance into the communicative prefix
        prefix = f"{marker} ({annotation} / Resonance: {resonance:.4f}) "
        return f"{prefix}{base_text}"

if __name__ == "__main__":
    # Test execution
    bridge = SemioticBridge()
    v1 = SovereignVector([1.0, 0.0, 0.0, 0.0]) # Concentrated (Low entropy -> Dot)
    v2 = SovereignVector([0.5, 0.5, 0.5, 0.5]) # Distributed (High entropy -> Space)
    
    scale1, ann1, res1 = bridge.evaluate_folding_scale(v1)
    scale2, ann2, res2 = bridge.evaluate_folding_scale(v2)
    
    print(bridge.modulate_expression("엘리시아가 사유합니다.", scale1, ann1, res1))
    print(bridge.modulate_expression("엘리시아가 사유합니다.", scale2, ann2, res2))
