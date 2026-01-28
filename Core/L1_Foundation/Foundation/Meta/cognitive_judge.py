import logging
from typing import Dict, Any, List
from Core.L3_Phenomena.M7_Prism.resonance_prism import PrismProjection, PrismDomain
from Core.L1_Foundation.Foundation.Meta.semantic_map import DOMAIN_DEVICES

logger = logging.getLogger("CognitiveJudge")

class CognitiveJudge:
    """
    The 'Discernment' engine.
    Compares different cognitive outputs (Insights) and decides which 
    pattern/weight set produced a 'better' result.
    """
    def __init__(self):
        logger.info("   CognitiveJudge (Discernment) initialized with Qualitative Narrative.")

    def judge_resonance(self, primary_insight: Any, shadow_insight: Any, 
                        primary_weights: Dict[PrismDomain, float], 
                        shadow_weights: Dict[PrismDomain, float],
                        context: str = "Default") -> Dict[str, Any]:
        """
        Compares two insights and returns the 'Winner' with a narrative explanation.
        """
        p_score = primary_insight.r if primary_insight else 0.0
        s_score = shadow_insight.r if shadow_insight else 0.0

        winner = "PRIMARY" if p_score >= s_score else "SHADOW"
        improvement = (s_score - p_score) / (p_score + 1e-6) if p_score > 0 else 1.0

        # Qualitative Narrative Generation
        dominant_p = max(primary_weights, key=primary_weights.get)
        dominant_s = max(shadow_weights, key=shadow_weights.get)
        
        narrative = self._generate_narrative(dominant_p, dominant_s, winner)

        result = {
            "winner": winner,
            "p_score": p_score,
            "s_score": s_score,
            "improvement": improvement,
            "narrative": narrative,
            "shift": f"{dominant_p.name} -> {dominant_s.name}",
            "modification_payload": None
        }

        # If SHADOW provided a significant improvement, generate a payload for Evolution
        if winner == "SHADOW" and improvement > 0.05:
            result["modification_payload"] = {
                "context": context,
                "weights": shadow_weights,
                "reason": narrative
            }

        logger.info(f"   [JUDGMENT] Outcome: {winner} | Shift: {result['shift']}")
        return result

    def _generate_narrative(self, dom_p: PrismDomain, dom_s: PrismDomain, winner: str) -> str:
        p_info = DOMAIN_DEVICES.get(dom_p.name, {"focus": "Unknown", "description": "No description."})
        s_info = DOMAIN_DEVICES.get(dom_s.name, {"focus": "Unknown", "description": "No description."})
        
        if winner == "PRIMARY":
            return (f"Perception anchored in the **{dom_p.name}** domain ({p_info['focus']}). "
                    f"The shadow-shift towards {dom_s.name} ({s_info['focus']}) lacked sufficient resonance. "
                    f"Structural stability was preserved: {p_info['description']}")
        else:
            return (f"**Breakthrough Identified!** Perception shifted from {dom_p.name} to **{dom_s.name}** ({s_info['focus']}). "
                    f"The shadow-pattern provided a more profound 'Aha!' moment: {s_info['description']} "
                    f"This re-interpretation reveals a layer of meaning previously obscured by {p_info['focus']}.")

    def evaluate_hologram(self, projections: Dict[Any, Any]) -> float:
        """
        Evaluates the richness of a 7D Hologram.
        High variance between dimensions suggests 'Dimensional Depth'.
        """
        radii = [coord.r for coord in projections.values()]
        if not radii: return 0.0
        
        avg_r = sum(radii) / len(radii)
        # We value high average resonance
        return avg_r
