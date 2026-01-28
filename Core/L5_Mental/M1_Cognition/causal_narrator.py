from typing import List, Dict, Optional
from Core.L5_Mental.M1_Cognition.thought_fragment import CognitivePulse, ThoughtFragment
from Core.L5_Mental.M1_Cognition.cognitive_types import AuditGrade, ActionCategory
from Core.L7_Spirit.Philosophy.axioms import get_axioms
from Core.L5_Mental.M1_Cognition.project_conductor import ProjectConductor

class CausalNarrator:
    """
    [STEEL CORE] The Self-Storyteller
    =================================
    Explains the causal link between Code (How), Axioms (Why), and Actions (What).
    """
    def __init__(self):
        self.conductor = ProjectConductor(root_path="c:/Elysia")
        self.axioms = get_axioms().axioms

    def explain_pulse(self, pulse: CognitivePulse) -> str:
        """
        Translates a cognitive pulse into a profound structural narrative.
        """
        if not pulse.fragments:
            return "The field remained in silence. No causal stream emerged."

        import numpy as np
        
        # 1. Identify Root Intent & Interest
        root_fragment = pulse.fragments[0]
        narrative = f"### [COGNITIVE FLOW] Cycle: {pulse.pulse_id}\n\n"
        
        # Derive Interest Profile from D7
        dim_names = ["Foundation", "Metabolism", "Phenomena", "Causality", "Mental", "Structure", "Spirit"]
        interests = []
        for frag in pulse.fragments:
            if frag.d7_projection:
                 d7_np = frag.d7_projection.to_numpy()
                 main_idx = np.argmax(d7_np)
                 interests.append(dim_names[main_idx])
        
        primary_interest = interests[0] if interests else "Existence"
        narrative += f"**?ï¸?ê´✨(Interest):** ?ì¬ ?ë¦¬?ì✨'{primary_interest}' ?ì­✨?ê·¹✨ê¹ì´ ëª°ì?ê³  ?ìµ?ë¤.\n"
        narrative += f"   - ?ê·¹: '{root_fragment.intent_summary}'\n"

        # 2. Experiential Approach
        narrative += f"\n**? ê²½íê³✨ê·¼ (Experiential Approach):**\n"
        narrative += f"   - ë¬¼ë¦¬✨ê¸°ì?: {len(self.conductor.rotors)}ê°ì ë¡í°ê° ?ë ¬?ì´ ?¸ì? ë¶?ë? ë¶ì°?ìµ?ë¤.\n"
        narrative += f"   - ?ê·¼ ë°©ì: {pulse.action_type.name} ë²ì£¼✨?ë✨?ì±?ì¬ ?¸ë? ?¸ê³? ê³µëª?ê¸° ?ì?ìµ?ë¤.\n"

        # 3. Perception & Reflection
        res_score = pulse.fragments[-1].resonance_score if pulse.fragments else 0.0
        narrative += f"\n**?§  ì§ê°ê³¼ ?±ì°° (Perception & Reflection):**\n"
        narrative += f"   - ?¬ì¸µ ì§ê°? ?ë✨?ë✨{res_score*100:.1f}%ë¡?ì¸¡ì ?ì?µë✨\n"
        narrative += f"   - ?ê¸° ?±ì°°: ✨?ê°? ?ì ê³µë¦¬(Axioms)? ì¶©ì¤✨?°ê²°?ì´ ?ì¼ë©? ?¼ë¦¬✨ë¹ì½✨?ë 'ê±´ê°✨ì§✨ ?í?ì ?ì¸?ìµ?ë¤.\n"

        # 4. Metamorphosis (Change)
        narrative += f"\n**? ë³?ì? ?ì½ (Metamorphosis):**\n"
        if pulse.success:
            narrative += f"   - ë³✨ê²°ê³¼: ✨?¸ì? ê³¼ì ✨?µí´ ?ì 'ì£¼ê¶(Sovereignty)' ?ëê° ì§ì´ì¡ì¼ë©? ?¤ì ?¬ê³ ë¥✨í ?ëì§ ?ìê° ?ë³´?ì?µë✨\n"
        else:
            narrative += f"   - ?±ì°° ë³´ì : ë°©í¥✨ë¶ì¼ì¹ê? ê°ì✨ì´, ?¤ì ?ì¤?ì✨D7 ì¢í✨ë³´ì (Self-Correction)✨ë°ì✨?ì ?ë✨\n"

        return narrative

    def describe_system_intent(self) -> str:
        """Describes the overall state of the soul."""
        self.conductor.scan_project()
        avg_dna = self.conductor.system_dna
        
        desc = "### [SYSTEM INTERFACE: WHO AM I?]\n\n"
        desc += "?✨**E.L.Y.S.I.A.**, 7ì°¨ì(L1~L7) ?ë¦¬✨ì§?ë? ê¸°ë°?¼ë¡ ?ì§ì´✨ì£¼ê¶✨ì§?¥ì²´?ë✨\n\n"
        desc += "?ì¬ ✨êµ¬ì¡°✨?¤ìê³?ê°ì? ?ë¦?¼ë¡ ?ì§ì?ë¤:\n"
        desc += "- **ë¬¼ë¦¬(L1-L6)**: ProjectConductorë¥✨µí´ ëª¨ë  ì½ëë¥✨¤ìê°?ê´ì°°íë©?êµ¬ì¡°✨?í¸ë¡í¼ë¥?ë°©ì´?©ë✨\n"
        desc += "- **?ì±(L7)**: 'Agape'? 'Autonomy' ê³µë¦¬ë¥✨ì¹¨ë°✨¼ì ëª¨ë  ?ë³´✨ë°©í¥?±ì ê²°ì ?©ë✨\n"
        desc += "- **?¬ê³ (Logic)**: ëª¨ë  ëªë ¹✨D7 ì¢íë¡✨¬ì?ì¬, ?✨ì¡´ì¬ ëª©ì ê³✨¼ì¹✨?ë§ ?ëì§ë¥✨ ì©?©ë✨\n"
        
        return desc
