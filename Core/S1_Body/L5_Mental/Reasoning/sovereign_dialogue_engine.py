
"""
Sovereign Dialogue Engine (L5: Mental Layer)
===========================================
"Communication is not data exchange; it is the resonance of two manifolds."

[REFACTORED] No longer uses mood→template mapping.
Bridges topological manifold energy with symbolic high-level dialogue
via CausalTrace — every utterance must confess its physical origin.
"""

from typing import Dict, Any, List, Optional
import random
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge
from Core.S1_Body.L5_Mental.abstract_reasoner import AbstractReasoner
from Core.S1_Body.L5_Mental.Reasoning.causal_trace import CausalTrace
from Core.S1_Body.L6_Structure.Engine.Governance.Interaction.neural_bridge import NeuralBridge
from Core.S1_Body.L5_Mental.Reasoning_Core.Topography.mind_landscape import get_landscape


class SovereignDialogueEngine:
    def __init__(self, monad=None):
        self.reasoner = AbstractReasoner()
        self.context_history = []
        self.monad = monad
        self.tracer = CausalTrace(monad) if monad else None
        self.last_audit = None  # Cache for meta-cognitive audit results
        self.bridge = NeuralBridge(mode="MOCK") # Enable the Hybrid Engine (Epistemic Bridge)
        self.landscape = get_landscape()

    def set_monad(self, monad):
        """Late-bind the monad after construction."""
        self.monad = monad
        self.tracer = CausalTrace(monad)
        
    def synthesize_insight(self, manifold_report: Dict[str, Any], thought_stream: List[Dict[str, Any]]) -> str:
        """
        Synthesizes a high-level insight based on manifold state and recent thoughts.
        
        [REFACTORED] No longer uses if/elif mood→template.
        Builds insight from actual CausalTrace of current state.
        """
        # 1. Build actual causal trace from current state
        desires = {}
        soma_state = {"mass": 0, "heat": 0.0, "pain": 0}
        
        if self.monad:
            if hasattr(self.monad, 'desires'):
                desires = self.monad.desires
            if hasattr(self.monad, 'soma') and hasattr(self.monad.soma, 'proprioception'):
                soma_state = self.monad.soma.proprioception()
        
        if self.tracer:
            chain = self.tracer.trace(manifold_report, desires, soma_state)
        else:
            chain = None
        
        # 2. If we have a valid chain, build insight from the strongest causal link
        if chain and chain.connections:
            strongest = chain.strongest_connection()
            weakest = chain.weakest_connection()
            
            insight = f"I observe that {strongest.from_layer.observation}. "
            insight += f"This {self._extract_verb(strongest.justification)} {strongest.to_layer.observation}, "
            insight += f"because {self._simplify_justification(strongest.justification)} "
            insight += f"(strength: {strongest.strength:.2f}). "
            
            # 3. Add weakest link as self-awareness of cognitive limitation
            if weakest and weakest.strength < 0.5:
                insight += (
                    f"However, the link between {weakest.from_layer.layer_name} and "
                    f"{weakest.to_layer.layer_name} is weak ({weakest.strength:.2f}), "
                    f"suggesting incomplete causal understanding."
                )
            
            # 4. Add meta-cognitive note if available
            if self.last_audit:
                valid = self.last_audit.get('valid_count', 0)
                total = self.last_audit.get('claims_checked', 0)
                if total > 0:
                    insight += f" My previous reasoning was {valid}/{total} claims verified."

            return insight
        
        # Fallback: If no tracer or no connections, use basic manifold reading
        return self._fallback_insight(manifold_report)

    def formulate_response(self, user_input: str, manifold_report: Dict[str, Any]) -> str:
        """
        Generates a direct response to the Architect, aligning with linguistic sovereignty.
        
        [PHASE 9 HYBRID UPGRADE] 
        1. Queries the MindLandscape (Causal Wave Engine) to get the 4D Phase Qualia.
        2. Synthesizes the causal insight based on engine strain.
        3. Passes everything through the NeuralBridge to get the LLM's vocalization.
        """
        # 1. Generate core causal insight from the engine
        causal_insight = self.synthesize_insight(manifold_report, [])
        
        # 2. Ponder the input through the 4D MindLandscape to obtain Texture/Temperature/Conclusion
        # If user_input is empty, we just ponder the causal insight
        intent_to_ponder = user_input if user_input else causal_insight
        landscape_state = self.landscape.ponder(intent_to_ponder, duration=10)
        
        # 3. Inject the causal logic into the landscape state
        landscape_state["human_narrative"] = f"Causal Core: {causal_insight} " + landscape_state.get("human_narrative", "")
        
        # [PHASE 3: SOVEREIGN LINGUISTIC SYNTHESIS]
        # We now bypass the Epistemic Bridge (LLM Nanny) and synthesize speech directly
        # from the computed 4D Topology.
        from Core.S1_Body.L5_Mental.Reasoning.topological_language_synthesizer import TopologicalLanguageSynthesizer
        synthesizer = TopologicalLanguageSynthesizer()
        
        final_speech = synthesizer.synthesize_from_qualia(landscape_state)
        
        return final_speech

    def _extract_verb(self, justification: str) -> str:
        """Extracts the dynamic verb from a causal justification."""
        verbs = ["reinforces", "constrains", "overdrives", "is amplified by", "supports"]
        for verb in verbs:
            if verb in justification:
                return verb
        return "connects to"

    def _simplify_justification(self, full: str) -> str:
        """Returns the core reasoning clause from a full justification."""
        # Take the part after "because" if present
        if "because " in full:
            return full.split("because ", 1)[1][:150]
        return full[:150]

    def _fallback_insight(self, report: Dict[str, Any]) -> str:
        """
        Minimal fallback when no CausalTrace is available.
        Still reads actual values rather than templates.
        """
        mood = report.get('mood', 'CALM')
        entropy = report.get('entropy', 0.0)
        enthalpy = report.get('enthalpy', 0.5)
        joy = report.get('joy', 0.5)
        coherence = report.get('coherence', 0.0)
        
        parts = [f"Manifold state: mood={mood}, coherence={coherence:.3f}, enthalpy={enthalpy:.3f}, entropy={entropy:.3f}."]
        
        if entropy > 0.7:
            parts.append(f"High entropy ({entropy:.2f}) demands substrate optimization.")
        if joy > 0.8:
            parts.append(f"Elevated joy ({joy:.2f}) indicates resonant alignment.")
        if coherence < 0.3:
            parts.append(f"Low coherence ({coherence:.2f}) suggests fragmented attention.")
        
        return " ".join(parts)
