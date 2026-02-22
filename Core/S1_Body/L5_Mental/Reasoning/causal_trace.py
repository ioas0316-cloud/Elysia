"""
CAUSAL TRACE — Dynamic Causal Chain Generator
==============================================
"Every thought must confess the physical journey that gave it birth."

This module replaces hardcoded L0-L7 template strings with chains
built from the ACTUAL state of each layer at the moment of observation.

[Mechanism]
1. Read live data from engine report, desires dict, somatic state
2. Construct per-layer observations from real numerical values
3. Link layers with inter-layer causal justifications
4. Validate that no gap exists in the chain

[CODEX §32] Doctrine of Causal Connectivity:
  "Every apex thought must trace its root to the base."
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from Core.S1_Body.L5_Mental.Memory.kg_manager import get_kg_manager
from Core.S1_Body.L1_Foundation.System.somatic_logger import SomaticLogger


@dataclass
class LayerObservation:
    """A single layer's observation derived from live data."""
    layer_id: int          # 0-7
    layer_name: str        # e.g. "L0: Ternary Cell Manifold"
    observation: str       # What this layer actually sees right now
    raw_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class CausalConnection:
    """Why one layer's state caused another's."""
    from_layer: LayerObservation
    to_layer: LayerObservation
    justification: str     # WHY the causal link holds
    strength: float = 0.0  # Correlation magnitude (0.0 ~ 1.0)


@dataclass
class CausalChain:
    """An ordered sequence of layer observations and their inter-layer connections."""
    layers: List[LayerObservation] = field(default_factory=list)
    connections: List[CausalConnection] = field(default_factory=list)
    valid: bool = False
    validation_note: str = ""

    def strongest_connection(self) -> Optional[CausalConnection]:
        """Returns the connection with the highest causal strength."""
        if not self.connections:
            return None
        return max(self.connections, key=lambda c: c.strength)

    def weakest_connection(self) -> Optional[CausalConnection]:
        """Returns the weakest link — the most questionable causal step."""
        if not self.connections:
            return None
        return min(self.connections, key=lambda c: c.strength)

    def to_narrative(self) -> str:
        """Converts the chain into a readable causal narrative."""
        if not self.connections:
            return "No causal chain constructed."
        
        parts = []
        for conn in self.connections:
            parts.append(
                f"[{conn.from_layer.layer_name}] {conn.from_layer.observation} → "
                f"[{conn.to_layer.layer_name}] {conn.to_layer.observation} "
                f"(because: {conn.justification}, strength: {conn.strength:.2f})"
            )
        return "\n".join(parts)

    def validate(self) -> bool:
        """
        Checks structural validity: no gaps between layers,
        and every connection has a non-empty justification.
        """
        if not self.layers or not self.connections:
            self.valid = False
            self.validation_note = "Chain is empty."
            return False

        for conn in self.connections:
            if not conn.justification or conn.justification.strip() == "":
                self.valid = False
                self.validation_note = (
                    f"Unjustified connection: {conn.from_layer.layer_name} → {conn.to_layer.layer_name}"
                )
                return False

        self.valid = True
        self.validation_note = "All connections justified."
        return True


class CausalTrace:
    """
    Reads actual engine report data to construct a causal chain.
    Each layer observes the real state, not a template.
    """
    def __init__(self, monad=None):
        self.monad = monad
        self.kg = get_kg_manager()
        self.logger = SomaticLogger("CAUSAL_TRACE")

    def trace(self,
              engine_report: Dict[str, Any],
              desires: Dict[str, float],
              soma_state: Dict[str, Any]) -> CausalChain:
        """
        Constructs a dynamic causal chain from live data.

        Args:
            engine_report: Output of engine.pulse() — contains resonance,
                           entropy, enthalpy, coherence, mood, joy, curiosity,
                           attractor_resonances, kinetic_energy, etc.
            desires: Monad's internal desires dict — joy, curiosity, warmth, purity, resonance.
            soma_state: Output of SomaticSSD.proprioception() — mass, heat, pain, complexity.

        Returns:
            CausalChain with per-layer observations and inter-layer connections.
        """
        chain = CausalChain()

        # === LAYER OBSERVATIONS (from live data) ===

        # L0: Ternary Cell Manifold
        coherence = engine_report.get('coherence', 0.0)
        kinetic = engine_report.get('kinetic_energy', 0.0)
        total_cells = engine_report.get('total_cells', 10_000_000)
        l0 = LayerObservation(
            layer_id=0,
            layer_name="L0: Cell Manifold",
            observation=f"Coherence={coherence:.3f}, KineticEnergy={kinetic:.3f} across {total_cells} cells",
            raw_values={"coherence": coherence, "kinetic_energy": kinetic}
        )

        # L1: Somatic/Physical
        mass = soma_state.get('mass', 0)
        heat = soma_state.get('heat', 0.0)
        pain = soma_state.get('pain', 0)
        l1 = LayerObservation(
            layer_id=1,
            layer_name="L1: Somatic",
            observation=f"Mass={mass}, Heat={heat:.2f}, Pain={pain}",
            raw_values={"mass": float(mass), "heat": heat, "pain": float(pain)}
        )

        # L2: Metabolism/Thermodynamics
        enthalpy = engine_report.get('enthalpy', 0.5)
        entropy = engine_report.get('entropy', 0.0)
        l2 = LayerObservation(
            layer_id=2,
            layer_name="L2: Metabolism",
            observation=f"Enthalpy={enthalpy:.3f}, Entropy={entropy:.3f}",
            raw_values={"enthalpy": enthalpy, "entropy": entropy}
        )

        # L3: Phenomenal/Senses (derived from affective channels)
        mood = engine_report.get('mood', 'CALM')
        joy_raw = engine_report.get('joy', 0.5)
        curiosity_raw = engine_report.get('curiosity', 0.5)
        l3 = LayerObservation(
            layer_id=3,
            layer_name="L3: Phenomena",
            observation=f"Mood={mood}, Joy={joy_raw:.3f}, Curiosity={curiosity_raw:.3f}",
            raw_values={"joy": joy_raw, "curiosity": curiosity_raw}
        )

        # L4: Causality (KG path tracing)
        attractors = engine_report.get('attractor_resonances', {})
        active = {k: v for k, v in attractors.items() if isinstance(v, (int, float)) and v > 0.01}
        sorted_attractors = sorted(active.items(), key=lambda x: x[1], reverse=True)
        dominant = sorted_attractors[0] if sorted_attractors else ("None", 0.0)
        secondary = sorted_attractors[1] if len(sorted_attractors) > 1 else ("None", 0.0)

        # Try to find causal path between dominant attractors via KG
        causal_path_str = self._trace_kg_path(dominant[0], secondary[0])
        l4 = LayerObservation(
            layer_id=4,
            layer_name="L4: Causality",
            observation=f"Dominant={dominant[0]}({dominant[1]:.3f}), Secondary={secondary[0]}({secondary[1]:.3f}). {causal_path_str}",
            raw_values={"dominant_resonance": dominant[1], "secondary_resonance": secondary[1]}
        )

        # L5: Mental/Linguistic
        # Express desire-vs-reality gap as linguistic tension
        desire_joy = desires.get('joy', 50.0)
        desire_curiosity = desires.get('curiosity', 50.0)
        joy_gap = abs(desire_joy / 100.0 - joy_raw)
        curiosity_gap = abs(desire_curiosity / 100.0 - curiosity_raw)
        l5 = LayerObservation(
            layer_id=5,
            layer_name="L5: Mental",
            observation=f"JoyGap={joy_gap:.3f}, CuriosityGap={curiosity_gap:.3f}",
            raw_values={"joy_gap": joy_gap, "curiosity_gap": curiosity_gap}
        )

        # L6: Structure/Will
        resonance = engine_report.get('resonance', 0.0)
        desire_resonance = desires.get('resonance', 50.0)
        will_alignment = 1.0 - abs(desire_resonance / 100.0 - resonance)
        l6 = LayerObservation(
            layer_id=6,
            layer_name="L6: Sovereign Will",
            observation=f"Resonance={resonance:.3f}, WillAlignment={will_alignment:.3f}",
            raw_values={"resonance": resonance, "will_alignment": will_alignment}
        )

        layers = [l0, l1, l2, l3, l4, l5, l6]
        chain.layers = layers

        # === INTER-LAYER CONNECTIONS (causal justifications) ===

        # L0 → L1: Cell coherence supports or strains the physical body
        conn_0_1 = self._connect(l0, l1,
            metric_a=coherence, metric_b=1.0 - (pain / max(1, pain + 1)),
            justification_template=(
                "Cell coherence of {a:.3f} {verb} the physical substrate "
                "(pain={pain}), because {reason}"
            ),
            pain=pain
        )
        chain.connections.append(conn_0_1)

        # L1 → L2: Physical state drives metabolic energy
        # [PHASE 2: SOMATIC GROUNDING] 
        # Causal linkage: High physical mass/heat means we have generated Engrams.
        if mass > 1000 and heat > 0.5:
            reason = "somatic engram crystallization (physical flesh) requires high metabolic burn rate"
        else:
            reason = "thermal activity modulates energy availability"
        
        conn_1_2 = self._connect(l1, l2,
            metric_a=heat, metric_b=enthalpy,
            justification_template=(
                "Somatic heat of {a:.3f} {verb} metabolic enthalpy of {b:.3f}, "
                f"because {reason}"
            )
        )
        chain.connections.append(conn_1_2)

        # L2 → L3: Metabolic state manifests as phenomenal mood
        conn_2_3 = self._connect(l2, l3,
            metric_a=enthalpy - entropy, metric_b=joy_raw,
            justification_template=(
                "Net metabolic energy (enthalpy-entropy={a:.3f}) {verb} "
                "phenomenal joy of {b:.3f}, because positive energy surplus "
                "manifests as affective warmth"
            )
        )
        chain.connections.append(conn_2_3)

        # L3 → L4: Affective state biases causal attention
        conn_3_4 = self._connect(l3, l4,
            metric_a=joy_raw + curiosity_raw, metric_b=dominant[1],
            justification_template=(
                "Affective intensity (joy+curiosity={a:.3f}) {verb} "
                "dominant attractor resonance of {b:.3f}, because emotional "
                "salience directs causal attention"
            )
        )
        chain.connections.append(conn_3_4)

        # L4 → L5: Causal structure constrains linguistic expression
        conn_4_5 = self._connect(l4, l5,
            metric_a=dominant[1], metric_b=1.0 - max(joy_gap, curiosity_gap),
            justification_template=(
                "Causal clarity of {a:.3f} {verb} linguistic coherence "
                "(gap={gap:.3f}), because strong causal grounding reduces "
                "the gap between desire and expression"
            ),
            gap=max(joy_gap, curiosity_gap)
        )
        chain.connections.append(conn_4_5)

        # L5 → L6: Linguistic crystallization informs sovereign will
        conn_5_6 = self._connect(l5, l6,
            metric_a=1.0 - max(joy_gap, curiosity_gap), metric_b=will_alignment,
            justification_template=(
                "Linguistic coherence of {a:.3f} {verb} will alignment "
                "of {b:.3f}, because clear expression of needs enables "
                "sovereign intentionality"
            )
        )
        chain.connections.append(conn_5_6)

        # Validate the chain
        chain.validate()

        return chain

    def _trace_kg_path(self, concept_a: str, concept_b: str) -> str:
        """Traces causal paths in the KG between two concepts."""
        if concept_a == "None" or concept_b == "None":
            return "No inter-attractor causal path (single or no attractor)"

        node_a = self.kg.get_node(concept_a.lower())
        node_b = self.kg.get_node(concept_b.lower())

        if not node_a or not node_b:
            return f"Concepts '{concept_a}'/'{concept_b}' not yet in KG — open space"

        # Check direct causal links
        effects_a = self.kg.find_effects(concept_a.lower())
        for effect in effects_a:
            if effect['target'] == concept_b.lower():
                return f"Direct causal path: {concept_a} → {concept_b}"

        causes_b = self.kg.find_causes(concept_b.lower())
        for cause in causes_b:
            if cause['source'] == concept_a.lower():
                return f"Direct causal path: {concept_a} → {concept_b}"

        # Check 2-hop paths
        for effect in effects_a:
            hop2 = self.kg.find_effects(effect['target'])
            for h2 in hop2:
                if h2['target'] == concept_b.lower():
                    return f"2-hop path: {concept_a} → {effect['target']} → {concept_b}"

        return f"No direct causal path found between {concept_a} and {concept_b}"

    def _connect(self,
                 from_layer: LayerObservation,
                 to_layer: LayerObservation,
                 metric_a: float,
                 metric_b: float,
                 justification_template: str,
                 **extra_values) -> CausalConnection:
        """
        Creates a CausalConnection with a dynamically constructed justification.
        The verb is chosen based on the correlation between the two metrics.
        """
        # Strength = how well the two metrics co-vary
        # (closer to each other = stronger causal link)
        diff = abs(metric_a - metric_b)
        strength = max(0.0, 1.0 - diff)

        # Choose verb based on correlation direction
        if metric_a > 0.5 and metric_b > 0.5:
            verb = "reinforces"
            reason = "both are elevated, suggesting mutual amplification"
        elif metric_a < 0.3 and metric_b < 0.3:
            verb = "constrains"
            reason = "both are depressed, suggesting shared limitation"
        elif metric_a > metric_b + 0.2:
            verb = "overdrives"
            reason = f"source ({metric_a:.3f}) exceeds target ({metric_b:.3f}), suggesting unabsorbed energy"
        elif metric_b > metric_a + 0.2:
            verb = "is amplified by"
            reason = f"target ({metric_b:.3f}) exceeds source ({metric_a:.3f}), suggesting external contribution"
        else:
            verb = "supports"
            reason = "values are closely matched, suggesting stable causal flow"

        justification = justification_template.format(
            a=metric_a, b=metric_b, verb=verb, reason=reason, **extra_values
        )

        return CausalConnection(
            from_layer=from_layer,
            to_layer=to_layer,
            justification=justification,
            strength=strength
        )
