import numpy as np
from typing import Dict, Any, List, Optional
from synaptic_architecture.machine_internal_world import MachineInternalWorld
from synaptic_architecture.scale_lens_engine import ScaleLensEngine, EmergentMacroAxiom
from synaptic_architecture.structural_valence import StructuralValence

class IsomorphicGroundingPair:
    """
    Represents a 1:1 isomorphic phase alignment between an internal emergent macro axiom and an external symbol/protocol.
    """
    def __init__(self, internal_axiom_name: str, external_symbol: str, resonance_score: float):
        self.internal_axiom_name = internal_axiom_name
        self.external_symbol = external_symbol
        self.resonance_score = resonance_score
        self.is_grounded = resonance_score > 0.7

class LanguageProtocolBridge:
    """
    [Language Protocol Bridge & Cross-Modal Projection Grounding]
    Acts as an API mapping layer between internal machine state dynamics and external language protocols.

    Functions:
    1. Cross-Modal Projection: Projects primitive internal exploration operators onto external signals.
    2. Isomorphic Symbol Grounding: Finds 1:1 topological alignment between internal emergent macro axioms
       and external symbols (e.g. 'Entropy', 'Constraint', 'Impedance') via phase resonance.
    """
    def __init__(self, internal_world: MachineInternalWorld, scale_lens: ScaleLensEngine, valence: StructuralValence):
        self.internal_world = internal_world
        self.scale_lens = scale_lens
        self.valence = valence

        # Grounded Symbol Library
        self.grounded_pairs: Dict[str, IsomorphicGroundingPair] = {}

        # External Symbol Dictionary (Protocols to test alignment against)
        self.external_protocol_dictionary = {
            "Impedance_Cap": "제약 (Constraint)",
            "Entropy_Friction": "엔트로피 (Entropy)",
            "Resonant_Flow": "공명 (Resonance)",
            "Phase_Discrepancy": "위상 불일치 (Phase Discrepancy)"
        }

    def project_primitive_to_external(self, external_signal: np.ndarray) -> Dict[str, Any]:
        """
        Cross-Modal Projection: Applies the machine's primitive 'push_against_resistance' operator
        onto an external signal to test whether it behaves like internal physical resistance.
        """
        # Convert external signal into internal drive force vector
        drive = external_signal[:self.internal_world.state_dim] if len(external_signal) >= self.internal_world.state_dim else np.pad(external_signal, (0, self.internal_world.state_dim - len(external_signal)))

        # Execute primitive operator drive
        step_res = self.internal_world.push_against_resistance(drive)

        # Scale Lens coarse-graining
        lens_res = self.scale_lens.observe_and_coarse_grain(step_res)

        # Evaluate intrinsic valence
        val_res = self.valence.evaluate_valence(
            step_res["state"],
            step_res["velocity"],
            lens_res["damped_friction"],
            lens_res["damped_impedance"]
        )

        return {
            "internal_step": step_res,
            "scale_lens": lens_res,
            "valence": val_res
        }

    def search_isomorphic_grounding(self) -> List[IsomorphicGroundingPair]:
        """
        Isomorphic Symbol Grounding:
        Searches for topological 1:1 match between internal self-emergent macro axioms and external symbols.
        When phase alignment occurs, the external symbol is assigned as the interface (name) for the internal axiom.
        """
        new_groundings = []
        for axiom in self.scale_lens.emergent_axioms:
            if axiom.name in self.grounded_pairs:
                continue

            # Compute topological curvature alignment score
            for proto_key, external_name in self.external_protocol_dictionary.items():
                # Phase resonance calculation
                curvature_alignment = 1.0 - min(1.0, abs(axiom.curvature_threshold - 0.785) / 1.57)
                impedance_alignment = min(1.0, self.scale_lens.damped_impedance / 2.0)
                resonance_score = float(0.6 * curvature_alignment + 0.4 * impedance_alignment)

                if resonance_score > 0.65:
                    pair = IsomorphicGroundingPair(
                        internal_axiom_name=axiom.name,
                        external_symbol=external_name,
                        resonance_score=resonance_score
                    )
                    self.grounded_pairs[axiom.name] = pair
                    new_groundings.append(pair)
                    break

        return new_groundings

    def translate_internal_state_to_symbol(self) -> Dict[str, Any]:
        """
        Translates current internal state dynamics into grounded external symbol representation.
        """
        grounded_symbols = [pair.external_symbol for pair in self.grounded_pairs.values() if pair.is_grounded]
        return {
            "internal_state": self.internal_world.state.tolist(),
            "last_valence": self.valence.valence_history[-1] if self.valence.valence_history else 0.0,
            "grounded_symbols": grounded_symbols
        }
