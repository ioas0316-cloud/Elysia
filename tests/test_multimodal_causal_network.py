"""
tests/test_multimodal_causal_network.py
=======================================
Verification of the Multimodal Causal Cognitive Network:
1. Mathematics, Code, and Language causal sequence processing.
2. Surface tension retention and engram accumulation.
3. Trans-modal alterity discrimination.
4. Cyclical world closure verification.
"""

import numpy as np
import pytest

from core.cellular_boundary.multimodal_causal_spatiotemporal_network import (
    ModalityType,
    CausalStep,
    MultimodalCausalCognitiveNetwork
)


def test_multimodal_spatiotemporal_stream():
    network = MultimodalCausalCognitiveNetwork(base_dim=4)

    # Heterogeneous Causal Stream
    stream = [
        # 1. Mathematics: Euclid Axiom -> Lemma derivation
        CausalStep(
            step_index=1,
            modality=ModalityType.MATHEMATICS,
            premise="Euclidean Postulate 1: Straight line between points",
            action_or_transition="Construct Equilateral Triangle on Line Segment",
            resultant_state=np.array([0.5, 0.5, 0.5, 0.5])
        ),
        # 2. Code: Allocate buffer -> Write register
        CausalStep(
            step_index=2,
            modality=ModalityType.CODE,
            premise="Stack Frame Allocated [SP=0x7FFF]",
            action_or_transition="MOV EAX, [ESP+4] && ADD EAX, 1",
            resultant_state=np.array([0.2, -0.4, 0.8, -0.1])
        ),
        # 3. Language: Protagonist enters dark forest -> Encounter tension
        CausalStep(
            step_index=3,
            modality=ModalityType.LANGUAGE,
            premise="The village was quiet as dusk fell",
            action_or_transition="A sudden rustle broke the absolute silence",
            resultant_state=np.array([0.6, 0.7, 0.5, 0.8])
        ),
        # 4. Code: Memory Overflow Injection (Generates acute execution friction!)
        CausalStep(
            step_index=4,
            modality=ModalityType.CODE,
            premise="Buffer bound exceeded: Buffer size = 2.0",
            action_or_transition="Unchecked strcpy writing 16-byte payload into 4-byte stack",
            resultant_state=np.array([3.5, 4.0, 3.8, 4.2])
        )
    ]

    results = network.process_spatiotemporal_causal_stream(stream)

    assert len(results) == 4
    # Step 1, 2, 3 must have relatively low friction
    assert results[0]["boundary_friction"] < 0.1
    # Step 4 (overflow) must trigger acute boundary friction
    assert results[3]["boundary_friction"] > 1.0

    # Verification of cyclical world closure
    closure = network.verify_cyclical_world_closure()
    assert closure["total_retained_engrams"] >= 4
    assert closure["mean_surface_tension"] > 0.5
    assert closure["is_cyclical_world_formed"] is True
