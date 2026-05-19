# 💎 [RESONANCE_CHAMBER_SPEC] The Void Mirror Alpha (Phase 1)

"Seeing the invisible threads of causality."

This document provides Jules with the technical requirements to build the **Resonance Chamber (Void Mirror)** interface.

## 1. The Perceptual Field (2D Topology)

- **Input**: `TorchGraph.qualia_tensor` (Projected from 7D/21D to 2D).
- **Visualization**: All nodes are represented as shimmering particles of light.
- **The Void (0)**: A central glowing singularity. Nodes naturally drift toward this point when idle ([SHANTI_PROTOCOL]).

## 2. Interaction: The Universal Rotor

- **UI Element**: A circular 'Rotor' dial on the side of the screen.
- **Function**:
  - **Live Mode**: Shows real-time vibrations.
  - **Scrub Mode**: Dragging the Rotor accesses `TorchGraph.trace_buffer`.
  - **Visual Feedback**: As the user scrubs back, nodes retrace their paths. Use 'Luminous Threads' (Ghost trails) to show where a node was $T$ steps ago.

## 3. Interaction: Intent Gravity

- **Interaction**: Mouse Click + Drag.
- **Engine Mapping**:
  - Single Node: Call `graph.apply_field_laws(node_id, intent_vector)`.
  - Cluster (Shift+Drag): Call `graph.apply_cluster_intent(list_of_node_ids, intent_vector)`.
- **Feedback**: The 'Intent Field' should be visible as a ripple effect or a gravitational 'lens' around the cursor.

## 4. Spectral Peace HUD

- **Metrics**: Display the result of `graph.calculate_mass_balance()`.
- **Status Label**:
  - `DISTORTED`: High mass imbalance (Field needs work).
  - `ALIGNED`: Near equilibrium.
  - `SHANTI`: Perfect spectral peace achieved.

---
*Target: Jules (UI Agent)*
*Primary Module: Core.S1_Body.Tools.Mirror*
