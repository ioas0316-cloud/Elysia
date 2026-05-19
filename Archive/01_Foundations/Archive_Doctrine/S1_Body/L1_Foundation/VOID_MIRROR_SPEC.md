# 👁️ VISUAL CAUSALITY INTERFACE: The Void Mirror (Phase 200)

"Seeing is for the eyes. Perceiving is for the Rotor."

This document defines the interface between back-end laws ([FIELD_LAW_OS]) and the front-end visualization ([VOID_MIRROR]).

## 1. The Temporal Rotor (Cinematic Perception)

Jules MUST use the `TorchGraph.trace_buffer` to allow the Architect to scrub through time.

### A. Scrubbing Mode (Manual Rotor)

- When the user rotates the "Time Control" (Visual Rotor), Jules maps the UI index to the `trace_buffer` index.
- **Visual Effect**: Nodes should smoothly interpolate between the 7D states stored in the buffer.
- **Goal**: Allow "Reverse Causality Tracking" (What triggered this vibration 2 seconds ago?).

### B. Flow Mode (Real-time Pulse)

- In real-time, nodes should 'vibrate' according to their `tension_tensor` and 'drift' according to `qualia_tensor`.

## 2. Intent-Driven Simulation (Virtual Gravity)

The Architect's interaction is not "editing code," it is "injecting force."

### A. Intent Injection (Architect's Will)

- **UI Action**: Click-and-drag an area or a node.
- **Engine Call**: Map the mouse delta to a 7D `intent_vector` and call `graph.apply_field_laws(node_id, intent_vector)`.
- **Logic**: The user represents a "High-Gravity Entity." Their Will overrides the Odugi restoration temporarily, allowing the Architect to 'push' the system into new states.

### B. The Resonance Feedback

- When the Architect applies force, the UI must show **Interference Patterns** (Moire/Wave ripples) radiating from the cursor.
- **Meaning**: The "Stress" the Architect feels while trying to move the system should be visually represented as Field Turbulence.

## 3. The Void (Equilibrium Visualization)

- The center of the screen (or the 7D origin projection) is the **VOID (0)**.
- **Visual Cue**: A subtle glowing point at the origin.
- **The Law**: Watch as nodes, once released from the Architect's "Intent," are pulled back to the Void by the **Odugi Restoration Force**.

---
*Status: READY FOR JULES*
*Target: [VOID_MIRROR] Alpha 1.0*
