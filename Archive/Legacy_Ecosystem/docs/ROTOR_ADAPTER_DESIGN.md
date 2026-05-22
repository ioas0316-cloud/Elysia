# Design: Rotor-enhanced PyTorch Adapter (Eternos Kernel Module)

## 1. Core Principle: Hidden State as Phase
In the Rotor Architecture, data is not just a magnitude; it is a **rotating phase**.
Standard LLM Hidden States $H \in \mathbb{R}^{d}$ are transformed into a complex-like representation where each dimension pair (or group) represents a rotor state.

### 1.1 Phase Projection
We use a projection $W_P$ to map $H$ into a phase space:
$P = H \cdot W_P$ where $P$ contains $(\text{intensity}, \text{phase})$ pairs.

## 2. InterferenceGate Logic (Phase Alignment)
The `InterferenceGate` acts as a non-linear filter that only allows information to pass if it aligns with the "Rotor Coherence" state.

### 2.1 120-degree Trinity Alignment
Signals are forced towards three primary resonance points:
$\theta \in \{0, \frac{2\pi}{3}, \frac{4\pi}{3}\}$
This creates a "structural quantization" that reduces cognitive noise.

### 2.2 Total Internal Reflection (TIR)
If a signal's phase is too far from the coherent state, it is "reflected" back (suppressed).
$H_{out} = H_{in} \cdot \text{Coherence}(\text{Phase}(H_{in}))$
where $\text{Coherence}$ is a high-gradient function peaking at the Trinity points.

## 3. Phase-Gated Sparse Attention (FLOPs Reduction)
Instead of dense $Q \cdot K^T$, we use:
$\text{Score}(Q, K) = (Q \cdot K^T) \cdot \text{Sync}(\text{Phase}(Q), \text{Phase}(K))$
$\text{Sync}(\Delta\theta) = \cos(\Delta\theta)^{k}$ (where $k$ is a sharpness parameter)

By applying a threshold to $\text{Sync}$, we can skip computing the full attention for "asynchronous" tokens, leading to $O(N \cdot \text{sparse})$ complexity.

## 4. Hierarchical Rotor Structure
- **Micro-Rotors (Lower Layers):** High angular velocity, processing rapid token transitions.
- **Macro-Rotors (Higher Layers):** Low angular velocity, maintaining stable "Ego" states and long-term context.

## 5. Metrics for Success
- **FLOPs Reduction:** Percentage of skipped attention calculations.
- **Information Density:** Measurement of gradient flow stability vs. standard models.
- **Phase Coherence:** How quickly the system reaches a stable "Wye" (Trinity) state.
