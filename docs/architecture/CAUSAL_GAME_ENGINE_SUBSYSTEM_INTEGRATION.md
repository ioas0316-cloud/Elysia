# Causal Game Mechanics Engine Subsystem Architecture Specification

## Overview

This document specifies the integration layer architecture for porting the **Causal Game Mechanics Engine (Elysia)** into commercial game engine ecosystems: **Unreal Engine 5 (UE5)** and **Unity Engine**.

The architecture transitions game systems from hardcoded event scripts to a **Systemic Dynamic Engine** driven by dynamic friction tension ($V_t$) fields and Causal Conservation Nodes (**CC-Nodes**).

---

## 1. High-Level Integration Layer Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  Game Engine Layer (UE5 / Unity)                                       │
│  - Gameplay Actors / GameObjects / ECS Entities                        │
│  - Events: OnActorKilled, OnResourceExhausted, OnTerritoryCaptured    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Event Intercept & Delegate Call
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Causal Subsystem Layer (UWorldSubsystem / C# Engine Subsystem)        │
│  - State Bridge: Game Event ──> External Causal Signal Conversion       │
│  - Tension Monitor: Real-time V_t Evaluation Thread / GPU Dispatcher   │
│  - Phase Space Attractor Basin & Homeostasis Governor                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Thread-safe Lock-free Execution
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Causal Engine Core (C++ TaskGraph / C# DOTS Burst / GPGPU Compute)    │
│  - PerceptualLensController (Macro/Micro Scale Manager)                │
│  - CCNodeGraph (Topological Invariants & Generative Rules)             │
│  - SealedAttractorVault (Quarantine & Restructure Loop)                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dynamic Control Equations

### 2.1 Damped Tension Field Equation
To prevent numerical overflow and **Chaos Cascade** in systemic narrative loops, tension evolution for node $i$ is governed by:

$$\frac{dV_i}{dt} = \sum_{j \in \mathcal{N}(i)} w_{ij} |V_i - V_j| - \lambda V_i - \gamma \left( \frac{V_i}{V_{\text{critical}}} \right)^k$$

Where:
- $\lambda$: Natural relaxation coefficient.
- $\gamma, k$: Non-linear saturation resistance scale parameters ($k = 4.0$).
- $V_i \ge V_{\text{critical}} \times 1.2$: Hard limit for the **Emergency Circuit Breaker**.

### 2.2 Phase Space Potential Well Energy Equation
NPC Behavior Tree (BT) goals adapt dynamically by rolling into the lowest energy Attractor Basin in phase space:

$$E(p) = d_b \cdot (V_t - V_{\text{center}, b})^2 - (0.2 \cdot |\nabla V_t| + 0.1 \cdot I_{\text{drive}}) - h_b \cdot \delta(b, b_{\text{current}})$$

Where $h_b$ is the hysteresis friction to prevent behavior chattering.

### 2.3 Active Inference & Variational Free Energy
The agent minimizes variational free energy ($\mathcal{F}$) representing divergence between internal belief ($\pi$) and external observations ($o$):

$$\mathcal{F} = \text{Complexity} - \text{Accuracy} \approx \Delta S$$

- **Perception Gradient**: $-\nabla_{\text{Perception}} \mathcal{F}$ rebinds internal CC-Node topology.
- **Action Gradient**: $-\nabla_{\text{Action}} \mathcal{F}$ emits external game signals to entrain environment states.

### 2.4 Topological Homeostasis Boundary
To preserve invariant core identity ($\mathcal{C}_{\text{core}}$), the structural deformation potential $\mathcal{U}_{\text{core}}$ and non-linear damping coefficient $\sigma_H$ are calculated as:

$$\mathcal{U}_{\text{core}} = \frac{1}{2} \kappa_{\text{core}} \| \mathbf{L}_{\text{current}} - \mathbf{L}_{\text{core}} \|_F^2$$

$$\sigma_H(\mathcal{U}_{\text{core}}) = \exp \left( -\lambda \left( \frac{\mathcal{U}_{\text{core}}}{\mathcal{U}_{\text{max}}} \right)^p \right)$$

$$V_t^* = \sigma_H(\mathcal{U}_{\text{core}}) \cdot V_t^{\text{raw}} - \gamma \nabla \mathcal{U}_{\text{core}}$$

When $\mathcal{U}_{\text{core}} \ge \mathcal{U}_{\text{max}}$, $\sigma_H \to 0$ and active adaptation halts while elastic restoration returns the topology to its core norm.

---

## 3. Subsystem Binding Implementations

### 3.1 Unreal Engine 5 (C++) Architecture
- **UCausalWorldSubsystem** (`modules/causal_game_engine/unreal/Public/CausalWorldSubsystem.h`):
  Auto-binds to `UWorld` lifecycle. Receives gameplay events via `RegisterPlayerAction()`, computes world tension, and dispatches dynamic multicast delegates:
  - `OnSealedAttractorTriggered`: Notifies Blueprints when a node ruptures ($V_t > V_{\text{critical}}$).
  - `OnPerceptualScaleShifted`: Triggers macro/micro camera and spawning scale switching.
- **FCausalParallelEvaluator** (`modules/causal_game_engine/unreal/Public/CausalParallelEvaluator.h`):
  Uses UE5 `ParallelFor` and Task Graph thread pool partitioning over raw contiguous memory chunks (`FCCNodeRawData`) for 60+ FPS high-performance execution.

### 3.2 Unity Engine (C# & DOTS) Architecture
- **CausalEngineSubsystem** (`modules/causal_game_engine/unity/Scripts/CausalEngineSubsystem.cs`):
  MonoBehaviour singleton subsystem for signal interception and event broadcasting.
- **CausalEngineSystem & EvaluateTensionJob** (`CausalEngineSystem.cs`):
  Unity DOTS system with `IJobEntity` and Burst compiler SIMD vectorization over `CCNodeComponent` contiguous memory chunks.
- **AttractorAIController & EvaluateNPCPhaseSpaceJob** (`AttractorAIController.cs`, `EvaluateNPCPhaseSpaceJob.cs`, `EvaluateNPCPhaseSystem.cs`):
  Evaluates potential energy wells for thousands of NPCs with zero GC allocations using blittable structs and native arrays.
- **GpuTensionBufferProvider & DirectGpuSamplingSystem** (`GpuTensionBufferProvider.cs`, `DirectGpuSamplingSystem.cs`):
  Provides Zero-Copy direct GPU buffer sampling via persistent host-visible memory wrapping and triple ring buffering ($N, N-1, N-2$).
- **HomeostaticGovernorSystem & PhaseRuptureBridge** (`HomeostaticGovernorSystem.cs`, `PhaseRuptureBridge.cs`):
  Burst-compiled system executing atomic float reduction for $\mathcal{U}_{\text{core}}$ and updating post-processing URP materials when deformation exceeds 80%.

---

## 4. HLSL & URP Shader Pipeline

1. **CausalTensionEvaluator.compute**:
   - `CSMain_EvaluateTension`: Evaluates $V_t$ tension across 1,000,000+ GPU nodes and appends rupture indices to lock-free `AppendStructuredBuffer`.
   - `CSMain_EvaluateTensionWithDamping`: Applies damped tension decay, exponential saturation, and emergency circuit breaker.
   - `CSMain_RenderTensionFieldTexture`: Maps tension values zero-copy into a 2D heatmap texture (`_TensionFieldTexture`).
2. **TerrainDisplacementHLSL.shader**:
   - Samples $V_t$ at vertex shader stage via `SampleLevel(..., 0)`.
   - Displaces vertex normals proportionally to tension and applies non-linear emission bloom ($V_t^{2.5}$).
3. **PostProcessAnomaly.shader**:
   - Computes spatial gradient $\nabla V_t = (\text{ddx}(V_t), \text{ddy}(V_t))$ for screen-space lens refraction.
   - Applies horizontal slice tearing and chromatic aberration for $V_t > V_{\text{critical}}$.
4. **PhaseRuptureShift.shader**:
   - Fullscreen URP post-processing pass rendering topological glitch lines, chromatic aberration, pulse vignette, and color phase inversion when $\mathcal{U}_{\text{core}} / \mathcal{U}_{\text{max}} \ge 0.8$.

---

## 5. Summary Table

| Feature | Legacy Hardcoded Scripting | Elysia Causal Subsystem |
|---|---|---|
| Narrative Trigger | `if (count >= 10)` Hardcoded tree | Systemic dynamic emergence via $V_t$ field |
| Memory Layout | Pointer chasing (Cache Miss) | Contiguous chunk arrays & Blittable Structs (L1/L2 ~95% Hit) |
| CPU Threading | Main thread blocking | UE5 Task Graph / Unity DOTS Burst ParallelFor |
| GPU Interaction | CPU-to-GPU data copies | Zero-Copy direct memory access & Ring Buffering |
| AI Decision Making | Discrete FSM / Static BT | Phase space potential energy well auto-traversal |
| Homeostasis Safety | None (Potential data explosion) | Damped equation, saturation resistance & Topological homeostasis |
