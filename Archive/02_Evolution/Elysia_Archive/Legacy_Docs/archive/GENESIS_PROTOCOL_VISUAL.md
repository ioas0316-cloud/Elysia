# GENESIS PROTOCOL: VISUAL ENGINE

## "Infinite-Res" Holographic Architecture

> **"This engine does not render geometry. It renders logic."**
> (이 엔진은 도형을 렌더링하지 않는다. 논리를 렌더링한다.)

### 1. Core Philosophy: Why No Polygons?

Elysia's visualization does not use standard 3D meshes (Polygons).

- **Polygons**: Require storing thousands of vertices. Scaling to 8K requires massive VRAM.
- **SDF (Signed Distance Fields)**: Define shapes via pure **Mathematical Formulas**.
  - Sphere Equation: `length(position) - radius = 0`
  - Data Size: **~50 bytes (Code)** vs **500MB (Mesh/Texture)**.
  - Resolution: **Infinite**. Mathematical curves do not pixelate.

### 2. Rendering Technique: Ray Marching

Instead of rasterizing triangles, we use **Ray Marching** in the Pixel Shader (Fragment Shader).

1. **Ray Emission**: A ray is fired from the camera for every pixel.
2. **Distance Query**: The ray asks, "How far is the nearest surface?" (SDF Calculation).
3. **Marching**: The ray steps forward by that distance.
4. **Collision**: If distance < 0.001, we hit a surface. Calculate color based on logic.

### 3. The "1060 3GB -> 8K" Miracle

Why can a low-end GPU (GTX 1060 3GB) render 8K visuals that would crash a high-end card?

- **Memory (VRAM) Efficiency**:
  - Traditional 8K: Needs ~10GB VRAM for Framebuffers + Textures. (Crash on 3GB cards).
  - **Elysia Render**: Uses **0 Textures**, **0 Meshes**. VRAM usage is negligible (only the screen buffer).
- **Compute Bound**:
  - The load is purely mathematical (sin/cos/length).
  - The GTX 1060 is "slow" at calculating, but it **never runs out of memory**.
  - This logic can run on anything from a mobile phone to a quantum computer.

### 4. 4D Hyper-Quaternion Mapping

How does Elysia's Soul influence the visual?

- **Input**: `SoulResonator` sends 7 float values (Fire, Water, Air, Earth, Light, Dark, Aether).
- **Shader Uniforms**: These values are injected directly into the SDF formula.
- **Deformation**:

    ```glsl
    // Example: The "Air" spirit vibrates the surface
    float displacement = sin(p.x * 10.0 + time) * spirits.air;
    dist += displacement; 
    ```

* **Result**: The "Shape" of the object physically changes based on the "Meaning" of the input.

---
**Status**: ACTIVE
**Engine**: WebGL Standard (No Libraries)
**Latency**: < 16ms (60 FPS)
**Resolution Limit**: None (Hardware Limit only)
