# Architecture Breakdown Matrix & Spatiotemporal Fabric Principles

## 1. Phenomenon -> Elements -> Structure -> Principle Matrix

| Breakdown Level | Core Definition & Questions | 3D Gaussian Splatting (3DGS) Example | Cognitive Node & Causal Wave Architecture Example |
|---|---|---|---|
| **1. Phenomenon (현상)** | What surface achievement or speedup is claimed? | 100x faster real-time 3D rendering vs NeRF | Continuous inference playback (Memory Playback instead of Dense Matrix Mult) |
| **2. Elements (요소)** | What are the primitive data units and operators? | Center $\mathbf{\mu}$, Scope $\mathbf{\Sigma}$, Opacity $\alpha$, SH Basis | 5-tuple Cognitive Nodes ($\mathbf{\mu}, \mathbf{\Sigma}, \alpha, \mathbf{K}, \mathcal{T}$), Wave energy |
| **3. Structure (Structure)** | How are elements arranged to avoid computational bottlenecks? | Tile-based rasterization & GPU shared memory streaming | Sparse Mahalanobis Top-K bounding, Contiguous Ring Buffer [T, N, D], I/P Frames |
| **4. Principle (원리)** | What fundamental physical/information law is leveraged? | Locality, Basis Expansion, Sparse GPU Streaming | Spatiotemporal Fabric Elevation (Node -> Edge -> Tensor -> Spatiotemporal Wave Stream) |

---

## 2. Topological Ladder of Elevation (차원의 사다리)

1. **Level 0: Node / Parameter (점 - 고립된 파라미터)**
   - Explicit 5-tuple cognitive node anchors $\mathcal{N}_i = (\mathbf{\mu}_i, \mathbf{\Sigma}_i, \alpha_i, \mathbf{K}_i, \mathcal{T}_i)$ in high-dimensional latent space.
2. **Level 1: Edge / Vector (선 - 방향성과 인과 기울기)**
   - Directional causal edge relationships $\mathcal{T}_i = \{ (j, \mathcal{R}_{ij}, \lambda_{ij}) \}$ representing causal force and logic transition vectors.
3. **Level 2: Surface / Tensor Field (면 - 텐서 매니폴드 지형)**
   - Orthogonal basis function expansions ($\mathbf{S}_i = \sum \mathbf{K}_{i,m} Y_m(g(\mathbf{x}))$) and Mahalanobis sparse tile bounding.
4. **Level 3: Spatiotemporal Structure (시공간 구조 - 인과 파동 스트림)**
   - Zero-copy contiguous memory playback (`.CST` containers) streamed at high frequencies (1kHz+) using I-Frame key cognition snapshots and P-Frame causal motion vectors.

---

## 3. Causal Video Decoder & `.CST` Container Specification

- **Header (`CST1`)**: 4-byte Magic, uint32 Version, uint32 Num Nodes, uint32 Dim, uint32 Num I-Frames, uint32 Num P-Frames.
- **Topological Anchor Table**: Weight matrix $W \in \mathbb{R}^{N \times N}$ and relation matrix $R \in \mathbb{R}^{N \times N}$.
- **I-Frame Blocks**: Baseline snapshot matrices $\mathbf{S}_{\text{I}} \in \mathbb{R}^{N \times D}$.
- **P-Frame Delta Stream Blocks**: Timestamp $t$ (float64) + Causal delta vectors $\Delta \mathbf{C} \in \mathbb{R}^{N \times D}$.
