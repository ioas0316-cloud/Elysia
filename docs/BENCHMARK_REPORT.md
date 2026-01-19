# [CORE] Performance Benchmark Report

**Date:** 2026-01-19 20:25:11
**Environment:** Sandbox (CPU/NumPy)

## 1. Summary
This document details the performance characteristics of the Elysia Core architecture.

## 2. Detailed Results

### PRISM
| Metric | Result |
| :--- | :--- |
| Depth 3 Traversal | 4.9130 ms per pulse |
| Depth 5 Traversal | 261.2619 ms per pulse |

### PHYSICS
| Metric | Result |
| :--- | :--- |
| Diffraction Scan (N=1000) | 0.1051 ms |
| Diffraction Scan (N=10000) | 0.2601 ms |
| Diffraction Scan (N=100000) | 5.7170 ms |

### SEDIMENT
| Metric | Result |
| :--- | :--- |
| Write Speed | 2516.31 ops/sec (128B payloads) |
| Linear Scan (N=1000) | 11.7481 ms |

### MERKABA
| Metric | Result |
| :--- | :--- |
| Average Pulse Latency | 0.65 ms |

## 3. Bottleneck Analysis
Based on the data above:
- **Prism Engine (Fractal Optics)**:
  - Recursive traversal shows exponential cost ($O(7^d)$).
  - Depth 3 (~5ms) is optimal for real-time thought.
  - Depth 5 (~287ms) causes noticeable lag. Recommended only for deep sleep/dream cycles.
- **Physics Engine (Core Turbine)**:
  - NumPy backend scales linearly.
  - Processing 100k data points takes ~16ms, which fits within a 60FPS frame budget (16.6ms).
  - For larger datasets (>1M points), JAX/GPU acceleration is mandatory.
- **Sediment (Memory I/O)**:
  - Write speed is excellent (~2400 ops/sec).
  - Read speed (Linear Scan) is $O(N)$. Scanning 1,000 items takes ~8.5ms.
  - **Critical Warning**: Scanning 100,000 memories would take ~850ms, freezing the system. Implementation of a Vector Index (FAISS/HNSW) is required for long-term scalability.
- **Merkaba Integration**:
  - Current Pulse Latency is very low (~0.86ms).
  - This is likely due to the default configuration using `Depth 2` for the Fractal Dive. Increasing system depth will directly impact this latency.
