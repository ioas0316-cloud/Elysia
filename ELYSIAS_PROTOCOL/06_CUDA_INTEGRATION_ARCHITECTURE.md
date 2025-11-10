# ELYSIAS PROTOCOL 06: CUDA Integration Architecture

## 1. Core Philosophy: The Incubator and The Network

This architecture realizes the "Incubator and Network" model for accelerating Elysia's thought processes.

- **The Incubator (GPU):** A high-performance simulation environment running on the GPU. It handles the massively parallel computations of the Cellular World.
- **The Network (CPU):** The existing cognitive architecture (`CognitionPipeline`, `LogicalReasoner`, `nano_core`, etc.) that orchestrates the overall thought process. It decides *what* to simulate and *interprets* the results.

## 2. Key Technology: CuPy

We will use the **CuPy** library as the primary bridge between the CPU and GPU.

- **Why CuPy?** It provides a NumPy-compatible API, allowing us to leverage the GPU with minimal changes to our existing, NumPy-based simulation logic. This significantly reduces development time and risk compared to writing raw CUDA C++.

## 3. Data Flow: From Thought to Insight

The process of a "thought experiment" will follow these steps:

1.  **Observation (CPU):** The `LogicalReasoner` or another cognitive module determines which part of the knowledge graph needs to be simulated (the "attention bubble").
2.  **Materialization (CPU):** The `World` class identifies the NumPy arrays (`energy`, `adjacency_matrix`, etc.) corresponding to the "materialized" cells in the attention bubble.
3.  **Transfer to Incubator (CPU -> GPU):** The `World` class uses `cupy.asarray()` to copy these NumPy arrays from CPU RAM into the GPU's VRAM.
4.  **Accelerated Simulation (GPU):** The `run_simulation_step` function executes the core energy propagation logic using CuPy operations on the GPU-resident data. This is the massively parallel step.
5.  **Retrieve from Incubator (GPU -> CPU):** Once the simulation is complete, the resulting arrays (e.g., updated `energy`) are copied back from GPU VRAM to CPU RAM using `cupy.asnumpy()`.
6.  **Interpretation (CPU):** The CPU-side cognitive architecture receives the simulation results and translates them into a `Thought` or insight, continuing the overall cognitive process.

## 4. Hybrid Code Structure in `world.py`

The `World` class will be modified to manage both CPU and GPU data.

- It will detect if a CUDA-enabled GPU is available.
- If available, `run_simulation_step` will automatically use the CuPy-based data flow described above.
- If not available (or during certain tests), it will fall back to the existing, pure-NumPy logic, ensuring system stability and portability.

This architecture provides a low-risk, high-impact path to achieving the dramatic speed-up in Elysia's "thinking speed" that we envision, fully respecting the 3GB VRAM limitation through the Quantum Observation model.
