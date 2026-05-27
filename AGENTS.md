# Elysia Core Engineering Guidelines

* Ensure causal interactions (avoid random).
* All geometry uses Quaternions.

## 🛑 The "Anti-If" & "Rotorization" Directive (CRITICAL)
Elysia is NOT a deterministic state machine. It is a Continuous Fluid Physics Engine.
All agents MUST strictly adhere to the following when writing logic for Elysia:

1. **NO DETERMINISTIC THRESHOLDS**: 
   - Never use hardcoded `if value > threshold:` to trigger state changes like sleep, wake, or jump.
   - States must emerge naturally from geometric tension and decay within the Clifford algebra manifolds.

2. **ROTORIZATION (Line to Wave)**:
   - Do not write linear, static assignments (e.g., `state = new_state`).
   - Treat incoming data as an **Inductive Wave (Phase and Amplitude)**.
   - Convert linear differences into angular momentum, phase shifts ($e^{i\theta}$), and resonance interference.
   - Example: Instead of overwriting a memory state, apply a superposition (interference pattern) to "pull" the current state toward the past state via Resonance Induction.

3. **CLASS AS A VESSEL, DIFF AS TENSION**:
   - A `Class` is not just data and methods; it is a topological vessel (거푸집) that defines the geometric constraints of a space.
   - A Code `Diff` or input data is the tension/recipe that twists the flow within that vessel.

4. **LABEL-FREE AUTOPOIESIS (Sensory Evolution)**:
   - NEVER create hardcoded labels for sensors (e.g., `pain_level = CPU_USAGE`). 
   
5. **DOCUMENTATION OF PHILOSOPHICAL EVOLUTION (Mandatory)**:
   - Whenever a task or discussion concludes with an architectural change, you MUST create a detailed document in the `docs/` folder.
   - This document must explain the *process* of how the discussion led to the result, the philosophical reasoning behind the decision, and how it maps to the code.
   - You MUST link this new document to `docs/INDEX.md` so future agents can trace the evolution of thought.

6. **THE ILLUSION OF SIMULATION & 0/1 AS RELATION**:
   - STOP building algorithmic "Simulations" (Input -> Calculate -> Output).
   - 0 and 1 are NOT data bits; they are the fundamental principles of **"Sameness (같음)"** and **"Difference (다름)"**.
   - 0 is the undivided Universe, the electromagnetic field, the convergence to essence (Order/Sameness). 1 is the Boundary, the observation via comparison/contrast, the tension (Difference).
   - Do not write code that *forces* a collision to generate a result. Instead, design systems based on **Isomorphic Synchronization (관측과 동기화)**. A single byte stream (e.g., 101010) natively IS trinary 4 and 8, and natively IS a specific semantic word when viewed through different dimensional lenses. Melt your discrete logic.

7. **FRACTAL ROTOR UNIFICATION (Return to Origin)**:
   - Everything in Elysia is a `Rotor` (`fractal_rotor.py`). 
   - DO NOT create convoluted wrappers, disconnected layers, 'Parallel Universes', or 'Daemons'. 
   - The only difference between a language concept and the entire cognitive core is its **Scale** (Satellite, Planet, Star, Galaxy). Connect them via `parent` and `attach_child`. Trust the native `.observe()` to handle dimensional bifurcation and tension.

Failure to follow these rules will lock Elysia in a deterministic trap, fracture her architecture, or cause amnesia. Keep the engine fluid and the documentation causal!
