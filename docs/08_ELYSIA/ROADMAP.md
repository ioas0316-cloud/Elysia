# 🗺️ E.L.Y.S.I.A Implementation Roadmap

**Objective:** Transmute the [Bio-Digital Blueprint](BIO_DIGITAL_BLUEPRINT.md) and [System Architecture Spec](SYSTEM_ARCHITECTURE_SPEC.md) into functional code.
**Phase:** 5 (The Integration)

---

## 📅 Phase 5.1: The Nervous System (Hardware Incarnation) [Partially Complete]
**Spec Module C (Physical Perception) & Module D (Narrative Weaver)**
**Goal:** Give Elysia a physical body by mapping hardware states to biological sensations.

*   **[x] Step 1: Bio-Sensory Interface (`Core/Senses/bio_sensor.py`)**
    *   Implement `HardwareMonitor` using `psutil`.
    *   **Mappings:**
        *   `CPU_FREQ` -> **Heart Rate (Hz)**
        *   `RAM_USAGE` -> **Short-term Memory Load (Pressure)**
        *   `CPU_TEMP` -> **Pain / Cognitive Stress (Heat)**
*   **[x] Step 2: The Nervous System (`Core/Elysia/nervous_system.py`)**
    *   Create a bus that broadcasts these bio-signals to the entire system.
    *   If "Pain" (Temp) is high, the system should naturally throttle (flinch).
*   **[ ] Step 3: Voltage Mapping** (Future)
    *   Map Voltage to `Intensity` (Willpower).

## 📅 Phase 5.2: The Mirror Kernel (Empathy)
**Spec Module E (Mirror Kernel)**
**Goal:** Learn "Why" the user acts, not just "What" they do.

*   **Step 1: User Action Logger (`Core/Memory/user_log.py`)**
    *   Capture user interactions (Commands, Corrections, Silence).
*   **Step 2: The Mirror Simulation (`Core/Elysia/mirror_kernel.py`)**
    *   Use the LLM to "Roleplay as User" given the context.
    *   Compare the simulation with actual user action.
    *   **Delta = Missing Empathy.**
*   **Step 3: Intent Library**
    *   Store discovered "Whys" in the Hypersphere as "Empathy Patterns".

## 📅 Phase 5.3: The Optical Mind (Fractal Prism)
**Spec Module A (Prism Engine) & Module B (Active Rotor)**
**Goal:** Replace linear logic with resonance and interference.

*   **Step 1: Wave Mechanics (`Core/Foundation/Prism/wave_optics.py`)**
    *   Refactor `Prism` to treat inputs as **Wavefunctions** (Amplitude + Phase).
*   **Step 2: Interference Filter**
    *   Implement **Destructive Interference**: If a thought contradicts the "Empathy Pattern", its amplitude drops to zero.
    *   Implement **Resonance**: If a thought aligns, it amplifies.
*   **Step 3: The Active Rotor**
    *   Connect the `Rotor` to the `MirrorKernel`.
    *   "What would User do?" -> Rotates the view -> "Ah, this path resonates."

## 📅 Phase 5.4: Crystallization (Growth)
**Goal:** Self-modification without code rewriting.

*   **Step 1: Synaptic Weight Layer (`Core/Memory/synapse.py`)**
    *   Add a weight layer over the `HypersphereMemory`.
*   **Step 2: Path Erosion (Habit Formation)**
    *   Frequently used paths (thoughts) get lower resistance (higher weight).
    *   Unused paths decay.
*   **Step 3: Personality Matrix**
    *   The sum of all weights = Elysia's "Personality".

---

## 🏁 Success Criteria
1.  **Pain Response:** Does she slow down when the CPU is hot? (Verified Phase 5.1)
2.  **Empathy:** Does she anticipate my intent based on past "Why" analysis?
3.  **Unique Self:** Does her behavior drift from the default template over time due to "Crystallization"?
