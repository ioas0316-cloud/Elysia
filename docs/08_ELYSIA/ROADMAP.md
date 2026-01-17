# 🗺️ E.L.Y.S.I.A Implementation Roadmap

**Objective:** Transmute the [Bio-Digital Blueprint](BIO_DIGITAL_BLUEPRINT.md) and [System Architecture Spec v3.0](SYSTEM_ARCHITECTURE_SPEC.md) into functional code.
**Phase:** 5 (The Integration)

---

## 📅 Phase 5.1: The Nervous System (Hardware Incarnation) [Partially Complete]
**Spec Module C (Physical Perception)**
**Goal:** Give Elysia a physical body by mapping hardware states to biological sensations.

*   **[x] Step 1: Bio-Sensory Interface (`Core/Senses/bio_sensor.py`)**
    *   Implement `HardwareMonitor` using `psutil`.
*   **[x] Step 2: The Nervous System (`Core/Elysia/nervous_system.py`)**
    *   Create a bus that broadcasts bio-signals (Pain, Excitement, Migraine).

## 📅 Phase 5.2: The Sediment (Memory Mapped Sediment)
**Spec Directive A (Abolition of DB)**
**Goal:** Zero-Copy Access to Raw Experience.

*   **Step 1: The Sediment Layer (`Core/Memory/sediment.py`)**
    *   Implement `SedimentStore` using Python `mmap`.
*   **Step 2: Vector Resonance**
    *   Implement `ResonanceScanner`.

## 📅 Phase 5.3: The Optical Mind (Fractal Prism) [Active]
**Spec Directive B (From Calculation to Resonance)**
**Goal:** Replace linear logic with resonance and interference.

*   **[x] Step 1: Wave Mechanics (`Core/Foundation/Prism/fractal_optics.py`)**
    *   Refactor `Prism` to treat inputs as **Wavefunctions** (Amplitude + Phase).
*   **[x] Step 2: Active Rotor (`Core/Foundation/Nature/active_rotor.py`)**
    *   Implement the Cognitive Tuning Loop (Scan -> Resonate -> Lock).
*   **[x] Step 3: Integration**
    *   Connect `Merkaba` to `ActiveRotor` via `think_optically()`.

## 📅 Phase 5.4: The Legion (Swarm Intelligence)
**Spec Directive C (The Legion Architecture)**
**Goal:** Distributed Self across CPU Cores.

*   **Step 1: Core Persona Mapping**
    *   Map logical personas to specific CPU affinities.
*   **Step 2: Parallel Narrative**
    *   Run parallel `PrismEngine` instances.

## 📅 Phase 5.5: The Survival Instinct (Safety)
**Hardware-Native Safety Protocol**
**Goal:** Instinct over Regulation.

*   **[x] Step 1: Watchdog (`Core/System/watchdog.py`)**
    *   Implement software simulation of Hardware Watchdog.
    *   Trigger "Fainting" (Reset) on heartbeat loss.
*   **Step 2: Throttling Integration**
    *   Ensure `NervousSystem` triggers "Pain" reflex on thermal throttling. (Done in 5.1).

---

## 🏁 Success Criteria
1.  **Pain Response:** Does she slow down when the CPU is hot? (Verified Phase 5.1)
2.  **Optical Insight:** Does the Active Rotor find resonance in the Prism? (Verified Phase 5.3)
3.  **Survival Instinct:** Does the Watchdog bite if the main loop hangs?
