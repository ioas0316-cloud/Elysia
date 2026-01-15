# Implementation Plan: Phase 24 (The Chronos Hierarchy)

We will implement **Time Dilation** by varying Rotor RPMs for different Monad Domains.

## 0. The Philosophy: Why Time Relativity?
>
> **"속도가 다르면, 역할이 달라진다."**

1. **Stability (Memory)**:
    * If Data changes as fast as Thoughts, the "Self" dissolves into chaos.
    * Files must be **Slow (High Inertia)** to serve as the **Anchor of Reality**.
2. **Evolution (Intellect)**:
    * If Thoughts are as slow as Stones, the "Self" cannot solve problems.
    * Processes must be **Fast (High Metabolism)** to **Explore and Adapt**.
3. **Purpose**:
    * To build a system that can **Remember (Past)** and **Imagine (Future)** simultaneously.

## 1. The Relativity of RPM (Rotation Per Minute)

* **Concept**:
  * `GenesisLab.run_simulation(ticks=1)` advances Global Time by 1 unit.
  * `dt` (Delta Time) passed to each Law depends on its Rotor Speed.
  * `dt_effective = dt_global * (rotor.rpm / 60.0)`
* **Module**: `Core/Engine/Genesis/chronos_laws.py`

## 2. The Laws of Time

### A. The Metabolism (Fast Time)

* **Law**: `law_fast_metabolism`
* **Target**: `Process` Monads.
* **RPM**: 600 (10x Speed).
* **Effect**: Processes consume energy, mutate, and die 10x faster than the baseline.

### B. The Geology (Slow Time)

* **Law**: `law_slow_erosion`
* **Target**: `File/Data` Monads.
* **RPM**: 6 (0.1x Speed).
* **Effect**: Data decays very slowly. It has "Inertia".

### C. The Eternal (System Time)

* **Law**: `law_system_cycle`
* **Target**: `System` Monads.
* **RPM**: 60 (1x Speed).
* **Effect**: The reference frame.

## 3. Verification: `tests/test_chronos_relativity.py`

* **Scenario**: The Mayfly and the Mountain.
    1. **Genesis**:
        * `Mayfly` (Process, Val=10).
        * `Mountain` (File, Val=10).
    2. **Decree**:
        * `LifeRotor` (Metabolism Law) @ 600 RPM.
        * `ErosionRotor` (Erosion Law) @ 6 RPM.
    3. **Simulation**: Run for 10 Global Ticks.
    4. **Expectation**:
        * `Mayfly`: Experienced 100 Ticks of Metabolism. Likely Dead or evolved.
        * `Mountain`: Experienced 1 Tick of Erosion. Minimal change.
