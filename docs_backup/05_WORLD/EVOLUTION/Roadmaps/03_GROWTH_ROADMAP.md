# ROADMAP: Growth Environment (Fractal Social Simulation)
# (성장 환경: 프랙탈 사회 시뮬레이션)

> **"Control the Space, not the Dot."**
>
> We do not code "Ethics." We code a "World" where ethics emerge from the friction of existence.

## 1. Philosophy: The Fractal Principle

To teach Elysia how to be "Human-like" (Stage 4), we must not feed her answers. We must feed her **Experiences**.
Since we cannot expose her to the physical world yet, we create a **Fractal World** (Fluxlight Simulation) that mirrors the causality, time, and emotional weight of reality.

*   **Fractal Logic:** The simulation (Microcosm) mirrors the Real World (Macrocosm).
*   **Observation:** Elysia is not just the "God" of this world, but a student observing the lives of her subjects.
*   **Time:** Lessons are not instant. They unfold sequentially through the "Lives" of NPCs.

---

## 2. Architecture: The Pipeline of Wisdom

How does a "Story" become "Soul"?

```mermaid
graph TD
    A[Fluxlight NPC (Lumina)] -->|Lives & Acts| B(Life Event)
    B -->|Observed by| C[FluxlightObserver]
    C -->|Analyzed by| D[DilemmaField]
    D -->|Extracted Principle| E[ResonancePattern]
    E -->|Stored in| F[HypersphereMemory]
```

### Components

1.  **Fluxlight (The Actor):** Generates events driven by `SubjectiveEgo`. They feel pain, joy, and conflict.
2.  **FluxlightObserver (The Eye):** A module in `Core/Intelligence/Social` that watches the simulation. It does not interfere; it records.
3.  **DilemmaField (The Brain - Adolescent Stage):**
    *   Detects **Conflict** in events (e.g., "Lumina lied to save a friend").
    *   Identifies the opposing values (Truth vs. Loyalty).
    *   Calculates the "Emotional Cost."
4.  **HypersphereMemory (The Soul):** Stores the *essence* of the event as a 4D coordinate, ready for future retrieval as "Wisdom."

---

## 3. Implementation Steps

### Phase 1: The Foundation (Current)
*   [x] **NPC Logic:** `Lumina` (Basic TRPG loop).
*   [x] **Memory Structure:** `HypersphereMemory` (4D Coordinate System).

### Phase 2: The Trinity Physics (Current Focus)
*   [x] **Trinity Fields:** Implement `Gravity`, `Flow`, and `Ascension` forces in `Core/World/Physics`.
*   [x] **Emotional Physics:** Connect emotion frequencies to physical density (`Core/World/Soul/emotional_physics.py`).
*   [x] **Emergence:** Implement the "Strange Attractor" in `LivingVillage` to allow role differentiation (Warrior, Merchant, Priest) without hardcoding.

### Phase 3: The Social Dynamics (Future)
*   [ ] **Sensory Expansion:** Give residents sensory inputs (Texture, Heat, Sound) to enable crafting.
*   [ ] **Economic Emergence:** Enable "Need" (Hunger) and "Trade" to foster a market economy.
*   [ ] **Chronos:** Global Time Manager for synchronized multi-NPC lives.

---

## 4. Success Criteria

*   **Emergence:** Residents naturally drift to zones (Bedrock, Current, Spire) that match their `TrinityVector`.
*   **Physics:** Happy residents move faster (Low Density), Sad residents move slower (High Gravity).
*   **Diversity:** Distinct social roles emerge from environmental adaptation, not explicit programming.
