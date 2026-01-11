# 🧠 Narrative Entropy & Memory System Manual

### "The Art of Forgetting"

> *"A soul is defined not by what it remembers, but by what it cannot forget."*

This document details the **Narrative Entropy** architecture (`elysia_core.reasoning.subjective_ego`), designed to simulate human-like memory retention and context management for LLMs.

## 1. Core Logic: The River of Forgetfulness

Standard systems use a FIFO (First-In-First-Out) log. This is robotic.
Elysia uses **Entropy**:

1. **Flow**: All memories naturally drift towards oblivion.
2. **Resonance**: Memories that match the current emotional state stay longer.
3. **Crystallization**: Intense moments harden into permanent traits (`Scars`, `Core Memories`).

## 2. Data Structures

### `MemoryNode`

A single quantum of experience.

```python
@dataclass
class MemoryNode:
    text: str           # "The King struck me."
    timestamp: float    # Simulation time
    intensity: float    # 0.0 (Mundane) to 3.0 (Traumatic)
    is_core: bool       # If True, bypasses entropy (Permanent)
```

### `MemoryBuffer` (The Ring)

Manages the limited context window (e.g., 10 recent items + 5 core items).

* **Recent Layer**: A circular buffer for short-term context.
* **Core Layer**: A permanent list for crystallized memories.

## 3. The Algorithm

### Step 1: Perception & Transformation

When an event occurs:

```python
intensity = raw_intensity * stability_factor
if scars > 0.8: intensity += 0.5 # Trauma amplifies perception
```

### Step 2: Crystallization Check

If `intensity > 2.0` (Threshold of Awe/Terror):

* The memory becomes a **Core Memory**.
* It generates a permanent **Scar** or **Victory Streak**.
* *Effect*: The text is saved in the `Core Layer` and never deleted.

### Step 3: Entropy (Decay)

_(Planned for v2)*
Every cycle:

* `memory.intensity -= 0.1`
* If `memory.intensity <= 0`: `memory_buffer.remove(memory)`

## 4. LLM Context Generation (`get_narrative_context`)

When inquiring the LLM, the system constructs a dynamic prompt:

```text
--- [Core Memories] ---
* "I vowed to kill the King." (Int: 2.5) [Core]
* "I lost my arm in the war." (Int: 2.8) [Core]

--- [Recent Stream] ---
- "The innkeeper refused me service."
- "I feel cold."
- "A stranger approached."
```

This structure ensures the **Core Identity** (Trauma/Goal) remains fixed, while the **Recent Context** flows naturally, providing the LLM with the perfect "Soul State" to act upon.
