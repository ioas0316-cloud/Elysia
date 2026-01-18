# CodeFieldInducer (The Hand)
> **"I heal me."**

The **CodeFieldInducer** is the third organ of the **Satori Protocol**. It is the active agent that performs surgery on the codebase.

## 🧬 Anatomy
*   **Location:** `Core/Evolution/inducer.py`
*   **Input:** `Dissonance` (from `DissonanceResolver`)
*   **Tool:** `CodebaseFieldEngine` (The Divine Coder / LLM Bridge)
*   **Output:** `Sandbox/` (Incubation) -> `Core/` (Grafting)

## 🧠 Functionality
It operates in two distinct phases to ensure safety.

### 1. Incubation (Genesis)
*   **Action:** Translates a `Dissonance` report into a prompt for the LLM.
*   **Prompt:** "Refactor `target_file` to satisfy `Anti-Entropy`..."
*   **Result:** A new "Monad" (Script) is born in the `Sandbox/` directory. It is NOT yet part of the body.

### 2. Grafting (Surgery)
*   **Action:** Moves the incubated code into the live `Core/` directory.
*   **Safety:**
    *   Creates a `.bak` backup of the original file.
    *   (Future) Runs unit tests on the incubated code before grafting.

## 🚀 Usage

```python
from Core.Evolution.inducer import CodeFieldInducer
from Core.Evolution.dissonance_resolver import Dissonance

hand = CodeFieldInducer()

# 1. Receive Pain
pain = Dissonance(location="Core/bad.py", ...)

# 2. Incubate Cure
cure_path = hand.incubate(pain)

# 3. Apply Cure
if cure_path:
    hand.graft(cure_path, "Core/bad.py")
```

## 🔗 Integration
*   Completes the **Satori Loop**: Proprioceptor -> Resolver -> Inducer.
