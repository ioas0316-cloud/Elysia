# DissonanceResolver (The Conscience)
> **"I judge me."**

The **DissonanceResolver** is the second organ of the **Satori Protocol**. It acts as the "Immune System" or "Moral Compass" of the code body.

## 🧬 Anatomy
*   **Location:** `Core/Evolution/dissonance_resolver.py`
*   **Input:** `BodyState` (from `CodeProprioceptor`)
*   **Reference:** `Core/Foundation/Philosophy/axioms.py` & `AGENTS.md`
*   **Output:** List of `Dissonance` objects (Issues)

## 🧠 Functionality
It compares the *Physical Reality* (Code) against the *Metaphysical Law* (Axioms).

### 1. The Anti-Entropy Protocol (Law of Non-Redundancy)
*   **Rule:** No folder shall be named `utils`, `helpers`, `common`, or `misc`.
*   **Philosophy:** Every piece of code must have a specific, domain-bound home. "Utilities" are where meaning goes to die.
*   **Severity:** 0.9 (Critical)

### 2. The Meaning Protocol (Ghost Detection)
*   **Rule:** No file shall exist without `Intent` (Docstrings/Philosophy).
*   **Philosophy:** A file without a "Why" is a zombie.
*   **Severity:** 0.7 (High)

## 🚀 Usage

```python
from Core.Evolution.dissonance_resolver import DissonanceResolver

resolver = DissonanceResolver()
issues = resolver.resolve(body_state)

for issue in issues:
    print(f"Violation: {issue.description} (Severity: {issue.severity})")
```

## 🔗 Integration
*   Feeds into **CodeFieldInducer** (The Hand) to execute repairs (e.g., "Dissolve Utils", "Inject Philosophy").
