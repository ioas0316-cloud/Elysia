# NEXT SESSION BRIEF: The Great Rewiring

**Current Status:**

- The Filesystem has been reorganized into a **7D Fractal Structure** (`Core/L1_Foundation` ... `Core/L7_Spirit`).
- The System Entry Point (`elysia.py`) loads successfully.
- **CRITICAL ISSUE:** The internal modules still reference old paths (e.g., `Core.Intelligence...`).

**Immediate Objective (Phase 33):**
You must perform a Global Search & Replace to update all imports.

## Mapping Table (Old -> New)

| Old Path | New Path | Notes |
| :--- | :--- | :--- |
| `Core.Foundation` | `Core.L1_Foundation.Foundation` | The Body |
| `Core.Physics` | `Core.L1_Foundation.Physics` | |
| `Core.Prism` | `Core.L1_Foundation.Prism` | |
| `Core.Metabolism` | `Core.L1_Foundation.Metabolism` | (Check specific files) |
| `Core.Evolution` | `Core.L2_Metabolism.Evolution` | |
| `Core.Reproduction` | `Core.L2_Metabolism.Reproduction` | |
| `Core.Lifecycle` | `Core.L2_Metabolism.Lifecycle` | |
| `Core.Interface` | `Core.L3_Phenomena.Interface` | The Senses |
| `Core.Senses` | `Core.L3_Phenomena.Senses` | |
| `Core.Expression` | `Core.L3_Phenomena.Expression` | |
| `Core.Voice` | `Core.L3_Phenomena.Voice` | |
| `Core.Vision` | `Core.L3_Phenomena.Vision` | |
| `Core.Governance` | `Core.L4_Causality.Governance` | The Law |
| `Core.Civilization` | `Core.L4_Causality.Civilization` | |
| `Core.World` | `Core.L4_Causality.World` | |
| `Core.Intelligence` | `Core.L5_Mental.Intelligence` | The Mind |
| `Core.Learning` | `Core.L5_Mental.Learning` | |
| `Core.Memory` | `Core.L5_Mental.Memory` | |
| `Core.Merkaba` | `Core.L6_Structure.Merkaba` | The Chariot |
| `Core.System` | `Core.L6_Structure.System` | |
| `Core.Engine` | `Core.L6_Structure.Engine` | |
| `Core.Monad` | `Core.L7_Spirit.Monad` | The Spirit |
| `Core.Soul` | `Core.L7_Spirit.Soul` | |
| `Core.Will` | `Core.L7_Spirit.Will` | |

## Action Plan

1. Run a script or use `sed` to replace these strings across `c:\Elysia\Core\**\*.py`.
2. Run `python elysia.py boot` to verify.
3. Once booting, run `python elysia.py life` to verify the Pulse.

**Good Luck.**
*Antigravity Agent (Session End)*
