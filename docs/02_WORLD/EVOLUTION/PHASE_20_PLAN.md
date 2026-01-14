# Implementation Plan: Phase 20 (The Continuum)

We will implement two new evolutionary paths for Elysia: Memory Management (Space) and File System (Persistence).

## 1. Memory Evolution (Tests of Space)

- **Goal**: Teach Elysia to handle Out-Of-Memory (OOM) situations without crashing.
- **Components**:
  - `law_naive_oom`: A law that strictly limits the number of monads. If `len(monads) > MAX`, it raises an Exception or deletes randomly (Crash).
  - `law_lru_paging`: A law that, when full, moves the Least Recently Used monad to a `disk_monads` list (Swap), keeping the main list clean.
- **Cognitive Cycle Update**:
  - Add metric `OOM_Crashes`.
  - Add paradigm shift: `Naive` -> `Paging`.

## 2. File System Principles (Tests of Time)

- **Goal**: Understand *why* File Systems are complex (Inodes, Blocks, Journaling).
- **Components**:
  - `VirtualFileSystem` (Class): A set of Monads representing `Inodes` and `Blocks`.
  - `law_naive_save`: Writes data linearly. Vulnerable to "Power Failure" mid-write (simulated).
  - `law_journaling_save`: Writes intent to a `Journal` monad first, then commits.
- **Experiment**:
  - Simulate a "Power Cut" (Stop tick mid-operation).
  - Verify data integrity. Naive = Corrupt. Journaling = Recoverable.

## 3. Verification Plan

- [ ] **Run**: `python tests/test_evolution_memory.py`
  - Verify evolution from OOM Crash -> Paging Success.
- [ ] **Run**: `python tests/test_filesystem_principles.py`
  - Verify evolution from Corrupt Data -> Recovered Data via Journaling.
