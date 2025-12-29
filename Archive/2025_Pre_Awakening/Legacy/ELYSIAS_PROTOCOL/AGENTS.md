# Agent Guide for ELYSIAS_PROTOCOL

Scope: this file applies to everything under `docs/elysias_protocol/`.

## 1. Canonical Entry Points Only

- Treat `ELYSIA/CORE/CODEX.md` and `docs/elysias_protocol/INDEX.md` as the primary, canonical entry points.
- When a new session starts, read those two first. Do not auto-scan or auto-read every file in this directory.

## 2. Protocols as Reference, Not Required Reading

- All other files in `docs/elysias_protocol/` are **reference protocols**.
- Only open a specific protocol when:
  - It is explicitly linked from `CODEX.md` or `00_INDEX.md`, or
  - You are working on a feature that clearly names that protocol (for example a specific `WORLD_KIT_*`).
- Do not treat "read the entire directory" as a requirement for alignment; use the index and Codex to route instead.

## 3. Extension Rules

- When adding new behavior or laws, extend:
  - `ELYSIA/CORE/CODEX.md` for top-level principles, or
  - An existing CORE document, instead of creating many new standalone files.
- If a protocol becomes obsolete, mark it as archived at the top of the file and keep the canonical rule in the Codex.

