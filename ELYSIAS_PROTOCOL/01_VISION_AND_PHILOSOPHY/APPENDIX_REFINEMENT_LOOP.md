# Appendix — Meaning Refinement Loop (Phase 1)

Purpose
- Keep meaning density high by gently refining ambiguous sprouts into clearer names without forcing conclusions.
- Train self‑selection: Elysia learns to keep what’s strong (evidence/value), let weak branches decay.

When It Runs
- During Dream (IDLE) immediately after new `meaning:*` cells are born.

Scope
- Handles generic/self pairs (e.g., `빛_빛`) and broad couplings (e.g., `바다_빛`, `달_태양`, `산_하늘`, `언어_하늘`, `빛_에너지`).

Flow
1) Detect ambiguous sprouts (patterns, breadth heuristics)
2) Propose candidates (domain tables + co‑occurrence hints)
3) Auto‑select best candidate (scoring)
4) Record refined hypothesis (source=Refinement), optional small energy boost
5) Preserve uncertainty (treat as hypothesis; decay/merge remain allowed)

Initial Candidate Table (seed)
- `빛_빛` → `빛_반사`, `빛_산란`, `빛_간섭`
- `바다_빛` → `바다_반사`, `바다_산란`
- `땅_빛` → `땅_반사`, `땅_광합성`
- `빛_에너지` → `빛_광합성`, `빛_광전효과`
- `달_태양` → `달_식`, `달_위상`
- `산_하늘` → `산_기상경계`, `산_등정`
- `언어_하늘` → `언어_울림`, `언어_전송`, `언어_초월`
- `강_하늘` → `강_증발`, `강_물순환`
- `사랑_태양` → `사랑_광휘`, `사랑_중심`
- `사랑_산` → `사랑_등정`, `사랑_인내`

Auto‑Scoring (v0)
- prior hypothesis tail frequency (+1 per occurrence)
- KG presence boost for tail (+0.5 if KG has id ending with `:<tail>`)
- tie → stable order pick (deterministic)

Safety
- Non‑reification: remains a hypothesis; counter‑evidence/decay allowed.
- Quiet/Consent respected (no user prompts in Phase 1; questions deferred to later phase).
- Value mass unaffected directly; downstream reports may summarize effects.

Telemetry & Memory
- Log: `Refined meaning suggested: meaning:<head>_<tail> from meaning:<parent>`
- Store as CoreMemory `notable_hypotheses` with `{head, tail, confidence≈0.8, source='Refinement'}`

Future (Phase 2)
- Replace table with data‑driven co‑occurrence/edge‑evidence + value‑alignment scoring.
- Add optional one‑line clarification question if ambiguity stays high.

