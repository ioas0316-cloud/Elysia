# 99. Idea Backlog (Parking Lot)

Format for each item
- Title: short, imperative
- Why: intent/motivation (1–2 lines)
- Shape: rough design (2–6 bullets)
- Effort: S/M/L (time/complexity)
- Risk: low/med/high (unknowns/impact)
- Next step: one concrete action to try

---

## 1) Korean intent patterns (safe mapping)
- Why: enable natural Korean commands without relying on LLMs
- Shape:
  - Small dictionary-based patterns → nano messages (link/verify/summarize/compose/explain)
  - Normalization: spacing/josa variants, conceptify(token → concept:token)
  - Fallback to English intent/nano if no match
  - Log intent hits to telemetry for tuning
- Effort: M
- Risk: low (scope can be tight), med for coverage creep
- Next step: add `nano_core/intent_gate_ko.py` with 5 base verbs and 10–15 patterns

## 2) Nano slot helper (friendly errors)
- Why: reduce friction when slots are missing (subject/object/text…)
- Shape:
  - Validate nano acts before enqueue; return human hint (e.g., “need subject=…”) 
  - Offer example snippet in response
  - Telemetry tag for invalid acts
- Effort: S
- Risk: low
- Next step: add a `_validate_act(verb, slots)` in bridge and hook into nano path

## 3) KG compaction (synonyms/weak links)
- Why: avoid bloat and keep clarity as the city grows
- Shape:
  - Merge synonyms via small lexicon; collapse isolated/duplicate nodes
  - Prune very weak links (confidence < ε, no evidence)
  - Summarize deltas in Daily Report (added/removed/merged)
- Effort: M
- Risk: med (incorrect merges)
- Next step: `scripts/compact_kg.py` dry‑run with report‑only mode

## 4) Validator enhancements (loops/duplicates)
- Why: reinforce consistency as activity increases
- Shape:
  - Detect simple cycles A↔B in same relation; suppress duplicates
  - Optional review mode for high‑impact relations
- Effort: S–M
- Risk: low
- Next step: extend `ValidatorBot` with small cycle/dup checks and telemetry

## 5) Monitor render parity (starfield/lightweight)
- Why: consistent visuals in monitor and viz
- Shape:
  - Port starfield + voxel/size params to `render_kg`
  - Toggle via function args/env
- Effort: S
- Risk: low
- Next step: replicate helper calls in `render_kg`

## 6) Nano help popover + example chips (i18n)
- Why: make nano approachable; reduce syntax anxiety
- Shape:
  - Small help panel with examples; EN/KR toggle
  - Keep chips; add KR labels (non‑breaking)
- Effort: S
- Risk: low
- Next step: add help div and toggle in chat‑ui; copy examples

## 7) Daily budget cap
- Why: protect low‑spec machines; peace of mind
- Shape:
  - Preferences: max nodes/edges per day; warn on exceed
  - BG daemon honours budget (sleep until next day)
- Effort: M
- Risk: low
- Next step: add counters in `activity_registry` + checks in `background_daemon`

## 8) City planning: namespaces/tags
- Why: readable districts (concept:, role:, value:, myth:, etc.)
- Shape:
  - Tag clusters; choose 3–5 public hubs; visualize with subtle color tint
- Effort: S–M
- Risk: low
- Next step: extend `kg_manager` to persist node `tags:[]`; color map in renderer

## 9) Evidence scoring for explanations
- Why: make “explainer” more meaningful
- Shape:
  - supported_by edges carry a score (0–1); reflect in reporting/visual tint
- Effort: M
- Risk: med (scoring heuristic)
- Next step: extend `ExplainerBot` to accept `score` slot; show in report

## 10) Message queue limits (back‑pressure)
- Why: avoid bursts if many commands are issued
- Shape:
  - Scheduler soft limit; drop lowest strength beyond cap; telemetry warning
- Effort: S
- Risk: low
- Next step: add queue cap param to `Scheduler.step`

## 11) Auto‑viz pulse (optional)
- Why: a touch of “life” without heavy animation
- Shape:
  - UI auto refresh viz every N seconds while a toggle is on
- Effort: S
- Risk: low
- Next step: add small JS interval/toggle in chat‑ui (off by default)

## 12) Quiet onboarding preset
- Why: remove pressure at first contact
- Shape:
  - Start with quiet=true, auto_act=false; invite to enable when ready
- Effort: S
- Risk: low
- Next step: set defaults in `preferences.json` writer paths

