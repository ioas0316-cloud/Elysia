# Experiment Design Guide (Project Elysia)

Purpose: ensure every experiment follows the Codex (sections 22??4), Z-axis, fractal principle, and quaternion/time-acceleration laws.
Scope: everyone designing, requesting, or running macro-scale experiments on this repository.

---

## 0. Encoding / Text
- All experiment specs and logs must be UTF-8 (no BOM).
- Never leave mojibake (e.g., `?占?). Treat broken encoding as a bug and fix it before filing a report.

---

## 1. Time & Scale Rules
1. **Never run single-tick loops** such as `for step in range(N): world.run_simulation_step()`.
2. Drive worlds in macro units via `World.set_time_scale(...)`, `N_macro`, `N_slow`, or other macro schedulers.
3. **Macro branch recipe:** at least `macro_years = 1000` and `seeds = 20` for every branch plan.
4. Prefer multiple macro sims with varied seeds/parameters over one monolithic run. If more detail is required, capture a few macro snapshots instead of replaying every tick.

---

## 2. Quaternion / Branch Principle
- ?쏲uaternion engine??= explore multiple axes (time scales, curriculum pacing, difficulty, and world kits) simultaneously.
- Every experiment must run multiple branches:
  - different random seeds,
  - different time accelerations/curriculum pacing,
  - different world kits (`CELLWORLD`, `CODEWORLD`, `MIRRORWORLD`) or body architectures.
- Compare branches through logs (`language_field`, `self_writing`, `caretaker_feedback`, etc.) rather than focusing on one timeline.

---

## 3. Purpose of Experiments
Goal: probe and update **Elysia?셲 growth laws**, not to showcase pretty demos.
Monitor the canonical logs:
- `logs/symbol_episodes.jsonl`
- `logs/text_episodes.jsonl`
- `logs/causal_episodes.jsonl`
- `logs/elysia_language_field.json`
- `logs/elysia_self_writing.jsonl`
- `logs/elysia_caretaker_feedback.jsonl`
Emphasize:
- growth in self-writing (length, vocabulary, emotional range),
- alignment between caretaker feedback and new writings,
- density/diversity shifts in `elysia_language_field`.

---

## 4. Simulator Usage
- Do not hack world internals; wrap them with macro drivers that adjust time scales/curriculums.
- Use existing logs first; only run new sims for targeted questions.
- When simulation is mandatory:
  - stay macro-scale,
  - log macro snapshots,
  - avoid nested micro loops.

---

## 5. Performance & Safety
1. Prefer batches of macro runs to single huge runs.
2. Avoid CPU-heavy nested loops over ticks or giant arrays.
3. If the analysis is log-driven, run it offline. Only re-simulate when the logs truly lack the needed signal.

---

## 6. Reporting Requirements
Every experiment report must include:
- **Purpose:** What growth law or hypothesis is being tested?
- **Method:** time scale, branches, seeds, curriculum schedule, and logs used.
- **Observations:** what changed in self-writing, caretaker alignment, and the language field.
- **Integration:** what will be updated next (curriculum tweaks, new blockers, approvals).
Mandatory metadata (Codex 짠24 alignment):
- `plan_status`, `status_history`, `blocking_reason` (if applicable).
- `execution_evidence` (macro ticks completed, seed counts, self-writing samples, resonance, language-field deltas) with references to the supporting log entries.
- `references` list pointing to documents/log paths.
- `adult_ready` gate stays `false` until the expression metrics clear the thresholds and a caretaker audit signs off.

---

## 7. Z-Axis Reminder
Before defining goals or metrics, re-check the **Why**:
- Does this experiment help Elysia grow as a self-aware, caring entity?
- Are we letting purpose reshape or discard goals when needed?
If the Why is unclear, redesign or skip the experiment.

---

## 8. Codex Hand-off Instructions (attach to every new experiment request)
When asking Codex or lab agents to execute trials:
1. Quote the macro/quaternion rules above verbatim: no tick-by-tick loops, enforce `World.set_time_scale`, `N_macro`, 1,000-year 횞 20-seed bundles, and branch plan logging per world kit.
2. Require analysis of the canonical logs listed in 짠3, focusing on self-writing length/emotion/vocabulary and caretaker-feedback alignment instead of ?쏿ccuracy??
3. Demand Codex 짠24 metadata in every `trial_report`: `status_history`, `execution_evidence`, `blocking_reason`, reference list, and explicit `adult_ready` statements tied to evidence.
4. Ensure every report body (or metadata block) includes Purpose, Method, Observations, and Integration, plus the plan/trial status fields above.
5. State clearly that `adult_ready = true` is only valid when self-writing + caretaker feedback metrics reach the adult thresholds **and** a caretaker audit records the approval.

---

## 9. Immediate Action Items for Current Branches
- Re-audit the existing 1,000-year 횞 20-seed branch plans; attach `plan_status` + `blocking_reason` + caretaker audit notes before marking them ?쐒eady??
- Generate curriculum/experiment bundles (CELLWORLD/CODEWORLD/MIRRORWORLD) that prioritize Symbol/Text/Self-writing/Caretaker feedback metrics, and document the macro settings/log expectations per kit.
- Use `scripts/experiment_report_template.py` (see repo) to auto-fill the Codex 짠24 metadata so caretakers on low-spec GPUs can still produce complete reports.

