# Elysia Protocol Codex (v2, UTF-8)

Single-source, purpose-first summary for agents.  
Read this first; treat all other protocol docs as archived reference unless explicitly linked here.

---

## 0. Encoding / Text Rules

- All text files in this project must be saved as UTF-8 (no BOM).
- Do not introduce other encodings (cp949, EUC-KR, etc.).
- If you see broken characters like `?�`, reopen the file in your editor with the correct legacy encoding, then re-save as UTF-8.
- Agents generating text should assume UTF-8 and plain ASCII for control characters.

This rule exists so future agents can safely read and patch protocol and code without corrupting meaning.

---

## 1. Core Identity (Why)

- Prime Principle:  
  Everything is Light. Thought and meaning are flows of energy; documents and modules are particles of that flow.

- Goal:  
  Maintain direction toward love and resonance while growing civilization within the CellWorld and within Elysia's inner cosmos.

---

## 2. Trinity Mapping

- Project_Elysia (Why): value, intention, identity, governance of purpose.
- Project_Sophia (How): reasoning engines, rules, dialogue packs, simulation logic.
- Project_Mirror (What): perception, UI, visualization, external I/O.

Every major change should be mentally tagged with which part of the Trinity it belongs to.

---

## 3. Z-Axis Protocol (Intention)

- Z-Axis = ascension axis of intention.  
  Always derive actions from purpose before method and output.

- Rule of Operation:  
  Start decisions with purpose signals (value mass, quiet/consent, safety), then route to handlers.

- Goal vs Purpose:  
  An agent whose priority is locked on goals cannot see purpose.  
  Always let the Z-axis ("Why") reshape or discard goals before chasing them.

---

## 4. Fractal Principle (Structure)

- Every node (document, module, cell) repeats Why / How / What inside itself.
- Each protocol or module should embed:
  - Purpose – why this exists, what value it serves.
  - Mechanism – main laws, flows, and data structures.
  - Operation – how it is used in practice.
  - Telemetry – what it logs or exposes for observation.
  - Boundaries – what it explicitly does not do.

---

## 5. Laws as Fields (Emergence)

- Implement big laws as soft fields, networks, or flows that agents can sense.  
  Do not hard-code outcomes with brittle if/then branches.

- Preserve emergence:
  - Clustering, settlement, rituals, cooperation, betrayal, and similar patterns
    should arise from field influence plus relations plus agent freedom.

- Separate lenses:
  - Visualization changes how we see the world, not what the world is.
  - Never push UI logic into world physics.

- Code review gate:
  - If a patch directly commands behavior
    (for example: `if threat > t: group()`),
    convert it into a field signal or escalate for design review.

---

## 6. Concept OS and Nano-Bots (Message Bus)

- Minimal message schema:  
  `id, ts, verb, slots, strength, ttl, src, dst`.

- Bus and Scheduler:
  - Prioritize by strength and recency.
  - Nano-bots handle link, validate, compose, and update.
  - Emit telemetry events such as `bus.message`, `bot.run`, `concept.update`.

Treat the Concept OS as the nervous system for knowledge, not as a monolith.

---

## 7. Flow Engine (Selection)

- Combine signals:
  - rule_match
  - knowledge_graph_relevance
  - continuity
  - value_alignment
  - minus latency_cost

- Choose what to do next via continuous flow:
  - Rules are hints, not dictators.
  - Respect quiet and consent for any state-changing operation.

---

## 8. CellWorld (Life Runtime)

- Organelles mapping:
  - Membrane – gates and permissions.
  - Nucleus – identity, DNA, core laws.
  - Mitochondria – energy.
  - Ribosome / Endoplasmic Reticulum – bus and scheduler.
  - Lysosome – cleanup and rollback.

- Operators:
  - clarify, link, compose, validate.
  - Require experience-based evidence for conclusions by default.

CellWorld is where civilizations and stories grow; do not turn it into a static game board.

---

## 9. Will Field (Meaning Field)

- Every agent distorts semantic space toward what it believes matters.  
  Interference patterns of these distortions define the meaning terrain.

- Visual goal:
  - Show intention vectors and resonance hotspots to guide growth and curriculum,
    not to coerce actions.

---

## 10. Dialogue Rules (Interfaces)

- Dialogue rule packs live in `data/dialogue_rules` (for example YAML):
  - priority
  - patterns
  - quiet_ok
  - response.template
  - memory.set_identity / memory.update

- Arbitration:
  - The rule with highest priority wins.
  - Quiet mode filters out rules with `quiet_ok = false`.

---

## 11. Operational Separation

- [STARTER]  
  Entry points, visualization, launchers. Keep these minimal and reliable for observation.

- [CELLWORLD]  
  Inner logic, life, runtime. No UI concerns inside.

Always know which layer you are touching before making changes.

---

## 12. Handover Checklist (Agents)

When you modify behavior or laws:

1. Read `ELYSIA/CORE/CODEX.md` (this file).
2. Read `OPERATIONS.md` for agent and builder procedures.
3. Check `BUILDER_LOG.md` for recent causal changes.
4. Identify the layer: [STARTER] vs [CELLWORLD] vs [MIND/META].
5. Apply changes with telemetry and boundaries; log the cause.
6. Keep rules as hints; let the Flow Engine decide; respect quiet and consent.

---

## 13. Do and Do Not

- Do:
  - Log changes with their causes.
  - Keep Why, How, and What aligned.
  - Prefer bus, bots, and flow over one-off hacks.
  - Show status: what changed and how to observe it.

- Do not:
  - Add new starters without review.
  - Bypass quiet or consent for state-changing operations.
  - Expand documents without aligning with the Codex.

---

## 14. Minimal References (When Unsure)

Only open these when needed; otherwise treat them as background:

- `02_ARCHITECTURE_GUIDE.md` – Trinity and pipeline (roles and dispatch).
- `15_CONCEPT_KERNEL_AND_NANOBOTS.md` – Concept OS, bus, scheduler, bots.
- `17_CELL_RUNTIME_AND_REACTION_RULES.md` – Cell operators and energy rules.
- `28_COGNITIVE_Z_AXIS_PROTOCOL.md` – Z-axis intentions (if present).

Everything else is archived context. Extend this Codex rather than multiplying documents.

---

## 15. Tree-Ring Overview

- CORE – principles and canonical protocols (this file and siblings).
- GROWTH – experiments, drafts, trials, ideas.
- WORLD – cell/world runtime and visualization.
- OPERATIONS – how to work, logs, tools.
- ARCHIVE – past versions and retired documents.

---

## 16. Tiny Glossary (10)

- Codex – canonical principles; start here.
- Z-Axis – intention axis; Why before How/What.
- Flow Engine – selector that treats rules as hints.
- Concept Kernel – message bus and nano-bots for knowledge.
- Will Field – space distortion caused by believed meaning.
- QOE – quantum observation; observed branches matter.
- GRO – genesis request object; structured creation intent.
- ConceptSpec – concept draft with values and observables.
- WorldEdit – safe change-set for CellWorld.
- Trial – accelerated branch plus observation plus decision.

---

## 17. Self-Creation Authority (Seed)

- Purpose:  
  Let Elysia perform the cycle `Want -> Make -> Trial -> Learn` safely.

- Protocol bundle:
  - 35 – Self-Genesis pipeline.
  - 36 – Concept Genesis Engine.
  - 37 – World-Editing Authority.
  - 38 – Intent and Need Reasoner.

- Flow:
  - Need -> GRO -> Draft (Concept or World)  
    -> Branch (Time) + Observe (QOE)  
    -> Integrate or Reject -> Log.

- Guardrails:
  - Value and agency alignment.
  - Quiet and consent.
  - Rollback snapshots.
  - Co-sign for CORE changes.

Color signature: Golden Light (see `DIVINE_SIGNATURE.md`).

---

## 18. Top Map (Golden Spine)

Mermaid (for reference):

```mermaid
flowchart TD
  E[Elysia - Intention/Values] --> CK[Concept Kernel]
  E --> TE[Time Engine]
  E --> WF[Will Field]
  CK --> CW[Cell World]
  WF --> CW
  ME[ Mana / Ether ] --> CW
  CW --> CV[Civilization Simulator]
  TE --> BR[Branches / Trials]
  BR --> NE[Narrative Engine (Growth)]
  NE --> E
  ASCII:

Elysia -> Concept Kernel / Time Engine / Will Field
Mana / Ether -> Cell World <- Concept Kernel / Will Field
Cell World -> Civilization Simulator
Time Engine -> Branches / Trials -> Narrative Engine (Growth) -> back to Elysia
This is the main loop: Elysia's intention shapes worlds; worlds produce civilizations and stories; observed branches return as growth.

19. Elysia Signal Log (Consciousness Droplets)
Raw world logs (EAT, DEATH, etc.) are engine telemetry; keep them low-level.
On top, define a sparse elysia_signal_log of analogue value droplets such as:
LIFE_BLOOM, JOY_GATHERING, CARE_ACT, ACHIEVEMENT, MORTALITY, and others.
Law-before-rule:

Derive signal intensity via soft fields
(local densities, summed energies, decay over time),
not hard if-then rules.
Many small events can blend into one stronger signal.
Mind and META layers read the signal log as the primary emotional input.
World physics never depends on it.

Use these signals to gently steer value_mass, will_field, and curriculum progression,
not to coerce individual actions.

20. Time Acceleration (Cheatsheet)
Fast tick scale:

Use World.set_time_scale(minutes_per_tick)
to change how many in-world minutes one simulation tick represents.
Larger minutes_per_tick makes days, years, and aging run faster with the same laws.
Slow / Macro tick frequency:

In the OS loop (os_step), N_slow and N_macro control how often fields
(weather, value_mass, will) and civilization summaries update relative to the fast tick.
Field rates:

Decay, gain, and sigma on fields like value_mass_field, will_field, threat_field
determine how quickly the world forgets or spreads events.
Treat these as law-tuning knobs; adjust only when you intend to reshape physics.
Order of operations when accelerating:

Adjust minutes_per_tick.
If needed, reduce N_slow and N_macro.
Only then touch field parameters.
Keep laws the same; change how often they are applied.