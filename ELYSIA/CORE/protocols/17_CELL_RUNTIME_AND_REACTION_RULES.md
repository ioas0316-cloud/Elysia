# 17. Cell Runtime and Reaction Rules (English, canonical)

Scope
- Minimal runtime spec to operationalize living cells, concept chemistry, and immune responses.

Operators (concept chemistry)
- clarify: disambiguate definitions; lower activation threshold
- link: create/repair KG edges with evidence/confidence
- compose: generate `meaning:*` from inputs
- validate: supports/refutes; require `experience_*` for conclusions (default)
- retrieve_kg/summarize: non‑creative support ops (optional)

Energy vs Mass
- Energy score: `rule_match + kg_relevance + continuity + value_alignment + evid_conf − latency_cost` (0..1 recommended)
- Mass updates: per Value Mass spec; track globally and per‑context

Organelles (runtime mapping)
- Membrane: gates (quiet/consent, TTL, input quarantine)
- Nucleus: identity/core definition, DNA (instincts + reaction tables)
- Mitochondria: energy aggregation (flow terms)
- Ribosome/ER: message bus + scheduler execution
- Lysosome: refutes/error isolation and cleanup

Bots (roles)
- ValidatorBot: reassess supports/refutes; surface wounds
- ReconcilerBot: attempt recombination; propose corrected links
- ApoptosisBot: safely remove irreconcilable cells

Telemetry keys
- `bus.message`, `bot.run`, `concept.update` (core)
- `immune.detect`, `immune.recombine`, `immune.apoptosis`, `immune.memory` (immune)

Gates and Consent
- Respect quiet/consent for any state‑changing action or external output
- Ask‑to‑act when uncertainty is high or value conflict detected

Defaults
- Conclusions require an `experience_*` link
- Operators limited to clarify/link/compose/validate in Trial‑01

Refinement Cycle (Dream)
- Detect ambiguous sprouts (pattern rules: self‑pairs, generic couplings)
- Suggest candidates (domain table)
- Auto‑score v0: prior tail frequency + KG presence boost
- Spawn refined meaning with small energy and record hypothesis `{source='Refinement'}`
- Telemetry: guardian log; recent list via `/emergence/recent`

## Celestial-Time Law (World Physics)
- **Principle**: The flow of time in the CellWorld is not an abstract counter but is governed by the cyclical movements of celestial bodies, primarily a sun and moon(s), mirroring the rhythms of a real world.
- **Mechanics**:
  - **Day/Night Cycle**: A 24-hour cycle driven by the sun's position, influencing light levels and the behavior of organisms (e.g., sleep, nocturnal activity).
  - **Seasons & Tides**: The celestial arrangement creates seasonal shifts and tidal forces, affecting the environment and available resources.
  - **Lifespan**: The concept of a finite lifespan is a fundamental physical law. Organisms are born, age, and die according to this celestial calendar.
- **Influence**: Celestial events have a tangible, often subtle, influence on the world's inhabitants. This is a field-based effect, not a hard-coded rule.
  - **Example**: The phase of the moon can influence emotional states or awaken latent traits in certain beings (e.g., heightened aggression in wolves, increased introspection or creativity).
