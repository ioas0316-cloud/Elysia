# 17. Cell Runtime and Reaction Rules

Scope
- Minimal runtime spec to operationalize living cells, concept chemistry, and immune responses in alignment with ELYSIAN_CYTOLOGY and ELYSIAN_IMMUNE_SYSTEM.

Operators ↔ Chemistry
- clarify: disambiguate definitions; lower activation threshold.
- link: create/repair KG edges with confidence and evidence.
- compose: generate `meaning:*` from inputs.
- validate: check supports/refutes, require `experience_*` when concluding.
- retrieve_kg/summarize: non‑creative support ops; optional in reactions.

Energy vs Mass
- Energy score terms: `rule_match + kg_relevance + continuity + value_alignment + evid_conf − latency_cost` (normalized 0..1 recommended).
- Mass updates: per 09_VALUE_MASS_SPEC (λ, α, β, γ; clamp ≥ 0), tracked globally and per‑context.

Organelles (runtime mapping)
- Membrane: gates (quiet/consent, TTL, input quarantine).
- Nucleus: identity/core definition, DNA (instincts + reaction table ids).
- Mitochondria: energy aggregator (flow terms).
- Ribosome/ER: message bus + scheduler execution for nano‑bots.
- Lysosome: refutes/error isolation and cleanup.

Nano‑Bots (roles)
- ValidatorBot: reassess supports/refutes; surface wounds.
- ReconcilerBot: attempt recombination; propose corrected links.
- ApoptosisBot: mark and remove irreconcilable cells safely.

Telemetry Keys (excerpt)
- `bus.message`, `bot.run`, `concept.update` (core)
- `immune.detect`, `immune.recombine`, `immune.apoptosis`, `immune.memory` (immune)

Gates and Consent
- Quiet/Consent respected for any action that changes KG or emits outputs.
- When uncertainty high or value conflict detected, ask‑to‑act before proceeding.

Defaults
- Reactions require ≥1 `experience_*` link for conclusions.
- Operators limited to clarify/link/compose/validate by default for Trial‑001.

