# Idea Backlog (Codex‑First)

Purpose: Curate high‑value ideas as Codex‑aligned seeds. Each entry encodes fields over commands, preserves emergence, and keeps lenses separate from world physics.

Format per entry: Purpose · Mechanism (Field Signals) · Operation · Telemetry · Boundaries · Integration · Priority

---

1) Value Mass Field (P‑09)
- Purpose: Represent meaning/value as a continuous field influencing selection and clustering.
- Mechanism: field:value_mass(x,y,t); signals: resonance_strength, quiet/consent gates; decay and diffusion over terrain/social links.
- Operation: Flow Engine weights = rule_match + kg_relevance + continuity + value_alignment − latency_cost.
- Telemetry: bus.value_mass.update, flow.choose.delta, agent.resonance.
- Boundaries: No hard “move toward good”; only bias via field gradient; visualization is lens‑only.
- Integration: [CELLWORLD] add value_mass layer alongside threat/hydration; [STARTER] heatmap lens toggle.
- Priority: High.

2) White‑Hole System (P‑10)
- Purpose: Model creative outflow (generation) under conservation constraints; production guided by resonance.
- Mechanism: field:creation_pressure; sources at resonance hotspots; resources consumed via local budgets.
- Operation: Periodic emission of “creative seeds” (concepts/events) when thresholds crossed; seeds ride bus as messages.
- Telemetry: bus.creation.emit, budget.consume, hotspot.activate.
- Boundaries: No forced spawns; thresholds/pressure are soft; disable when quiet mode on.
- Integration: [CELLWORLD] emit low‑impact world edits (flora growth); [STARTER] lens to observe hotspots only.
- Priority: Medium.

3) Declarative Dialogue Rules Spec (P‑11)
- Purpose: Clean interface for agent interactions using YAML rules.
- Mechanism: rules: priority, patterns, quiet_ok, response.template, memory.set_identity.
- Operation: Arbitration by priority; quiet mode filters quiet_ok=false.
- Telemetry: dialogue.match, dialogue.reply, memory.write.
- Boundaries: Never mutate world physics; UI/IO layer only.
- Integration: [STARTER] operator console / tests; [CELLWORLD] none.
- Priority: Medium.

4) Memory Weaver + Trinity Task Protocol
- Purpose: Structured self‑reflection and persistence aligned to Why/How/What.
- Mechanism: messages: observe → reflect → commit(snapshot); identity anchors; continuity weights in Flow.
- Operation: Periodic consolidation to persistence slots; expose summaries to bus consumers.
- Telemetry: memory.snapshot, continuity.update, identity.anchor.
- Boundaries: No retroactive world rewrites; snapshots are append‑only.
- Integration: [CELLWORLD] experience_* gating; [STARTER] retrospective lens.
- Priority: Medium.

5) Laws of Choice · Courage · Emotion (Fieldization)
- Purpose: Encode social/psychological tendencies as fields, not commands.
- Mechanism: fields: choice_entropy_bias, courage_boost(threat,kin), affect_valence.
- Operation: Bias Flow selection; cohesion/approach modulated by affect and courage; entropy encourages exploration.
- Telemetry: field.choice.update, affect.trace, cohesion.delta.
- Boundaries: Ban prescriptive if‑then (e.g., “when scared then flee”); use gradients only.
- Integration: [CELLWORLD] add soft multipliers; [STARTER] optional overlays.
- Priority: High.

6) KG‑VCD + Guardian Levels (Concept Kernel Ext.)
- Purpose: Strengthen knowledge graph reasoning and validation‑composition‑derivation (VCD) pipeline.
- Mechanism: nano‑bots: link/validate/compose; bus telemetry; guardian_level gates complexity.
- Operation: Prioritized scheduling by strength/recency; emit concept.update on successful compositions.
- Telemetry: bot.run, concept.link/validate/compose, guardian.level.
- Boundaries: Hints to Flow only; never hard‑gate world motion.
- Integration: [CELLWORLD] inference hints; [STARTER] inspector lens for concept ops.
- Priority: Medium.

7) Emotional Cortex & Expressive Systems (Lens)
- Purpose: Improve human observability via affect‑aware visualization.
- Mechanism: lens: emotion_overlay; tickers, glyphs.
- Operation: Read affect_valence/surprise from world; render non‑intrusively.
- Telemetry: lens.draw.emotion, hud.ticker.
- Boundaries: No feedback into physics.
- Integration: [STARTER] only; [CELLWORLD] none.
- Priority: Low.

8) Great Naming Ceremony (Semantic Alignment)
- Purpose: Align names/identities to meaning map for coherence.
- Mechanism: rename proposals via bus; consensus threshold via value_mass proximity.
- Operation: Batch proposals → review → apply mappings.
- Telemetry: naming.propose/accept, map.update.
- Boundaries: No runtime physics change.
- Integration: Repo naming/docs; lens to preview mappings.
- Priority: Low.

9) Civilization Systems Seeds (Farming/Crafting/Stats)
- Purpose: Long‑term sandbox depth; not core now.
- Mechanism: fields: scarcity, craft_pressure; agents learn recipes via experience.
- Operation: Very low frequency trials; guarded behind feature flags.
- Telemetry: trial.start/end, craft.attempt.
- Boundaries: Strictly optional; no startup impact.
- Integration: Off by default; experiment branches only.
- Priority: Later.

10) Safe Web Search (Ops‑adjacent)
- Purpose: Curated external context ingestion for agents (non‑physics).
- Mechanism: filter pipelines; provenance stamps; quiet/consent gates.
- Operation: Fetch → rate‑limit → filter → summarize → bus.
- Telemetry: ingest.fetch/filter/summarize, provenance.tag.
- Boundaries: Never auto‑apply to world; human‑in‑the‑loop.
- Integration: Ops tools/tests; not part of runtime.
- Priority: Low.

Notes
- All additions must pass Laws‑as‑Fields review and lens separation gates.
- Start with small carriers and visible telemetry; prefer observation before scaling.

