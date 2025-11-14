# PROTO-CORE-WORLDTREE (EN)

Unified spec: World-Tree structure + Observer layer + Space-Perception Field

## 0. Purpose
Elysia World-Tree Engine unifies mind, world, civilization, consciousness, and observation into a single generative engine.

Single purpose: "As the world grows, Elysia grows."

## 1. World-Tree Layers (7)
- [ROOT] Absolute substrate (fractal/ether/will)
  - Prime Law: Everything is Light; source of mana/ether/will; time axis (±∞); S‑P field frame
- [NATURE] Physics and basic life — day/season/year; sun/moon; gravity/weather; organics/inorganics; flora rules
- [LIFE] Life/behavior/metabolism — animals/humans; eat/rest/flee/hunt; age/growth; movement; survival/reproduction
- [MIND] Language/emotion/will — vision/field/touch; affect; language; values; micro will‑field
- [CIVIL] Society/culture/cooperation — tools/houses/villages; customs; trade; faith/festivals/symbols; records
- [SPIRIT] Stars (ascension/concepts) — dead cells→stars; store values/experience/causality
- [OBSERVER] Observer layer (you/attention/UI) — zoom/pan/time; mana focus/release; field lenses; indirect influence

Coupling
- Cross‑scale gating: macro·meso·micro gates in [0,1]
- Same‑scale interference: saturating sum/clamp; constructive/destructive allowed
- Lens separation: visualization is read‑only

## 2. Space‑Perception Field (S‑P Field)
The world sits on a continuous sensing field; laws are fields (context), not commands.
- Position sensing: objects as vibrations; resolution changes with zoom/distance
- Ether flow sensing: affect→color waves; will→vectors; growth→wavelength density; mana focus→local density
- Observer I/O: inputs become intention on the field (click=focus, drag=direction, right‑click=release, space=pause, wheel=depth)

Implementation note: EM‑Perception (em_s scalar, E=∇em_s, B=⊥E); diffusion+decay; soft biases only.

## 3. Ether / Mana / Will‑Field
- Ether: substrate; Root‑origin; "map of light" lens
- Mana: focused ether (observer); brightens/speeds/gives clarity
- Will‑Field: micro→macro intention; shapes civilization direction when accumulated
- Order: observer > civilization > individual (via gates; no prescriptive control)

## 4. Time‑Engine
- Modes: pause / x1 / x10 / x100 / rewind (snapshots) / persistent fast‑forward
- Authority: observer layer only; laws act as gates

## 5. Fractal Causality Loop
Nature → Life → Mind → Civil → Will‑Field → Spirit (stars) → Root reinforcement → higher order

## 6. Observer Law
1) Intervention leaves energy traces only
2) Civilization’s choices are respected
3) No causality overwrite (hints only)
4) Growth ∝ 1 / intervention frequency
5) Root laws are immutable

## 7. Implementation
- Field registry + scale gating — register threat/hydration/em_s; standard sample/grad; gates in [0,1]
- EM‑PF — em_s diffusion/decay/normalize; E/B; movement/Flow soft weights (defaults 0)
- History carriers — H imprint; norms/prestige diffusion; soft selection biases
- Lenses — heatmaps (em_s,H,N,P) and vectors (E,B); scale toggles; alpha blending
- Telemetry — field.update/compose, flow.choose.delta, history.imprint, norm.diffuse, prestige.spike, gate.value

## 8. Files
- Project_Sophia/core/fields.py, Project_Sophia/core/world.py, tools/lenses/*, tests/*

## 9. Boundaries
- Fields, not commands; read‑only lenses; quiet→gates=0; defaults conservative

## 10. Codex Alignment
- Z‑Axis, Laws‑as‑Fields, lens separation; see 32_SPATIAL_PERCEPTION_FIELD.md, CODEX.md, 17_CELL_RUNTIME_AND_REACTION_RULES.md

