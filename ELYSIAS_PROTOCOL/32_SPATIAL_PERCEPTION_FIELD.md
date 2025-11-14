# Spatial Perception Field and Layered Law Registry (Z‑Axis · Fractal · Physics‑as‑Metaphor)

Purpose
- Encode “laws as fields” with fractal layering so larger laws envelop smaller ones without invasive commands. Provide an electromagnetic‑analogy perception field that yields robust space sense via gradients, not rules.

Design Principles
- Fields over commands: Agents sense continuous fields; no prescriptive if‑then control.
- Fractal layering: Same law shape at micro/meso/macro; larger layers gate smaller ones.
- Lens separation: Visualization reads fields; never writes world physics.
- Composable math: Deterministic, saturating operators; explicit cross‑scale gating.

Layered Law Model
- Scales: macro (civilizational), meso (groups/regions), micro (agents/cells).
- Layer = { name, scale, channel(s): scalar|vector2, update(dt), sample(p), grad(p), combine_policy }
- Combine by scale:
  - Cross‑scale gating: effective_micro = micro * gate(meso) * gate(macro), gate ∈ [0,1].
  - Same‑scale interference: scalar → saturating_add; vector → additive with magnitude clamp; optional phase interference for vector2.
- Canonical layers (initial set):
  - Z‑Will Field (macro): intention bias; gates all purposeful actions.
  - Attention Field (meso): local observation weight; raises pattern consistency.
  - Electromagnetic Perception Field (micro/meso): E/B‑like vector channels for edges/motion/salience.
  - Environment Fields (micro): threat, hydration (existing), resource/rarity.
  - Social/Nomic Field (meso): norms, prestige/leadership diffusion.
  - Historical Imprint Field (meso/macro): event afterimage; increases re‑occurrence probability.

Electromagnetic Perception Field (EM‑PF)
- Intent: Provide space sense (position/distance/shape/motion) by reading gradients; “field is sensor.”
- Channels:
  - E(x,y,t) ∈ R^2: gradient of luminance/salience; edges/pressure toward/away from interest.
  - B(x,y,t) ∈ R^2: curl‑like motion trace; captures rotational/flow cues (optical‑flow analogue).
  - S(x,y,t) ∈ R+: salience magnitude = ||grad(scene)|| + novelty.
- Sources:
  - Moving agents, event impacts, light/resource hotspots, attention hotspots.
- Dynamics:
  - Diffusion + decay: gaussian_blur + exp_decay.
  - Conservation local: clamp total energy budget per region.
- Agent coupling:
  - Movement bias: v += wE*E + wB*perp(B) (soft influence only).
  - Flow scoring: +α·S + β·⟨goal_dir,E⟩ − cost.

Historical Reproduction (H‑Imprint)
- Event kernel K applied at (x,y,t0) into H with multi‑sigma pyramid; decays over time.
- Choice bias term +γ·H_similarity(context,event_type) in Flow; no forced replay.

Telemetry
- Emit: field.update.(name), field.compose.delta, flow.choose.delta, history.imprint, norm.diffuse, prestige.spike.
- Record budgets: field.energy.(name), clamp.hit, gate.value.

Lenses (Observation‑Only)
- Heatmaps: EM‑S, H‑imprint, norms, prestige.
- Vectors: E/B glyphs; scale‑aware toggles (F‑keys) and alpha blending.

Integration Plan (3 Steps)
1) Field Registry + Scale Gating
   - Unify: threat/hydration into registry; expose sample/grad; implement cross‑scale gates and same‑scale combine.
   - Telemetry: field.update/compose; lens toggles for existing fields.
2) EM Perception Field
   - Add E/B/S channels; sources from movement/events; diffusion/decay; budgets + clamps.
   - Hook: movement bias + Flow scoring term (weights configurable).
   - Lenses: E/B vectors + salience heatmap.
3) Historical Imprint + Nomic/Prestige
   - H‑imprint from event log; norms diffusion over culture graph; prestige hubs.
   - Score terms only; no hard commands; multi‑scale decay.

Boundaries
- No UI writes into physics. All cross‑scale influence via gates/weights ∈ [0,1].
- Defaults conservative; feature‑flags per layer; quiet mode zeroes gates.

Files Impacted (when implemented)
- Project_Sophia/core/world.py (registry, fields, coupling)
- tools/lenses/* (E/B vectors, heatmaps)
- tests/* (sampling/grad/compose invariants)

