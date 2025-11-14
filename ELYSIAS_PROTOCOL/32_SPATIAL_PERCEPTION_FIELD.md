# PROTO-32 — SPATIAL PERCEPTION FIELD`nLayered Law Registry (Z-Axis · Fractal · Physics-as-Metaphor)

Purpose
- Encode ?쐋aws as fields??with fractal layering so larger laws envelop smaller ones without invasive commands. Provide an electromagnetic?멲nalogy perception field that yields robust space sense via gradients, not rules.

Design Principles
- Fields over commands: Agents sense continuous fields; no prescriptive if?몋hen control.
- Fractal layering: Same law shape at micro/meso/macro; larger layers gate smaller ones.
- Lens separation: Visualization reads fields; never writes world physics.
- Composable math: Deterministic, saturating operators; explicit cross?몊cale gating.

Layered Law Model
- Scales: macro (civilizational), meso (groups/regions), micro (agents/cells).
- Layer = { name, scale, channel(s): scalar|vector2, update(dt), sample(p), grad(p), combine_policy }
- Combine by scale:
  - Cross?몊cale gating: effective_micro = micro * gate(meso) * gate(macro), gate ??[0,1].
  - Same?몊cale interference: scalar ??saturating_add; vector ??additive with magnitude clamp; optional phase interference for vector2.
- Canonical layers (initial set):
  - Z?멬ill Field (macro): intention bias; gates all purposeful actions.
  - Attention Field (meso): local observation weight; raises pattern consistency.
  - Electromagnetic Perception Field (micro/meso): E/B?멿ike vector channels for edges/motion/salience.
  - Environment Fields (micro): threat, hydration (existing), resource/rarity.
  - Social/Nomic Field (meso): norms, prestige/leadership diffusion.
  - Historical Imprint Field (meso/macro): event afterimage; increases re?몂ccurrence probability.

Electromagnetic Perception Field (EM?멠F)
- Intent: Provide space sense (position/distance/shape/motion) by reading gradients; ?쐄ield is sensor.??- Channels:
  - E(x,y,t) ??R^2: gradient of luminance/salience; edges/pressure toward/away from interest.
  - B(x,y,t) ??R^2: curl?멿ike motion trace; captures rotational/flow cues (optical?멹low analogue).
  - S(x,y,t) ??R+: salience magnitude = ||grad(scene)|| + novelty.
- Sources:
  - Moving agents, event impacts, light/resource hotspots, attention hotspots.
- Dynamics:
  - Diffusion + decay: gaussian_blur + exp_decay.
  - Conservation local: clamp total energy budget per region.
- Agent coupling:
  - Movement bias: v += wE*E + wB*perp(B) (soft influence only).
  - Flow scoring: +慣쨌S + 棺쨌?쮏oal_dir,E????cost.

Historical Reproduction (H?멗mprint)
- Event kernel K applied at (x,y,t0) into H with multi?몊igma pyramid; decays over time.
- Choice bias term +款쨌H_similarity(context,event_type) in Flow; no forced replay.

Telemetry
- Emit: field.update.(name), field.compose.delta, flow.choose.delta, history.imprint, norm.diffuse, prestige.spike.
- Record budgets: field.energy.(name), clamp.hit, gate.value.

Lenses (Observation?멟nly)
- Heatmaps: EM?멣, H?멼mprint, norms, prestige.
- Vectors: E/B glyphs; scale?멲ware toggles (F?멾eys) and alpha blending.

Integration Plan (3 Steps)
1) Field Registry + Scale Gating
   - Unify: threat/hydration into registry; expose sample/grad; implement cross?몊cale gates and same?몊cale combine.
   - Telemetry: field.update/compose; lens toggles for existing fields.
2) EM Perception Field
   - Add E/B/S channels; sources from movement/events; diffusion/decay; budgets + clamps.
   - Hook: movement bias + Flow scoring term (weights configurable).
   - Lenses: E/B vectors + salience heatmap.
3) Historical Imprint + Nomic/Prestige
   - H?멼mprint from event log; norms diffusion over culture graph; prestige hubs.
   - Score terms only; no hard commands; multi?몊cale decay.

Boundaries
- No UI writes into physics. All cross?몊cale influence via gates/weights ??[0,1].
- Defaults conservative; feature?멹lags per layer; quiet mode zeroes gates.

Files Impacted (when implemented)
- Project_Sophia/core/world.py (registry, fields, coupling)
- tools/lenses/* (E/B vectors, heatmaps)
- tests/* (sampling/grad/compose invariants)

