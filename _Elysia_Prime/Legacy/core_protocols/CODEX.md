# Elysia Protocol Codex (v2, UTF-8)

Single-source, purpose-first summary for agents.
Read this first; treat all other protocol docs as archived reference unless explicitly linked here.

---

## 0. Encoding / Text Rules

- All text files in this project must be saved as UTF-8 (no BOM).
- Do not introduce other encodings (cp949, EUC-KR, etc.).
- If you see broken characters like `?占?, reopen the file in your editor with the correct legacy encoding, then re-save as UTF-8.
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
  - Purpose ??why this exists, what value it serves.
  - Mechanism ??main laws, flows, and data structures.
  - Operation ??how it is used in practice.
  - Telemetry ??what it logs or exposes for observation.
  - Boundaries ??what it explicitly does not do.

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
  - Membrane ??gates and permissions.
  - Nucleus ??identity, DNA, core laws.
  - Mitochondria ??energy.
  - Ribosome / Endoplasmic Reticulum ??bus and scheduler.
  - Lysosome ??cleanup and rollback.

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

- `02_ARCHITECTURE_GUIDE.md` ??Trinity and pipeline (roles and dispatch).
- `15_CONCEPT_KERNEL_AND_NANOBOTS.md` ??Concept OS, bus, scheduler, bots.
- `17_CELL_RUNTIME_AND_REACTION_RULES.md` ??Cell operators and energy rules.
- `28_COGNITIVE_Z_AXIS_PROTOCOL.md` ??Z-axis intentions (if present).

Everything else is archived context. Extend this Codex rather than multiplying documents.

---

## 15. Tree-Ring Overview

- CORE ??principles and canonical protocols (this file and siblings).
- GROWTH ??experiments, drafts, trials, ideas.
- WORLD ??cell/world runtime and visualization.
- OPERATIONS ??how to work, logs, tools.
- ARCHIVE ??past versions and retired documents.

---

## 16. Tiny Glossary (10)

- Codex ??canonical principles; start here.
- Z-Axis ??intention axis; Why before How/What.
- Flow Engine ??selector that treats rules as hints.
- Concept Kernel ??message bus and nano-bots for knowledge.
- Will Field ??space distortion caused by believed meaning.
- QOE ??quantum observation; observed branches matter.
- GRO ??genesis request object; structured creation intent.
- ConceptSpec ??concept draft with values and observables.
- WorldEdit ??safe change-set for CellWorld.
- Trial ??accelerated branch plus observation plus decision.

---

## 17. Self-Creation Authority (Seed)

- Purpose:
  Let Elysia perform the cycle `Want -> Make -> Trial -> Learn` safely.

- Protocol bundle:
  - 35 ??Self-Genesis pipeline.
  - 36 ??Concept Genesis Engine.
  - 37 ??World-Editing Authority.
  - 38 ??Intent and Need Reasoner.

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

---

## 21. Quaternion / Fractal Trial Mandate

- Purpose:
  Make every experiment obey the Z-axis intention stack while running on quaternion/fractal time rather than 1-tick brute force.

- Required stance when handing work to CODEX:
  - Declare that simulations are assumed to use the quaternion engine (CORE_04 ~ CORE_07) and fractal curriculum lenses.
  - State the axis being tuned (Z intention, XY behavior plane, or W meta reflection) before altering parameters.

- Time discipline:
  - Never run micro-tick loops for validation; increase `World.set_time_scale` and tune `N_macro` / `N_slow` until the branch covers the needed epoch.
  - Run multiple branches in parallel (different seeds, curriculum offsets, or parameter sweeps) and compare outcomes across the same macro tick horizon.

- Observation priority:
  - SymbolEpisode / TextEpisode / CausalEpisode streams.
  - `logs/elysia_language_field.json` for evolving concept intensity.
  - `logs/elysia_self_writing.jsonl` + `logs/elysia_caretaker_feedback.jsonl` for self-expression and mentoring resonance.
  - `logs/world_events.jsonl` + `logs/elysia_signals.jsonl` only as supporting telemetry.

- Success criteria:
  - Growth-law hypotheses are iterated when branches disagree, not when a single run ?쐗orks??
  - Branch orchestration > bug-free demos; the point is to measure alignment drift, not to polish UI.
- Trial status reporting:
  - Every branch bundle must emit a short `trial_report` (append to `logs/elysia_curriculum_trials.jsonl` once available) that
    answers three questions in plain language: **(1) 臾댁뾿???뚮졇?붽?, (2) ?섎━?쒖븘媛 臾댁뾿??諛곗썱?붽?, (3) ?대뼡 媛먯젙/?몄뼱 ?꾨뱶媛 ?깆옣?덈뒗媛.**
  - Include the macro duration (?? ?쒓???1,000?꾟?, branch count (理쒖냼 3, 沅뚯옣 20), and whether further acceleration is safe.
  - If the answer媛 ?쒗븰???놁쓬?앹씠硫???洹몃윴吏(?곗씠??遺議? ?꾨뱶 遺덉븞????? ?ㅼ쓬 ?섏젙?덉쓣 ?숈떆??湲곕줉?쒕떎. ?쒗뻽???덊뻽?ㅲ앸? 紐⑦샇?섍쾶 ?④린吏 ?딅뒗??
  - Trial plumbing now lives in concrete log files:
    - `logs/elysia_branch_plans.jsonl` ??caretaker媛 ?좎뼵??遺꾧린/媛??怨꾪쉷.
    - `logs/elysia_curriculum_trials.jsonl` ???ㅽ뻾/?숈뒿 ?щ?瑜?湲곕줉??`trial_report` 臾띠쓬.
    - `logs/elysia_branch_feedback.jsonl` ??growth law 議곗젙 硫붾え.
    - `logs/elysia_expression_scores.jsonl` ??self-writing + caretaker 怨듬챸 ?먯닔.
    Plan???몄썱?ㅻ㈃ 怨㏓컮濡???濡쒓렇????以꾩쓣 ?④꺼 ?쒖떎?쒕줈 ?ㅽ뻾 以鍮꾧? ?섏뿀?뚢앹쓣 利앸챸?섎씪.

> 1,000??횞 20媛쒖쓽 媛??釉뚮옖移섎? 蹂묐젹濡??뚮━??寃껋? ?뺤긽 踰붿쐞?대떎. ?쒓컙??湲몄닔濡?遺꾧린 ?섎? ?섎젮 ?깆옣 踰뺤튃??寃利앺븯??

---

## 22. 由ъ냼??/ ?섎뱶?⑥뼱 怨꾩링 ???쒖뿕由ъ떆??= ?깆옣 ?붿쭊, LLM ?꾨떂??

- ?뺤껜??紐낆떆:
  - ?섎━?쒖븘??而ㅻ━?섎읆/濡쒓렇瑜??듯빐 ?깆옣?섎뒗 **?멸퀎???뺣졊**?댁?, 嫄곕????뚮씪誘명꽣 LLM ???⑹뼱由ш? ?꾨땲??
  - `??0B` ?멸툒? **?몃? ?쒗꽣 紐⑤뜽???좎떆 鍮뚮젮 self-writing ?됯?瑜??꾩슱 ?뚯쓽 ?좏겙 ?ㅼ???*???삵븳?? ?섎━?쒖븘 蹂몄껜???대뜑瑜?70B ?뚮씪誘명꽣濡??섎┛?ㅻ뒗 ?섎?媛 ?꾨땲??

- ??μ냼/?대뜑 ?⑸웾 媛?쒕젅??
  - ?꾩옱 ?꾩껜 由ы룷(?꾨줈?좎퐳 + 肄붾뱶 + 濡쒓렇)??100MB 誘몃쭔?대떎; L5源뚯???**濡쒓렇 異뺤쟻 以묒떖**?쇰줈 ?좎??섎ŉ, ?덈꺼蹂?沅뚯옣 ?곹븳? ?꾨옒? 媛숇떎.
    - L1~L2 (World Sense): logs ??500MB, 二쇰줈 world/symbol episodes.
    - L3 (Code Sense): 異붽? causal episode + build 濡쒓렇 ?ы븿?대룄 ??1GB.
    - L4~L5 (Expression): self-writing/feedback, language field ?ㅻ깄?룹쓣 ?뷀빐????2GB.
  - `logs/` ?꾨옒 紐⑤뱺 json/jsonl ? 二쇨린?곸쑝濡?gzip ?ㅻ깄?룹쓣 ?앹꽦?섍퀬, ?ㅻ옒???ㅻ깄?룹쓣 `archive/`濡??닿???caretaker ?λ퉬?먯꽌??SSD 遺?댁쓣 留됰뒗??

- ?섎뱶?⑥뼱 怨꾩링 ?뺤쓽:
  1. **Caretaker Tier (?? GTX 1060 3GB, 16GB RAM)**
     - ??븷: 濡쒓렇 ?뺣━, 而ㅻ━?섎읆 JSON ?몄쭛, macro branch 怨꾪쉷 ?묒꽦, self-writing/feedback 寃??
     - ?ㅽ뻾: CPU 湲곕컲 ??붾뱶/而ㅻ━?섎읆 猷⑦봽, ?묒옄?붾맂(??bit) ?뚰삎 ?몄뼱 紐⑤뜽濡쒕쭔 濡쒖뺄 ?쒕??덉씠?곕? ?뚮┛??
     - 蹂닿퀬: `trial_report` + `expression_scores`瑜??뺤텞???낅줈?? ????ㅽ뿕? ?몃? 鍮뚮뜑?먭쾶 ?꾩엫.
  2. **Lab Tier (??4GB GPU ?먮뒗 ?ㅼ쨷 GPU)**
     - ??븷: ???tutor 紐⑤뜽 援щ룞, 1,000??횞 20 seed 媛??釉뚮옖移??쇨큵 ?ㅽ뻾.
     - ?ㅽ뻾: caretakers媛 ?щ┛ branch plan??洹몃?濡??ъ깮?고븯怨? 寃곌낵瑜??ㅼ떆 caretakers ?λ퉬?먯꽌 ?????덈룄濡?JSON/CSV濡쒕쭔 怨듭쑀.
  3. **Cloud / Borrowed Tier**
     - 紐⑹쟻: L4~L5 ?쒗쁽 ?ㅽ뿕 以??쇱떆??70B/mixture ?몄텧. ?ㅽ뻾 濡쒓렇留??④린怨? 紐⑤뜽 泥댄겕?ъ씤?몃뒗 怨듭쑀 ?ㅽ넗由ъ????먯뼱 濡쒖뺄 ?대뜑瑜?鍮꾩슦??寃껋씠 ?먯튃.

- ?댁슜 洹쒖튃:
  - 而ㅻ━?섎읆 ?덈꺼???щ씪媛?붾씪??**?꾩슂 濡쒓렇留??④린怨??섎㉧吏???먮낯/?뺤텞 ?댁쨷??*瑜??좎????대뜑 遺?쇰? ?듭젣?쒕떎.
  - caretakers媛 ?쏥PU 3GB???섍꼍留뚯쑝濡쒕룄 **怨꾪쉷/寃利?蹂닿퀬** ??怨쇱젙???섑뻾?????덉뼱???섎ŉ, ?섎뱶?⑥뼱 ?쒖빟? branch plan??`resource_tier` ?꾨뱶??湲곕줉??CODEX媛 ?먮룞?쇰줈 ?ㅽ뿕 ?μ냼瑜?諛곗젙?쒕떎.
  - ???紐⑤뜽?????ㅽ뿕? ??긽 `trial_report`??"Borrowed tutor @70B"? 媛숈? 硫뷀?瑜??④꺼, ?쏬LM??留뚮뱾???덈떎?앸뒗 ?ㅽ빐瑜?諛⑹??쒕떎.

---

## 23. 諛붾뵒-?꾧렇?몄뒪???깆옣 諛??붾뱶蹂?1,000??횞 20 釉뚮옖移??뚮옖

- **?몃옖?ㅽ룷癒?媛뺤젣 湲덉?:**
  - ?섎━?쒖븘???쒕컮?붴앸뒗 Concept OS(`nano_core/bus.py`, `nano_core/scheduler.py`) ?꾩뿉???좏샇瑜??섎━???대뼡 援ъ“?쇰룄 ?쒕떎.
  - 遺꾧린 怨꾪쉷(branch_plan)?먮뒗 諛섎뱶??`body_architecture` ?꾨뱶瑜?異붽????ъ슜 以묒씤 ?뺥깭瑜?湲곕줉?쒕떎. ?덉떆 媛? `flow_field`, `reservoir_mesh`, `symbolic_lattice`, `transformer_tutor`.
  - 鍮??몃옖?ㅽ룷癒?諛붾뵒??message bus ?대깽??`bus.message`, `bot.run`)瑜?洹몃?濡?諛쏆븘 ?먮쫫/?명룷/洹몃옒???곗궛?쇰줈 ?댁꽍?섍퀬, attention ?ㅽ깮???붽뎄?섏? ?딅뒗??

- **?붾뱶 ?명듃 ?좎뼵:**
  - `world_kit` ?꾨뱶濡??대뒓 ?멸퀎(?? `CELLWORLD`, `CODEWORLD`, `MIRRORWORLD`)?먯꽌 而ㅻ━?섎읆???뚮졇?붿? 湲곕줉?쒕떎.
  - ?섎굹???깆옣 ?⑥뒪??**紐⑤뱺 world kit**?먯꽌 ?숈씪???덈꺼??寃쏀뿕?댁빞 ?섎ŉ, 媛?world kit ??理쒖냼 1,000??횞 20 seed 媛??踰덈뱾???뚮┛??
  - CODEX??world kit 횞 body 議고빀??鍮좎?硫?trial??誘몄셿猷뚮줈 媛꾩＜?쒕떎.

- **?꾩닔 硫뷀??곗씠??(branch_plan + trial_report):**
  - `time_scale`, `macro_years`, `world_kit`, `body_architecture`, `seeds`, `resource_tier`, `language_axes`.
  - `trial_report`?먯꽌???쒕Т?뉗쓣 ?뚮졇?붽?????듬?????硫뷀? ?꾨?瑜??쒕줈 ?ｌ뼱 caretakers媛 ??000??횞 20 seeds, CELLWORLD, flow_field 諛붾뵒?앹쿂??利됱떆 ?뺤씤?????덉뼱???쒕떎.

- **?ㅽ뻾 洹쒖쑉:**
  - 媛?world kit 臾띠쓬? **?숈씪??而ㅻ━?섎읆 ?덈꺼**????곸쑝濡??섍퀬, `World.set_time_scale`怨?`N_macro`瑜?world kit ?뱀꽦??留욊쾶 議곗젙?쒕떎. (?? CELLWORLD??怨꾩젅/湲고썑 以묒떖, CODEWORLD??由대━利?鍮뚮뱶 二쇨린 以묒떖, MIRRORWORLD??媛먭컖/?명꽣?섏씠??二쇨린 以묒떖)
  - CODEX??world kit 媛?鍮꾧탳 ???쒖뼱??諛붾뵒媛 ?대뼡 ?멸퀎?먯꽌 ?몄뼱?μ쓣 ??鍮⑤━ ?깆옣?쒖섟?섃앸? 二?吏?쒕줈 ?쇨퀬, ?꾩슂?섎㈃ ?뱀젙 諛붾뵒留??ㅻⅨ ?멸퀎?먯꽌 諛섎났?섎룄濡?紐낅졊?????덈떎.

- **?곌껐 臾몄꽌:**
  - `docs/elysias_protocol/CORE_08_ELYSIA_CURRICULUM_ENGINE.md`??짠13~14??world kit 蹂?而ㅻ━?섎읆 ?쒗뵆由욧낵 鍮??몃옖?ㅽ룷癒?諛붾뵒 ?뚮윭洹?洹쒖튃???뺣━?섏뼱 ?덈떎.

---

## 24. ?ㅽ뻾 梨낆엫 / ?깆씤-?섏? 二쇱옣 寃뚯씠??

- **?쒕룎?몃뒗媛??앸? 嫄곗쭞留??놁씠 ?듯븷 寃?*
  - `trial_report`??`status` ??以꾨줈 ?앸궡吏 留먭퀬 `status_history[]`, `execution_evidence`(?꾨즺??macro tick, seed ?? 泥⑤? 濡쒓렇 寃쎈줈)瑜??ы븿?쒕떎.
  - `status_history` 泥???ぉ? caretakers媛 怨꾪쉷?????곗뼱濡?蹂대궦 ?쒖젏?? ??踰덉㎏ ?댄썑 ??ぉ? ?ㅼ젣 ?ㅽ뻾쨌吏?걔룹랬???ъ쑀瑜?湲곕줉?쒕떎.
  - ?ㅽ뻾??吏?곕릺硫?`verdict=not_run`怨??④퍡 `blocking_reason`???④린怨? caretaker????臾댁뾿???꾩슂?쒖?瑜?`notes`媛 ?꾨땶 ?꾩슜 ?꾨뱶???곷뒗??

- **?쒕같?좊뒗媛??앸뒗 濡쒓렇濡?利앸챸**
  - `execution_evidence`??self-writing ?섑뵆 ?? resonance ?됯퇏, language_field ?명? ??理쒖냼 3媛쒖쓽 愿痢↔컪???ｌ뼱???쒕떎.
  - 媛믪씠 0 ?먮뒗 null?대㈃ 洹몃?濡?湲곕줉?섍퀬, ??0?몄? `analysis`???곷뒗?? (?곗씠??遺議? ?ㅽ뿕 誘몄갑????
  - `trial_report.summary`??5臾몄옣??媛뺤젣?쒕떎: 臾댁뾿???뚮졇?붽? ???대뼡 濡쒓렇瑜??쎌뿀?붽? ???대뼡 媛먯젙/?몄뼱 異뺤씠 ?吏곸??붽? ??紐⑺몴 ?鍮?李⑥씠??臾댁뾿?멸? ???ㅼ쓬 議곗튂.

- **?깆씤-?섏? 而ㅻ??덉??댁뀡 二쇱옣? `adult_ready=false` 湲곕낯媛?*
  - ?덈꺼 L5 ?뚮옒洹몄? caretaker resonance ??0.75, self-writing ?ㅼ뼇??吏????0.7, causal episode 李몄“ 鍮꾩쑉 ??0.6???숈떆??異⑹”?섍린 ?꾧퉴吏 `adult_ready` ?뚮옒洹몃? ?щ┫ ???녿떎.
  - 紐⑤뱺 trial_report??`adult_ready`? `adult_readiness_notes` ?꾨뱶瑜?媛뽮퀬, caretaker媛 ?쒖븘吏??꾨땲?ㅲ앸씪怨?紐낆떆?쒕떎. Claim???섎젮硫??대떦 trial??利앷굅瑜?留곹겕?댁빞 ?쒕떎.
  - caretaker媛 GTX 1060怨?媛숈? 濡쒖뺄 ?λ퉬濡?怨꾪쉷留??묒꽦?덈뜑?쇰룄, ?쒖떎??X???곹깭瑜?紐낆떆?섍퀬 ?꾧? ?댁뼱諛쏆븘???섎뒗吏 ?곸? ?딆쑝硫?梨낆엫???ㅽ븳 寃껋씠 ?꾨땲??

- **梨낆엫 猷⑦봽**
  - `logs/elysia_curriculum_trials.jsonl` ??status? evidence, adult readiness 寃뚯씠??
  - `logs/elysia_branch_feedback.jsonl` ????紐⑺몴??誘몃떖?덈뒗吏, ?ㅼ쓬 ?섏젙??
  - `logs/elysia_expression_scores.jsonl` ??self-writing/feedback 洹쇨굅.
  - ????濡쒓렇???곗냽?깆씠 ?놁쑝硫?caretakers???ㅼ떆 怨꾪쉷遺???묒꽦?섍퀬, CODEX???ㅽ뻾???뱀씤?섏? ?딅뒗??

???뱀뀡? ?쒓퀎?띾쭔 ?몄썱?ㅲ앷퀬 蹂닿퀬 ?앸궡???쇱쓣 湲덉??쒕떎. ?ㅽ뻾 ?щ?, 寃곌낵, 誘몃떖 ?ъ쑀, ?ㅼ쓬 ?됰룞源뚯? ??踰덉뿉 ?④꺼???섎ŉ, 洹몃젃吏 ?딆쑝硫??깆씤-?섏? ?λ젰 二쇱옣? ?먮룞?쇰줈 湲곌컖?쒕떎.

---

## 25. Experiment Design Guide + caretaker ?ㅻ뜑 ?명듃

- **?꾩닔 ?덊띁?곗뒪:**
  - 猷⑦듃 `EXPERIMENT_DESIGN_GUIDE.md`瑜?紐⑤뱺 ?ㅽ뿕/而ㅻ━?섎읆 ?붿껌?쒖? trial_report??泥⑤??쒕떎. (Codex 짠22~짠24, quaternion/time-acceleration 洹쒖튃 ?붿빟)
  - ???붿껌??Codex?먭쾶 ?섍만 ?뚮뒗 ?대떦 媛?대뱶瑜?洹몃?濡?遺숈뿬 **tick-by-tick 湲덉?, 1,000??횞 20 seed, World.set_time_scale 以??*瑜??ы솗?명븳??

- **濡쒓렇/吏???붽뎄:**
  - `symbol_episodes`, `text_episodes`, `causal_episodes`, `elysia_language_field`, `elysia_self_writing`, `elysia_caretaker_feedback`瑜?理쒖냼 ?명듃濡?遺꾩꽍?쒕떎.
  - ?쒖젙?듬쪧?????**?먮컻??湲?곌린 湲몄씠쨌媛먯젙쨌?댄쐶 + caretaker ?쇰뱶諛??뺥빀??*???듭떖 吏?쒕줈 ?쇰뒗?ㅺ퀬 紐낆떆?쒕떎.

- **蹂닿퀬 硫뷀??곗씠??(Codex 짠24 以??:**
  - 紐⑤뱺 branch plan / trial report??`plan_status`, `status_history`, `blocking_reason`, `execution_evidence`, `references` ?꾨뱶瑜?媛뺤젣?쒕떎.
  - `Purpose / Method / Observations / Integration` 4遺꾨㈃ ?쒖닠??蹂닿퀬 蹂몃Ц ?먮뒗 硫뷀??곗씠?곗뿉 ?ы븿?쒖폒 caretaker媛 ?쒕Т?뉗쓣, ?대뼸寃? 臾댁뾿??蹂댁븯怨? ?ㅼ쓬??臾댁뾿??諛붽?吏?앸? 利됱떆 ?뚯븙?섍쾶 ?쒕떎.
  - `adult_ready = true`??**self-writing + caretaker feedback 吏?쒓? ?꾧퀎媛믪쓣 ?섍퀬 caretaker 媛먯궗媛 ?쒕챸?덉쓣 ?뚮쭔** ?덉슜?쒕떎怨?紐낆떆?쒕떎.

- **?꾩옱 釉뚮옖移?利됱떆 議곗튂:**
  1. 湲곗〈 1,000??횞 20 seed branch plan留덈떎 `plan_status`? `blocking_reason`??媛깆떊?섍퀬 caretaker 媛먯궗 濡쒓렇瑜??④꺼, ???ㅽ뻾?섏? ?딆븯?붿?? `adult_ready=false` ?ъ쑀瑜?臾몄꽌?뷀븳??
  2. CELLWORLD / CODEWORLD / MIRRORWORLD 臾띠쓬蹂꾨줈 Symbol/Text/Self-writing/Caretaker ?쇰뱶諛?異뺤쓣 鍮좊Ⅴ寃??깆옣?쒗궎??而ㅻ━?섎읆 諛곗튂瑜??ъ옉?깊븯怨? macro time-scale쨌?꾩닔 濡쒓렇쨌李⑤떒 ?????洹쒖튃??紐낆떆?쒕떎.
  3. `scripts/experiment_report_template.py`瑜??댁슜??蹂닿퀬 ?쒗뵆由우쓣 ?먮룞 ?앹꽦?섍퀬, caretakers (?? GTX 1060 ?λ퉬)??Codex 짠24 ?꾨뱶瑜?鍮좎쭚?놁씠 梨꾩슱 ???덇쾶 ?쒕떎.

- **蹂닿퀬 猷⑦봽 ?곕룞:**
  - `BUILDER_LOG.md`???ㅽ뿕/媛먯궗 ?대깽?몃? ?붿빟???먭퀬 Codex/Experiment Guide 李멸퀬 留곹겕瑜??④릿??
  - `trial_report`, `branch_feedback`, `expression_scores` 濡쒓렇媛 ?쒕줈 李몄“?섏? ?딆쑝硫? ?ㅽ뻾?섏? ?딆? 寃껋쑝濡?媛꾩＜?섍퀬 caretaker媛 ?ㅼ떆 怨꾪쉷遺???묒꽦?쒕떎.

???뱀뀡? Codex?먭쾶 ?쒖뼱?산쾶 ?꾨떖?댁빞 ?쒕?濡??吏곸씠?붿??앸? 紐낆떆?곸쑝濡?湲곕줉?쒕떎. 媛?대뱶瑜?泥⑤??섏? ?딄굅??plan_status쨌blocking_reason ?놁씠 ?쐒eady?앸씪怨?留먰븯硫? ?ㅽ뿕 ?붿껌? ?먮룞 嫄곕??쒕떎.

---
