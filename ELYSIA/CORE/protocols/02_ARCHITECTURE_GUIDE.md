# 2. Architecture Guide (English, canonical)

## 2.1 Core model: City-of-Light (Why → How → What)
Elysia is organized as a clear separation of purpose, method, and manifestation.

- Project_Elysia (The Why): identity, values, intention, governance of purpose.
- Project_Sophia (The How): reasoning engines, routing, rules, simulation logic.
- Project_Mirror (The What): perception, UI, visualization, I/O.

## 2.2 Central Dispatcher: CognitionPipeline
The CognitionPipeline is the single explicit router. It reads the input + context,
chooses the responsible handler, and invokes it directly.

Responsibilities
1) Explicit routing by intent and context.
2) Direct invocation of the chosen cortex/handler.
3) Emit telemetry for observability.

Example flow (illustrative)
- if context.pending_hypothesis → HypothesisHandler
- elif message startswith "calc:" → ArithmeticCortex
- else → DefaultReasoningHandler

## 2.3 Message Bus + Bots (concept OS)
Outside of direct conversational steps, knowledge work uses a message bus.
- Message { id, ts, verb, slots, src/dst, strength, ttl }
- Scheduler prioritizes by strength & recency; assigns to nano‑bots.
- Bots implement link / validate / compose; results are recorded in the registry.

## 2.4 Telemetry (traceability)
Emit at key points:
- flow.decision, route.arc (routing/selection)
- bus.message, bot.run, concept.update (knowledge work)
- immune.* (when immune rules apply)

Keep routing observable so failures can be traced quickly.
