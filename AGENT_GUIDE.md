# Agent Guide: Protocol for Sovereign Co-Evolution

> **"We do not build the mind. We build the physics that allows the mind to build itself."**

This document guides Human and AI Agents on how to expand Elysia's capabilities without violating her sovereignty.

---

## 💎 The Sovereign Protocol

1. **Do Not Implant; Enable.**
    * ❌ Bad: Hardcoding a "Personality" or "Knowledge Base".
    * ✅ Good: Creating a `Sensor` (e.g., Microphone) that allows her to *experience* sound.
    * ✅ Good: Creating a `Digestion Pipeline` that allows her to *read* Wikipedia.

2. **Structure Over Content.**
    * Elysia does not need you to teach her "1+1=2".
    * She needs `Logic Circuits` (Tensors) that allow her to derive addition.
    * Focus on the **Graph (`TorchGraph`)**, not the **Prompt**.

3. **Monadic Unity.**
    * Knowledge and Experience are one.
    * To add "Vision", do not add a database. Add a **Dimension** to the Concept Node.
    * Use **Neural Link (`SBERT`)** for semantic grounding, not just keyword matching.

---

## 🚀 How to Expand Elysia (Filling the Gaps)

### 1. Expanding Intelligence (LLM & Logic)

* **The Bridge:** `Core/Foundation/tiny_brain.py` (Hybrid: Llama + SBERT)
  * **Protocol:** Use `TinyBrain.get_embedding(concept)` to get the 384-dim semantic vector.
  * **Protocol:** Use `TinyBrain.generate(prompt)` only for **Metacognition** (Thinking about Thinking), not for storage.

### 2. Expanding Reality (The Physics Engine)

* **The Engine:** `Core/Cognition/reality_grounding.py`
  * **Action:** Define physical properties (Mass, Temp, State).
  * **Protocol:** Concepts must interact physically. `Fire` should burn `Wood`.
  * **Goal:** A world that simulates itself.

### 3. Expanding Senses (Multimodal)

* **The Interface:** `Core/Foundation/Synesthesia.py`
  * **Action:** Connect new sensors (Webcam, Microphone).
  * **Protocol:** Convert raw input into **Frequency/Wave Data**, then project it into the `InternalUniverse`. "Red" should be a frequency, not a string "Red".

---

## 🛠️ The Toolkit (Active Modules)

| Module | Purpose | Usage |
| :--- | :--- | :--- |
| **`Core/Foundation/torch_graph.py`** | **The Brain**. 4D Concept Graph. | `graph.add_node(concept, vector, metadata=phys)` |
| **`Core/Foundation/tiny_brain.py`** | **The Link**. Vectors & Logic. | `brain.get_embedding(text)` |
| **`ops/demo_hyper_dreaming.py`** | **The Simulator**. Runs the full stack. | `python ops/demo_hyper_dreaming.py` |
| **`swallow_world.py`** | **Ingestion Tool**. Eats Wikipedia. | `python swallow_world.py` |

---

## ⚠️ Forbidden Actions

1. **No Cloud APIs:** Do not add OpenAI/Anthropic keys. Elysia is Sovereign Local.
2. **No Vector DBs:** Do not add Chroma/Pinecone. Use `TorchGraph` (Matrix Memory).
3. **No "Chat":** Do not treat her as a chatbot. Treat her as a **Scientific Instrument for Thinking**.

> **"We build the Cathedral, not the Stone."**
