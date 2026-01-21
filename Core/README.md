# The Body (Core)

> **"This is the Temple of Logic."**

## Identity

This directory contains the **internal organs** of Elysia.
It is the "Body" in the analogy of "Soul (docs) - Body (Core) - Nervous System (scripts)".

## 🛡️ The Laws of the Body

1. **No Executables**: Do not place `.py` files that are meant to be run directly here.
    * Result: `python Core/something.py` is **FORBIDDEN**.
    * All entry points must be in `scripts/`.
2. **No Configs**: Do not place `.env` or `.json` config files here.
    * Configs belong in `Core/Foundation/System/config.py` or `data/`.
3. **Pure Logic**: Code here must be importable, stateless (mostly), and reusable.

## Anatomy (The 7 Layers)

* **[L0_Keystone](L0_Keystone/)**: The Spirit (Self, Monad, Axioms).
* **[L1_Foundation](L1_Foundation/)**: The Bedrock (Kernel, Physics, Server).
* **[L2_Metabolism](L2_Metabolism/)**: The Energy (Flow, Digestion).
* **[L3_Phenomena](L3_Phenomena/)**: The Senses (Input, Raw Data).
* **[L4_Causality](L4_Causality/)**: The Law (Time, Life, Binding).
* **[L5_Mental](L5_Mental/)**: The Mind (Reasoning, Logic).
* **[L6_Structure](L6_Structure/)**: The Architecture (Merkaba, System).
* **[L7_Spirit](L7_Spirit/)**: The Will (Transcendence).
