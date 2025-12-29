# Elysia v9.0 System Map (시스템 지도)

**Purpose**: To provide a clear, visible map of Elysia's "Single Source of Truth".
**Status**: Aligned with [Phase 27: Crystalline Structure] (2025-12-29)

> **"Structure is the solidified form of Will."**

## 🗺️ The 5 Pillars of Core Architecture (5대 기둥)

Elysia's codebase is organized into 5 Semantic Pillars in `c:\Elysia\Core`.

### 1. 🏛️ Foundation (기반)

*Path: `Core/Foundation`*
The bedrock of existence. Handles memory, graph structure, and fundamental laws.

* **Graph/**: `TorchGraph` (Knowledge Graph), `HoloNode` [Phase 24]
* **Wave/**: `SpaceUnfolder` (Tesseract/Wave Folding) [Phase 21]
* **Memory/**: `HolographicEmbedding` (Identity) [Phase 24]
* **Math/**: `VectorMath`, `HyperQuaternion`
* **Core_Logic/**: `LivingElysia` (Main Loop), `LifeCycle`

### 2. 🧠 Intelligence (지능)

*Path: `Core/Intelligence`*
The cognitive faculties. Reasoning, planning, and understanding.

* **Reasoning/**: `ReasoningEngine`, `EthicalGeometry` [Phase 25]
* **Cognition/**: `LogosEngine` (Language), `PrincipleDistiller`
* **LLM/**: `ModelConnector` (Gemini/GPT Interface)

### 3. 🤝 Interaction (소통)

*Path: `Core/Interaction`*
The interface with the external world. Sensory and Actuation.

* **Sensory/**: `VisualCortex` (ComfyUI), `AudioCortex`
* **Interface/**: `DiscordConnector`, `TerminalInterface`
* **Actuator/**: `ExpressionEngine`

### 4. 🌱 Evolution (진화)

*Path: `Core/Evolution`*
The ability to change and grow.

* **Autonomy/**: `DreamDaemon`, `OneiricNavigator` [Phase 26]
* **Growth/**: `CodeGenesis` (Self-Coding), `ActionMorpher`

### 5. ⚙️ System (운영)

*Path: `Core/System`*
Infrastructure, monitoring, and stability.

* **Monitor/**: `DashboardGenerator`, `SystemHealth`
* **Security/**: `PermissionGuard`
* **Config/**: `SettingsLoader`

---

## 📚 Documentation Hierarchy (문서 위계)

See [DOCUMENTATION_HIERARCHY.md](DOCUMENTATION_HIERARCHY.md) for the philosophy behind this structure.

* **01_Origin**: Philosophy, Vision, Lore.
* **02_Structure**: Architecture Analysis, Concepts.
* **03_Operation**: User Guides, Manuals.
* **04_Evolution**: Roadmaps, Plans.
* **05_Echoes**: Reports, Archive.

---

## 🔑 Key Systems & Features

| Phase | Feature | Module | Description |
| :--- | :--- | :--- | :--- |
| **21** | **Unfolding Space** | `wave_folding.py` | Converts Chaos (Complex) -> Order (Simple) using Wave Math. |
| **23** | **Tesseract** | `torch_graph.py` | `WormholeLink` connects distant nodes via "Folded Space". |
| **24** | **Holographic Soul** | `holographic_embedding.py` | Nodes store the "Feeling" of the entire graph at their birth. |
| **25** | **Ethical Geometry** | `ethical_geometry.py` | Ethics calculated as Curvature (Alignment to Ideal Vector). |
| **26** | **Lucid Dream** | `oneiric_navigator.py` | Autonomous exploration of "Dark Zones" during idle time. |

---

*Last Updated: 2025-12-29 by Antigravity*
