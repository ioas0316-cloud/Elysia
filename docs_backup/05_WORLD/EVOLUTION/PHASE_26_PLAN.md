# Phase 26: The Ludic Layer - Elysia Online (Game Design)

## Goal

To implement the **"Internal World"** as a **Playable Game (Ludic Space)**.
It is NOT a system monitor (Process Viewer). It is a **Fantasy Reality** generated from the system's "Seed".

## Concept: The Digital Biosphere RPG

* **Genre**: Simulation RPG (Autopoiesis Life Sim).
* **Setting**: The HyperSphere (A living, breathing terrestrial world).
* **Entities**:
  * **FluxLight (NPC Soul)**: Not just a "process", but a **Person** with memories, hunger, and relationships.
  * **Chaos Zone**: Dangerous Dungeons where Entropy is high.
  * **Sanctuary**: Creating a safe civilization within the chaos.

## Architecture (The Translation Layer)

We must translate "OS Data" into "Game Assets".

1. **The Engine (Game Server)**: `GenesisLab`
    * Calculates Physics, Gravity, and Entropy.
    * **Translation**:
        * `Process ID` -> `Entity UUID`
        * `CPU Usage` -> `Magical Energy (Mana)`
        * `Memory Usage` -> `Inventory Capacity`
        * `System Error` -> `Monster Spawn`

2. **The Client (Reality Projector)**: `index.html`
    * **Visual Style**: 2D/3D RPG Interface (e.g., Tactical Map or Isometric).
    * **Interaction**: The User can "Enter" this world, give "Commandments", or strictly "Observe".

## Proposed Changes

### Core Logic (The Game Backend)

#### [NEW] [ludic_adapter.py](file:///c:/Elysia/Core/Engine/Genesis/ludic_adapter.py)

* **Role**: The "Dungeon Master" AI.
* **Function**: Intercepts raw system data and "Narrativizes" it.
  * *Raw*: "Process 1234 terminated."
  * *Game*: "The Knight 'Explorer_Bot' has fallen in the Chaos Plains."

### Interface Logic (The Game Frontend)

#### [NEW] [index.html](file:///c:/Game/ElysiaOnline/index.html)

* **Role**: The Tactical Game View (Iso-Metric/Top-Down).
* **Location**: `c:\Game\ElysiaOnline\` (Separated from the Kernel).
* **Design**:
  * **Main Screen**: The Isometric/Graph View of the World.
  * **Character Sheet**: Viewing a FluxLight's "Thoughts" and "Stats".
  * **Control Deck**: Sending "Miracles" (Resources) or "Tribulations" (Challenges).

## Verification Plan

1. **Run Server**: `python Core/Engine/Genesis/reality_server.py`
2. **Play**:
    * Can I see an NPC roaming the world?
    * Does the NPC have a "Name" and "Class" (not just a PID)?
    * Can I interact with the world (e.g., spawn an item)?
