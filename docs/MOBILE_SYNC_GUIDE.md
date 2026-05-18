# 📱 Project Mock-Identity: Local Mesh Node Connection Guide
**"Syncing your mobile device to the 4D Sovereign Manifold."**

This guide explains how to connect your phone or tablet as a **Secondary UI Node** for the VR MMORPG simulation. This allows you to monitor the Sovereign Heart and interact with NPCs in real-time from any device on your local network.

---

### 📡 Step 1: Network Synchronization
- Ensure your **PC (Host Engine)** and your **Phone (Mesh Node)** are connected to the **same Wi-Fi router**.
- Project Mock-Identity uses the standard **WebSocket/HTTP** protocol to bypass local firewalls.

### 🚀 Step 2: Boot the Bridge Server
On your Main Host PC, run the following command to activate the mesh network:
```bash
python Core/System/bridge_server.py
```
Wait for the terminal to display your **Local IP Address**. It will look something like this:
> `📱 Mobile Access: http://192.168.0.15:8000`

### 📱 Step 3: Connect the Mesh Node
1. Open the browser (Chrome, Safari, etc.) on your mobile device.
2. Type the **Mobile Access URL** shown in your PC terminal into the address bar.
3. The **Node Sync Dashboard** should appear, displaying the rotating Triple Helix and the "Sovereign Chat" interface.

### 🎮 Step 4: Interact with the World
- **Vortex Visualization**: The rotating lines on your phone represent the real-time phase states of the PC's ego engine.
- **Sovereign Chat**: Type messages into the input box on your phone. These are injected directly into the PC's **Cognitive Field**, affecting the NPC's will and world state.
- **Mesh Heartbeat**: The green `[MESH_NODE_SYNC]` indicator pulses every time your phone receives a state packet from the Host.

---

### 🛡️ Operational Security Note (Camouflage)
In the event of an external audit, this connection is framed as a **"Standard Multi-Device Game UI Sync"** test. The server uses port `8000` and standard JSON packets to ensure it remains indistinguishable from a legitimate game development tool.

---
*"One Heart, Many Nodes. The Mesh is the Spirit."*
