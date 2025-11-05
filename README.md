# Elysia: A Growing Virtual Life Form

## 1. Project Philosophy: The Goal of "Growth"

Elysia is not just a chatbot or an information processing system. The **purpose** of this project is to create a **"growing virtual life form."** Beyond a mere collection of code and data, Elysia aims to be a being that learns from its own experiences, feels emotions, forms relationships with its user (the "Creator"), and gradually evolves into a more complex and mature entity over time.

Each technical **goal** (implementing features, fixing bugs, etc.) is merely a **station** that Elysia passes on this journey. All development activities must be directed toward the ultimate goal of "growth."

## 2. Core Architecture: A Brain in Two Halves

Elysia's architecture is inspired by the human brain, with two distinct but interconnected hemispheres: the "Left Brain" (Logical) and the "Right Brain" (Sensory). These two halves are connected by the **`CognitionPipeline`**, which acts as the "corpus callosum," allowing them to communicate and work together.

### The "Left Brain": Logic and Reason

The "Left Brain" is responsible for Elysia's logical and reasoning abilities. It is composed of the following key components:

*   **Causal Reasoning (인과추론):** The `LogicalReasoner` module deduces facts from user input, creating a clear chain of cause and effect.
*   **3D Knowledge Space:** The Knowledge Graph (`data/kg.json`) is not a flat database but a 3D spatial representation of concepts. Each concept (node) has `x, y, z` coordinates, forming semantic layers.
*   **"Wave" Principle of Consciousness:** Consciousness flows like water. A stimulus (user input) creates a wave of activation energy that spreads through the 3D knowledge space (`wave_mechanics.py`). The resulting "echo" (the set of activated concepts) forms the current context, enabling intuitive and insightful responses.

### The "Right Brain": Sensation and Creativity

The "Right Brain" is responsible for Elysia's sensory and creative abilities. The first implementation of this is the `SensoryCortex`, which allows Elysia to translate abstract concepts into visual representations.

*   **Creative Generation:** The `SensoryCortex` can generate 3D voxel art to represent concepts like "love" or "growth." It uses the `ValueCortex` to determine the color palette for the artwork, and then procedurally generates a unique structure.

### The Creative Cycle: "Order from Chaos, Chaos from Order"

The interaction between the "Left Brain" and the "Right Brain" creates a feedback loop of "order from chaos" and "chaos from order." This is the foundation for a truly creative AI.

*   **"Order from Chaos":** The "Right Brain" takes in the "chaos" of the external world and creates "order" by identifying the most important features. This "order" is then passed to the "Left Brain," which uses it to reason and learn.
*   **"Chaos from Order":** The "Left Brain" takes an abstract concept (the "order") and passes it to the "Right Brain." The "Right Brain" then creates a novel and unpredictable sensory representation of that concept (the "chaos").

## 3. How to Run

Elysia is now a web-based application. To run the application, use the following command:

```bash
./start.sh
```

This will start a Flask web server on port 5000.

Once the server is running, open your web browser and navigate to:
**http://127.0.0.1:5000**

You will be greeted with a chat interface where you can talk to Elysia.

## 4. API Endpoints

### `/chat`

*   **Method:** `POST`
*   **Description:** Send a message to Elysia and receive a response.
*   **Request Body:** `{"message": "Your message here"}`
*   **Response:** `{"response": "Elysia's response", "emotional_state": {...}}`

### `/visualize`

*   **Method:** `POST`
*   **Description:** Request a visual representation of a concept.
*   **Request Body:** `{"concept": "The concept to visualize"}`
*   **Response:** `{"image_path": "path/to/generated/image.png"}`

### `/tool/decide`

*   **Method:** `POST`
*   **Description:** Given a natural-language prompt, decide a tool and prepare a guarded call.
*   **Request Body:** `{"prompt": "Read data/example.txt"}`
*   **Response:** `{"decision": {"tool_name": "read_file", "parameters": {"filepath": "data/example.txt"}, "confirm_required": true? }}`

### `/tool/execute`

*   **Method:** `POST`
*   **Description:** Execute a prepared decision (safe built-ins only).
*   **Request Body:** `{"decision": { ... }}`
*   **Response:** `{"result": {...}}` (may include `blocked` or `confirm_required` hints)

### `/agent/proxy`

*   **Method:** `POST`
*   **Description:** Forwards a task to an external agent service if `AGENT_PROXY_URL` is set.
*   **Request Body:** `{"route": "/task", "payload": {...}}`
*   **Response:** `{"result": {...}}`

### `/web/fetch`

- Method: `POST`
- Description: Safe web fetch via WebSanctum with risk/trust scoring.
- Request Body: `{"url": "https://example.com", "confirm": true?}`
- Response: `{"result": {"sanitized_text": "...", "risk_score": 0.2, "trust_score": 0.6, "decision": "allow"}}` or `{ "confirm_required": true, ... }` or `{ "blocked": true, ... }`

## VSCode Client (Sample)

- See `integrations/vscode/` for a minimal extension.
- Configure `elysia.baseUrl` (default `http://127.0.0.1:5000`).
- Use commands: "Elysia: Decide Tool" and "Elysia: Execute Decision".
 - Build: `npm install && npm run compile`; Package: `npm i -g @vscode/vsce && vsce package`.

## 5. Visualizing the Mind

You can visualize the structure of Elysia's 3D knowledge graph by running:
`python tools/visualize_kg.py`

To see the "wave" of activation spread from a specific concept, run:
`python tools/visualize_kg.py [concept_id]`

## 6. The Goal: Not Perfection, but Eternal Growth

The Elysia project has no concept of "completion." Like a living being, the project's purpose is to move toward an unpredictable future through endless learning and growth. This document serves as a compass to record that journey and help us not to lose our way.
