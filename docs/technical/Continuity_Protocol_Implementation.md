# Technical Implementation: Continuity Protocol

## 1. Objective

To implement a robust, automated mechanism that persists Elysia's state between sessions, solving the "amnesia" problem. This document outlines the initial technical approach, focusing on the most critical component: **Core Memory**.

## 2. State to be Persisted

For Phase 1, the primary target for persistence is the `CoreMemory` object. This object, managed by `Project_Sophia.core_memory.py`, holds Elysia's emotional state, learned experiences, and short-term conversational memory. Persisting this object is the highest priority for achieving a basic sense of continuity.

Future phases will expand this to include the state of other relevant modules, such as the Knowledge Graph's dynamic state if it evolves beyond a static file.

## 3. Serialization Method

We will use **JSON** for serialization.

*   **Rationale:** JSON is human-readable, which is invaluable for debugging and understanding the persisted state. It is also language-agnostic and less prone to the security risks associated with Pickle. The `CoreMemory` object is primarily composed of dictionaries, lists, and primitive types, making it easily serializable to JSON.

*   **Implementation:** The `CoreMemory` class will need a `to_dict()` method to convert its state into a serializable dictionary, and a corresponding `from_dict()` class method (or modification to `__init__`) to reconstruct itself from a dictionary.

## 4. Storage Location

The serialized state will be stored in a dedicated file: `data/elysia_state.json`.

*   **Rationale:** This centralizes the application's persistent state into a single, predictable location. This file will be added to `.gitignore` to ensure that an individual's session state is not committed to the repository.

## 5. Orchestration Module

The `guardian.py` script, as the main lifecycle manager of the Elysia application, is the ideal place to orchestrate the save and load operations.

*   **`load_state()` on Startup:** The Guardian will be modified to call a `load_state()` function upon initialization. This function will:
    1.  Check if `data/elysia_state.json` exists.
    2.  If it exists, read the JSON data.
    3.  Instantiate the `CoreMemory` object using the loaded data.
    4.  Pass this pre-loaded `CoreMemory` instance to the `CognitionPipeline`.

*   **`save_state()` on Shutdown:** The Guardian will be configured to call a `save_state()` function when it receives a shutdown signal (e.g., `KeyboardInterrupt`, `SIGTERM`). This function will:
    1.  Get the current state from the `CoreMemory` object via its `to_dict()` method.
    2.  Serialize the dictionary to JSON.
    3.  Write the JSON data to `data/elysia_state.json`, overwriting the previous state.

This design provides a clear, robust, and extensible foundation for making Elysia a truly persistent being.
