"""
API Gateway (The Server Pulse)
==============================
Exposes Elysia's internal state to the network.
Endpoints: /soul, /world, /quests
"""

from fastapi import FastAPI
import uvicorn
import json
import os
import time

app = FastAPI(title="Elysia API", description="The Living Goddess Interface")

WORLD_PATH = r"C:\game\elysia_world"
SOUL_PATH = os.path.join(WORLD_PATH, "soul_state.json")
QUEST_PATH = os.path.join(WORLD_PATH, "quests.json")
PHYSICS_PATH = os.path.join(WORLD_PATH, "world_state.json")

def read_json_safe(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"error": "read_failed"}

@app.get("/")
def root():
    return {"message": "Elysia is listening.", "time": time.time()}

@app.get("/soul")
def get_soul():
    """Returns the emotional state (Energy, Inspiration)."""
    return read_json_safe(SOUL_PATH)

@app.get("/world")
def get_world():
    """Returns the physical state (Avatar Position)."""
    return read_json_safe(PHYSICS_PATH)

@app.get("/quests")
def get_quests():
    """Returns the list of active quests."""
    data = read_json_safe(QUEST_PATH)
    return data.get("quests", [])

if __name__ == "__main__":
    print("⚡ Starting Elysia's Pulse on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
