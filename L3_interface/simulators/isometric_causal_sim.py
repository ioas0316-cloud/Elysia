r"""
Standalone Interactive Isometric Causal Simulator (`simulators/isometric_causal_sim.py`).

Demonstrates 2.5D Isometric Town Building & Multi-Scale Lens simulation powered by Elysia's Causal Engine,
with 4-Constraint Elysian Intervention Mechanics.
Supports Scale Lens zooming (Micro Sims, Mid City, Macro Civ), SVG exporting, and standalone interactive HTML Canvas visualization output.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from modules.causal_game_engine.isometric_adapter import IsoTileConfig
from modules.causal_game_engine.isometric_causal_engine import IsometricCausalEngine


class IsometricCausalSimulator:
    """
    Simulator wrapper for running multi-scale isometric town dynamics, logging state trajectories,
    and rendering standalone SVG and HTML Canvas interactive viewers.
    """

    def __init__(self, width: int = 12, height: int = 12):
        self.width = width
        self.height = height
        config = IsoTileConfig(tile_width=64.0, tile_height=32.0, elevation_scale=16.0, origin_x=width * 32.0, origin_y=60.0)
        self.engine = IsometricCausalEngine(width, height, projection_config=config)

        # Initialize sample town layout with CC0 assets & terrain
        self._setup_initial_town_layout()

        # Telemetry history
        self.telemetry_history: List[Dict[str, Any]] = []

    def _setup_initial_town_layout(self):
        """Builds an initial town layout with Town Hall, Houses, Farms, Trees, and Water hazards."""
        self.engine.tilemap.set_building(5, 5, "town_hall")
        self.engine.tilemap.set_building(3, 4, "granary")
        self.engine.tilemap.set_building(7, 4, "house")
        self.engine.tilemap.set_building(4, 7, "house")

        self.engine.tilemap.set_building(2, 2, "farm_field")
        self.engine.tilemap.set_building(8, 8, "lumber_camp")

        self.engine.tilemap.set_resource(1, 8, "tree_oak")
        self.engine.tilemap.set_resource(2, 9, "tree_oak")
        self.engine.tilemap.set_resource(9, 2, "stone_quarry")
        self.engine.tilemap.set_resource(10, 3, "gold_mine")

        for x in range(8, 12):
            for y in range(0, 4):
                self.engine.tilemap.set_elevation(x, y, 1.5)

        for y in range(8, 12):
            self.engine.tilemap.get_cell(0, y).terrain_asset_id = "water"

        # Spawn Villager, Builder, and Anchor NPCs
        self.engine.spawn_npc("npc_1", 5.0, 6.0, "villager", is_anchor=False)
        self.engine.spawn_npc("npc_2", 4.0, 5.0, "builder", is_anchor=False)
        self.engine.spawn_npc("npc_anchor", 6.0, 4.0, "saint_prophet", is_anchor=True, sync_rate=0.9)

    def run_simulation(self, steps: int = 30) -> List[Dict[str, Any]]:
        """Runs the simulation for N steps across Micro, Mid, and Macro Scale Lenses."""
        for step_idx in range(steps):
            if step_idx < 10:
                self.engine.set_scale_lens(0.1)  # Micro Sims
            elif step_idx < 20:
                self.engine.set_scale_lens(0.5)  # Mid City
            else:
                self.engine.set_scale_lens(0.9)  # Macro Civ

            # Triggers & Elysian Interventions demonstration
            if step_idx == 5:
                # Trigger famine (Food drop) -> Survival Lens
                self.engine.lens_state.resource_reserves["food"] = 10.0
            elif step_idx == 8:
                # Indirect Intervention via Anchor NPC (High sync rate -> Low cost & low rebound)
                self.engine.apply_elysian_intervention(
                    action_type="INJECT_IDEA",
                    target_x=6, target_y=4,
                    anchor_npc_id="npc_anchor",
                    is_indirect=True
                )
            elif step_idx == 12:
                # Harvest boost -> Gathering Lens
                self.engine.lens_state.resource_reserves["food"] = 60.0
                self.engine.lens_state.resource_reserves["wood"] = 20.0
            elif step_idx == 15:
                # Direct Miracle Intervention (High cost & high rebound side effects)
                self.engine.apply_elysian_intervention(
                    action_type="SPAWN_BUILDING",
                    target_x=2, target_y=6,
                    is_indirect=False
                )
            elif step_idx == 22:
                # Geopolitical war strain -> Macro Empire Expansion Lens
                self.engine.macro_geopolitical_tension = 50.0

            log = self.engine.step(delta_time=0.2)
            log["step"] = step_idx + 1
            self.telemetry_history.append(log)
        return self.telemetry_history

    def export_svg(self, filename: str = "isometric_town_render.svg") -> str:
        """Renders current isometric tilemap into SVG image vector format."""
        w2 = self.engine.tilemap.projection.config.tile_width / 2.0
        h2 = self.engine.tilemap.projection.config.tile_height / 2.0

        canvas_w = int(self.width * 64 + 100)
        canvas_h = int(self.height * 32 + 250)

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" style="background-color: #1a1a24; font-family: monospace;">',
            f'<text x="20" y="30" fill="#4ea8de" font-size="18" font-weight="bold">Elysia Isometric Causal Engine (Multi-Scale & Intervention)</text>',
            f'<text x="20" y="50" fill="#a8b2d1" font-size="14">Scale Lens: {self.engine.get_scale_category()} (L={self.engine.scale_lens:.1f}) | Lens: {self.engine.lens_state.current_lens}</text>'
        ]

        terrain_colors = {"grass": "#4a7c59", "dirt_path": "#8d5b4c", "water": "#2b5c8f", "stone_pavement": "#707b7c"}
        building_colors = {"town_hall": "#e76f51", "house": "#f4a261", "granary": "#e9c46a", "farm_field": "#2a9d8f", "lumber_camp": "#8f5d34"}

        tiles_sorted = []
        for x in range(self.width):
            for y in range(self.height):
                tiles_sorted.append((x, y, x + y))
        tiles_sorted.sort(key=lambda item: item[2])

        for x, y, _ in tiles_sorted:
            cell = self.engine.tilemap.get_cell(x, y)
            iso_x, iso_y = self.engine.tilemap.projection.grid_to_iso(x, y, cell.elevation)

            pt_top = f"{iso_x},{iso_y - h2}"
            pt_right = f"{iso_x + w2},{iso_y}"
            pt_bottom = f"{iso_x},{iso_y + h2}"
            pt_left = f"{iso_x - w2},{iso_y}"

            fill_color = terrain_colors.get(cell.terrain_asset_id, "#4a7c59")
            stroke_color = "#2c3e50" if cell.causal_potential <= 2.0 else "#ff4d4d"

            svg_lines.append(f'<polygon points="{pt_top} {pt_right} {pt_bottom} {pt_left}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="0.8"/>')

            if cell.building_asset_id:
                b_color = building_colors.get(cell.building_asset_id, "#d35400")
                b_box_h = 24.0
                b_top = f"{iso_x},{iso_y - h2 - b_box_h}"
                b_right = f"{iso_x + w2*0.6},{iso_y - b_box_h}"
                b_bottom = f"{iso_x},{iso_y + h2*0.6 - b_box_h}"
                b_left = f"{iso_x - w2*0.6},{iso_y - b_box_h}"
                svg_lines.append(f'<polygon points="{b_top} {b_right} {b_bottom} {b_left}" fill="{b_color}" stroke="#ffffff" stroke-width="1.0"/>')
                svg_lines.append(f'<text x="{iso_x - 15}" y="{iso_y - b_box_h/2}" fill="#ffffff" font-size="10">{cell.building_asset_id[:4]}</text>')
            elif cell.resource_asset_id:
                svg_lines.append(f'<circle cx="{iso_x}" cy="{iso_y}" r="6" fill="#27ae60" stroke="#ffffff" stroke-width="0.8"/>')

        for npc_id, voxel in self.engine.npc_voxels.items():
            gx, gy = voxel.position[0], voxel.position[1]
            iso_x, iso_y = self.engine.tilemap.projection.grid_to_iso(gx, gy, 0.0)
            color = "#a855f7" if "anchor" in npc_id else "#f1c40f"
            svg_lines.append(f'<circle cx="{iso_x}" cy="{iso_y - 4}" r="5" fill="{color}" stroke="#000000" stroke-width="1.5"/>')
            svg_lines.append(f'<text x="{iso_x + 6}" y="{iso_y - 4}" fill="{color}" font-size="10">{npc_id}</text>')

        svg_lines.append('</svg>')
        svg_content = "\n".join(svg_lines)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(svg_content)

        return svg_content

    def export_html_interactive_viewer(self, filename: str = "isometric_causal_demo.html") -> str:
        """Generates a self-contained interactive HTML5 Canvas viewer."""
        history_json = json.dumps(self.telemetry_history)
        grid_width = self.width
        grid_height = self.height

        grid_data = []
        for x in range(self.width):
            row = []
            for y in range(self.height):
                cell = self.engine.tilemap.get_cell(x, y)
                row.append({
                    "x": x,
                    "y": y,
                    "elevation": cell.elevation,
                    "terrain": cell.terrain_asset_id,
                    "building": cell.building_asset_id,
                    "resource": cell.resource_asset_id,
                    "potential": float(cell.causal_potential)
                })
            grid_data.append(row)

        grid_json = json.dumps(grid_data)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Elysia Multi-Scale & Intervention Engine Viewer</title>
    <style>
        body {{ margin: 0; padding: 20px; background-color: #0d1117; color: #c9d1d9; font-family: sans-serif; }}
        .container {{ display: flex; flex-direction: row; gap: 20px; }}
        #canvas-container {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px; }}
        #hud {{ width: 380px; background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }}
        h2 {{ color: #58a6ff; margin-top: 0; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 14px; }}
        .badge-survival {{ background-color: #da3633; color: white; }}
        .badge-gathering {{ background-color: #d29922; color: white; }}
        .badge-construction {{ background-color: #238636; color: white; }}
        .badge-empire_expansion {{ background-color: #8957e5; color: white; }}
        .stat-box {{ background-color: #21262d; padding: 10px; margin-bottom: 10px; border-radius: 6px; }}
        button {{ background-color: #238636; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; margin-right: 5px; }}
        button:hover {{ background-color: #2ea043; }}
        .scale-slider {{ width: 100%; margin-top: 10px; }}
    </style>
</head>
<body>
    <h2>Elysia Causal Game Engine (Sims ↔ City ↔ Civ & Elysian Interventions)</h2>
    <div class="container">
        <div id="canvas-container">
            <canvas id="isoCanvas" width="800" height="550"></canvas>
            <div style="margin-top: 10px;">
                <button onclick="playSimulation()">Play Telemetry</button>
                <button onclick="toggleHeatmap()">Toggle Causal Heatmap</button>
            </div>
            <div style="margin-top: 15px;">
                <label><strong>Scale Lens ($\mathcal{{L}}_{{scale}}$):</strong> <span id="scale-val">0.5</span> (<span id="scale-cat">MID_CITY</span>)</label>
                <input type="range" id="scaleSlider" min="0" max="1" step="0.05" value="0.5" class="scale-slider" oninput="onScaleChange(this.value)">
            </div>
        </div>
        <div id="hud">
            <h3>Engine Telemetry HUD</h3>
            <div class="stat-box">
                <div>Step: <span id="hud-step">0</span> / {len(self.telemetry_history)}</div>
                <div>Scale Category: <span id="hud-scale-cat" style="color: #a5d6ff; font-weight: bold;">MID_CITY</span></div>
                <div>Cognitive Lens: <span id="hud-lens" class="badge badge-survival">Survival</span></div>
                <div>Field Tension ($\Omega$): <span id="hud-tension">0.0</span></div>
                <div>⚡ Elysia Energy: <span id="hud-energy">1000.0</span></div>
            </div>
            <h3>Resource Reserves</h3>
            <div class="stat-box">
                <div>🌾 Food: <span id="hud-food">100.0</span> | 🪵 Wood: <span id="hud-wood">100.0</span></div>
                <div>🪨 Stone: <span id="hud-stone">50.0</span> | 🪙 Gold: <span id="hud-gold">20.0</span></div>
            </div>
        </div>
    </div>

    <script>
        const gridData = {grid_json};
        const telemetry = {history_json};
        const tileWidth = 64, tileHeight = 32, elevScale = 16, originX = 400, originY = 80;

        let currentStepIdx = 0, showHeatmap = false, activeScaleLens = 0.5;
        const canvas = document.getElementById('isoCanvas'), ctx = canvas.getContext('2d');

        function gridToIso(gx, gy, elev) {{
            return {{ x: (gx - gy) * (tileWidth / 2) + originX, y: (gx + gy) * (tileHeight / 2) - (elev * elevScale) + originY }};
        }}

        function onScaleChange(val) {{
            activeScaleLens = parseFloat(val);
            document.getElementById('scale-val').innerText = activeScaleLens.toFixed(2);
            let cat = activeScaleLens < 0.3 ? 'MICRO_SIMS (Sims)' : (activeScaleLens > 0.7 ? 'MACRO_CIV (Civilization)' : 'MID_CITY');
            document.getElementById('scale-cat').innerText = cat;
            document.getElementById('hud-scale-cat').innerText = cat;
            draw();
        }}

        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const stepData = telemetry[currentStepIdx] || telemetry[0];

            for (let x = 0; x < {grid_width}; x++) {{
                for (let y = 0; y < {grid_height}; y++) {{
                    const cell = gridData[x][y];
                    const pos = gridToIso(x, y, cell.elevation);

                    ctx.beginPath();
                    ctx.moveTo(pos.x, pos.y - tileHeight / 2);
                    ctx.lineTo(pos.x + tileWidth / 2, pos.y);
                    ctx.lineTo(pos.x, pos.y + tileHeight / 2);
                    ctx.lineTo(pos.x - tileWidth / 2, pos.y);
                    ctx.closePath();

                    let fill = cell.terrain === 'water' ? '#2b5c8f' : (cell.terrain === 'dirt_path' ? '#8d5b4c' : '#4a7c59');
                    if (showHeatmap) fill = `rgba(${{Math.min(255, cell.potential * 20)}}, 50, 100, 0.8)`;

                    ctx.fillStyle = fill; ctx.fill();
                    ctx.strokeStyle = '#2c3e50'; ctx.stroke();

                    if (cell.building) {{
                        ctx.fillStyle = '#e76f51';
                        ctx.fillRect(pos.x - 12, pos.y - 24, 24, 20);
                        ctx.fillStyle = '#ffffff'; ctx.font = '10px monospace';
                        ctx.fillText(cell.building.substring(0, 4), pos.x - 10, pos.y - 10);
                    }}
                }}
            }}

            if (stepData && stepData.npc_positions) {{
                for (const npcId in stepData.npc_positions) {{
                    const posData = stepData.npc_positions[npcId];
                    const iso = gridToIso(posData.grid_pos[0], posData.grid_pos[1], 0);

                    ctx.beginPath();
                    ctx.arc(iso.x, iso.y - 4, 6, 0, Math.PI * 2);
                    ctx.fillStyle = npcId.includes('anchor') ? '#a855f7' : '#f1c40f';
                    ctx.fill(); ctx.strokeStyle = '#000000'; ctx.stroke();
                }}
            }}

            if (stepData) {{
                document.getElementById('hud-step').innerText = stepData.step;
                const lensElem = document.getElementById('hud-lens');
                lensElem.innerText = stepData.cognitive_lens;
                lensElem.className = 'badge badge-' + stepData.cognitive_lens.toLowerCase();

                document.getElementById('hud-tension').innerText = stepData.tension_level.toFixed(2);
                document.getElementById('hud-energy').innerText = stepData.elysian_causal_energy.toFixed(1);
                document.getElementById('hud-food').innerText = stepData.resource_reserves.food.toFixed(1);
                document.getElementById('hud-wood').innerText = stepData.resource_reserves.wood.toFixed(1);
                document.getElementById('hud-stone').innerText = stepData.resource_reserves.stone.toFixed(1);
                document.getElementById('hud-gold').innerText = stepData.resource_reserves.gold.toFixed(1);
            }}
        }}

        function playSimulation() {{
            let step = 0;
            const timer = setInterval(() => {{
                currentStepIdx = step;
                if (telemetry[step]) onScaleChange(telemetry[step].scale_lens);
                draw();
                step++;
                if (step >= telemetry.length) clearInterval(timer);
            }}, 300);
        }}

        function toggleHeatmap() {{ showHeatmap = !showHeatmap; draw(); }}
        draw();
    </script>
</body>
</html>
"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_content


if __name__ == "__main__":
    sim = IsometricCausalSimulator(width=12, height=12)
    print("Running 30-step Multi-Scale Isometric Causal Town Simulation with Elysian Interventions...")
    history = sim.run_simulation(steps=30)
    print(f"Simulation completed ({len(history)} steps).")

    svg_file = "isometric_town_render.svg"
    sim.export_svg(svg_file)
    print(f"Exported SVG rendering to: {svg_file}")

    html_file = "isometric_causal_demo.html"
    sim.export_html_interactive_viewer(html_file)
    print(f"Exported Interactive HTML Canvas Viewer to: {html_file}")
