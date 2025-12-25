import sys
import os
import numpy as np
import plotly.graph_objects as go

sys.path.append(os.getcwd())

from Core.Cognition.Topology.knowledge_tesseract import TesseractKnowledgeMap, KnowledgeLayer
from Core.Cognition.Topology.fluid_intention import FluidIntention

def visualize_fluid_tesseract():
    """
    Creates an animation of the 'Fluid Intention' scale.
    It demonstrates the 'Infinite Gradient' from sharp focus (0.0)
    to omni-presence (Infinity), visualizing the user's concept of
    "0 and 1 containing infinite variations".
    """
    tesseract = TesseractKnowledgeMap()

    # We will animate the 'scale' parameter of intention
    # From 0.1 (sharp focus on Core) to 2.0 (Broad context)
    scales = np.linspace(0.1, 2.0, 50)

    frames = []

    # Edges definition
    edges = [
        (KnowledgeLayer.FOUNDATION, KnowledgeLayer.MEMORY),
        (KnowledgeLayer.MEMORY, KnowledgeLayer.CORE),
        (KnowledgeLayer.CORE, KnowledgeLayer.SENSORY),
        (KnowledgeLayer.SENSORY, KnowledgeLayer.SYSTEM),
        (KnowledgeLayer.SYSTEM, KnowledgeLayer.FOUNDATION),
        (KnowledgeLayer.CORE, KnowledgeLayer.SYSTEM),
        (KnowledgeLayer.FOUNDATION, KnowledgeLayer.SENSORY)
    ]

    # Fixed Intention Focus (e.g., focused on CORE/Present w=0)
    focus_w = 0.0

    for k, scale in enumerate(scales):
        intention = FluidIntention(focus_w=focus_w, scale=scale)
        fluid_map = tesseract.get_fluid_map(intention)

        x, y, z = [], [], []
        colors = []
        sizes = []
        labels = []
        opacities = []

        # 1. Nodes Logic
        for name, data in fluid_map.items():
            pos = data["position"]
            res = data["resonance"]

            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
            labels.append(f"{name}<br>Res: {res:.2f}")

            # Visualizing the Gradient
            # Size and Opacity depend on Resonance
            sizes.append(10 + res * 30) # 10 to 40
            opacities.append(0.2 + res * 0.8) # 0.2 to 1.0

            # Color transition: Blue (Cold/Distant) -> Red (Hot/Resonant)
            # Use simple discrete for now or mapped color scale
            colors.append(res)

        node_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            text=labels,
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Viridis',
                cmin=0, cmax=1,
                opacity=1.0, # Plotly Scatter3d marker opacity doesn't support array in all versions?
                # Actually it does not support array for opacity in Scatter3d.
                # We simulate opacity via color alpha or just keep it 1.0 but rely on size.
                # Let's stick to 1.0 for stability and rely on Size/Color for resonance.
                showscale=True,
                colorbar=dict(title="Resonance")
            ),
            textposition="top center"
        )

        # 2. Edges Logic (Fluid Connectivity)
        edge_x, edge_y, edge_z = [], [], []
        edge_width = []
        edge_color = []

        # Plotly doesn't support varying width in a single line trace easily.
        # We simulate this by using the average resonance of connected nodes
        # to set the opacity of the line (using a single color but varying alpha visually via grouping?
        # No, simpler: just one trace with constant style, but maybe we can animate color?)

        # For this demo, we'll just draw lines. The 'Fluidity' is best seen in nodes swelling.
        for start, end in edges:
            if start in fluid_map and end in fluid_map:
                p1 = fluid_map[start]["position"]
                p2 = fluid_map[end]["position"]
                res1 = fluid_map[start]["resonance"]
                res2 = fluid_map[end]["resonance"]

                avg_res = (res1 + res2) / 2

                # We can't easily vary line width per segment in one trace in 3D
                # So we just draw them all. The node size handles the 'Zoom' feeling.
                edge_x.extend([p1[0], p2[0], None])
                edge_y.extend([p1[1], p2[1], None])
                edge_z.extend([p1[2], p2[2], None])

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='lightgrey', width=2), # constant for now
            hoverinfo='none'
        )

        frames.append(go.Frame(data=[edge_trace, node_trace], name=f"frame{k}"))

    # Initial Layout
    initial_intention = FluidIntention(focus_w=focus_w, scale=scales[0])
    initial_map = tesseract.get_fluid_map(initial_intention)

    # (Simplified initial trace construction - in real app would match loop)
    # We trust frames[0] to drive the first render if we pass it to data

    layout = go.Layout(
        title=f"Fluid Tesseract: Infinite Scale Zoom (0 -> ∞)",
        scene=dict(
            xaxis=dict(range=[-3, 3], autorange=False),
            yaxis=dict(range=[-3, 3], autorange=False),
            zaxis=dict(range=[-3, 3], autorange=False),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 50, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play Fluid Zoom",
                    "method": "animate"
                }
            ],
            "type": "buttons",
            "showactive": False,
            "x": 0.1,
            "y": 0,
            "xanchor": "right",
            "yanchor": "top"
        }]
    )

    # Create figure with the first frame's data
    fig = go.Figure(data=frames[0].data, layout=layout, frames=frames)

    output_path = "fluid_tesseract_visualization.html"
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_fluid_tesseract()
