"""
City Visualizer (The Dashboard)
===============================

Renders the "Cyber-City" map of Elysia's mind.
Nodes are districts, edges are roads, and particles are Pulse Packets.

Runs a Plotly Dash server.
"""

import sys
import os
import json
import time
import dash
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import networkx as nx

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.Scripts.city_map import DISTRICTS, ROADS, get_node_position
from Core.Scripts.hyper_resonator import get_resonator

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/city_traffic.json")

# Initialize Graph
G = nx.Graph()
for name, info in DISTRICTS.items():
    G.add_node(name, **info)
for src, dst, weight in ROADS:
    G.add_edge(src, dst, weight=weight)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Elysia: City of Logic (Pulse Dashboard)", style={'textAlign': 'center', 'color': '#00ffcc'}),

    html.Div([
        html.Div(id='stats-panel', style={'color': 'white', 'padding': '20px', 'textAlign': 'center'}),

        # Hyper-Resonator Controls
        html.Div([
            html.Label("Hyper-Resonator Controls:", style={'color': '#aaa'}),
            html.Br(),
            html.Button("▲ Pitch Up", id='btn-pitch-up', n_clicks=0, style={'margin': '5px'}),
            html.Button("▶ Yaw Right", id='btn-yaw-right', n_clicks=0, style={'margin': '5px'}),
            html.Button("❄️ COLLAPSE", id='btn-collapse', n_clicks=0, style={'margin': '5px', 'color': 'cyan'}),
            html.Button("🔥 RESURRECT", id='btn-resurrect', n_clicks=0, style={'margin': '5px', 'color': 'orange'}),
        ], style={'textAlign': 'center', 'marginBottom': '10px'}),

        dcc.Graph(id='city-graph', style={'height': '75vh'}),
    ]),

    dcc.Interval(
        id='interval-component',
        interval=1000, # 1 sec update
        n_intervals=0
    )
], id='main-layout', style={'backgroundColor': '#111111', 'transition': 'background-color 0.5s ease'})

# Callback for Resonator Control
@app.callback(
    Output('stats-panel', 'style'),
    [Input('btn-pitch-up', 'n_clicks'), Input('btn-yaw-right', 'n_clicks'),
     Input('btn-collapse', 'n_clicks'), Input('btn-resurrect', 'n_clicks')]
)
def manual_control(b1, b2, b3, b4):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    resonator = get_resonator()

    if button_id == "btn-pitch-up": resonator.rotate('x', 0.2)
    elif button_id == "btn-yaw-right": resonator.rotate('y', 0.2)
    elif button_id == "btn-collapse": resonator.collapse()
    elif button_id == "btn-resurrect": resonator.resurrect()
    return dash.no_update

@app.callback(
    [Output('city-graph', 'figure'), Output('stats-panel', 'children'), Output('main-layout', 'style')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    # 1. Read Traffic & Resonator State
    traffic = {"load": {}, "total": 0}
    resonator = get_resonator()
    observation = resonator.observe()
    bg_color = observation['color']

    try:
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r") as f:
                traffic = json.load(f)
    except Exception:
        pass

    # 2. Prepare 3D Plot
    edge_x, edge_y, edge_z = [], [], []
    edge_colors = []

    # Calculate Traffic per Edge
    # Heuristic: If both source and target are busy, the road is busy
    node_load = traffic.get("load", {})

    for u, v in G.edges():
        x0, y0, z0 = get_node_position(u)
        x1, y1, z1 = get_node_position(v)
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

        # Traffic Color Logic
        load_u = node_load.get(u, 0)
        load_v = node_load.get(v, 0)
        total_load = load_u + load_v

        if total_load > 10: color = "red"      # Congestion
        elif total_load > 5: color = "yellow"  # Moderate
        else: color = "#888"                   # Clear
        edge_colors.append(color)

    # Nodes
    node_x, node_y, node_z = [], [], []
    node_color = []
    node_size = []
    node_text = []

    for node in G.nodes():
        x, y, z = get_node_position(node)
        node_x.append(x); node_y.append(y); node_z.append(z)

        info = DISTRICTS[node]
        node_color.append(info["color"])

        # Size grows with load (Breathing City)
        load = node_load.get(node, 0)
        size = 20 + (load * 2)
        node_size.append(size)

        node_text.append(f"{info['label']}<br>Load: {load}")

    # Create Traces
    trace_edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='#888', width=2), # Base lines
        hoverinfo='none'
    )

    trace_nodes = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(symbol='circle', size=node_size, color=node_color, opacity=0.9),
        text=[DISTRICTS[n]["label"] for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo='text'
    )

    # Layout (Apply Resonator Theme)
    title_text = f"Resonator: {observation['dominance']}"
    if observation.get("is_particle"):
        title_text = "❄️ STATE: COLLAPSED MEMORY ORB (PARTICLE)"

    layout = go.Layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor=bg_color),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor=bg_color),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor=bg_color),
            bgcolor=bg_color
        ),
        paper_bgcolor=bg_color,
        margin=dict(t=0, b=0, l=0, r=0),
        title=dict(text=title_text, font=dict(color="#ffffff"))
    )

    fig = go.Figure(data=[trace_edges, trace_nodes], layout=layout)

    # Stats Panel
    dominant_mood = "Void"
    if observation.get("is_particle"):
        dominant_mood = "Memory Orb (Frozen)"
    elif observation.get('dominance'):
        dominant_mood = max(observation['dominance'], key=observation['dominance'].get)

    stats_html = [
        html.H3(f"Total Pulses: {traffic.get('total', 0)}"),
        html.P("Active Districts: " + ", ".join([f"{k}:{v}" for k,v in node_load.items() if v > 0])),
        html.H4(f"💎 State: {dominant_mood}", style={'color': bg_color, 'filter': 'brightness(200%)'})
    ]

    layout_style = {'backgroundColor': bg_color, 'transition': 'background-color 0.5s ease', 'minHeight': '100vh'}

    return fig, stats_html, layout_style

if __name__ == '__main__':
    print("🏙️ Launching City of Logic Dashboard on http://localhost:8050")
    app.run(debug=True, port=8050)
