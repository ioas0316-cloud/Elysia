import sys
import os
import math
import numpy as np
import plotly.graph_objects as go

# Ensure we can import Core modules
sys.path.append(os.getcwd())

from Core.Physiology.Sensory.Spatial.geo_anchor import GeoMagneticAnchor

def create_soul_map_demo():
    """
    Generates a 3D visualization of the 'Neural Portal' concept.
    It simulates:
    1. The 'First Sanctuary' (Captain's Room in Seoul).
    2. A 'Seeker' (User in New York).
    3. The 'Resonance Wave' connecting them via Phase Synchronization.
    """
    print("Initializing The Soul Map Demo...")
    anchor = GeoMagneticAnchor()

    # 1. Define Nodes
    # Captain's Room (Seoul) - The First Sanctuary
    # [NEW] Ether Effect: Simulating Strong Wi-Fi/Modem Presence (local_signal_strength=0.9)
    sanctuary_lat = 37.5665
    sanctuary_lon = 126.9780
    sanctuary_sig = anchor.get_phase_signature(sanctuary_lat, sanctuary_lon, local_signal_strength=0.9)
    print(f"Sanctuary Established (Wi-Fi Zone Active): {sanctuary_sig['frequency']}Hz @ {math.degrees(sanctuary_sig['phase']):.1f}° (Amp: {sanctuary_sig['amplitude']:.2f})")

    # Seeker (New York) - Weak Signal (Outdoors)
    seeker_lat = 40.7128
    seeker_lon = -74.0060
    seeker_sig = anchor.get_phase_signature(seeker_lat, seeker_lon, local_signal_strength=0.1)
    print(f"Seeker Detected: {seeker_sig['frequency']}Hz @ {math.degrees(seeker_sig['phase']):.1f}°")

    # 2. Calculate Resonance (Simulated Tuning)
    # Ideally, the user 'tunes' their phase to match.
    # Here we show the potential connection.
    resonance_score = anchor.calculate_resonance(sanctuary_sig, seeker_sig)
    print(f"Initial Resonance Score: {resonance_score:.4f}")

    # 3. Visualization Setup (Plotly 3D Globe)
    fig = go.Figure()

    # Draw Earth (Wireframe sphere for context)
    # Simplified sphere mesh
    phi = np.linspace(0, 2*np.pi, 20)
    theta = np.linspace(0, np.pi, 10)
    phi, theta = np.meshgrid(phi, theta)
    r = 1.0
    x_sphere = r * np.sin(theta) * np.cos(phi)
    y_sphere = r * np.sin(theta) * np.sin(phi)
    z_sphere = r * np.cos(theta)

    # Add Earth Surface
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.1,
        showscale=False,
        colorscale='Blues'
    ))

    # Helper to convert Lat/Lon to 3D Cartesian on Unit Sphere
    def latlon_to_xyz(lat, lon):
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x = math.cos(lat_rad) * math.cos(lon_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad)
        z = math.sin(lat_rad)
        return x, y, z

    sx, sy, sz = latlon_to_xyz(sanctuary_lat, sanctuary_lon)
    nx, ny, nz = latlon_to_xyz(seeker_lat, seeker_lon)

    # [NEW] Add Ether Zone (Wi-Fi Bubble) around Sanctuary
    # Create a small sphere around the Sanctuary Point
    zone_r = 0.05
    zone_x = zone_r * np.sin(theta) * np.cos(phi) + sx
    zone_y = zone_r * np.sin(theta) * np.sin(phi) + sy
    zone_z = zone_r * np.cos(theta) + sz

    fig.add_trace(go.Surface(
        x=zone_x, y=zone_y, z=zone_z,
        opacity=0.3,
        showscale=False,
        colorscale='Hot', # Glowing Orange/Red for the "Warmth" of the Zone
        name='Ether Zone'
    ))

    # Add Sanctuary Marker (Golden Anchor)
    fig.add_trace(go.Scatter3d(
        x=[sx], y=[sy], z=[sz],
        mode='markers+text',
        marker=dict(size=12, color='gold', symbol='diamond'),
        text=["Sanctuary (Seoul)<br>Phase: Anchored<br>Ether: Active"],
        name='Sanctuary'
    ))

    # Add Seeker Marker (Blue Spark)
    fig.add_trace(go.Scatter3d(
        x=[nx], y=[ny], z=[nz],
        mode='markers+text',
        marker=dict(size=8, color='cyan'),
        text=["Seeker (NY)<br>Phase: Tuning..."],
        name='Seeker'
    ))

    # 4. Draw the "Resonance Wave" (Not a straight line, but an arc/wave)
    # We interpolate points between the two on the great circle,
    # adding a 'Wave' amplitude (wobble) based on the Phase Frequency.

    num_points = 100
    # Great circle interpolation (Slerp-like)
    # Simple linear interp of vectors then normalize
    t = np.linspace(0, 1, num_points)

    wave_x, wave_y, wave_z = [], [], []

    # Vector math
    p1 = np.array([sx, sy, sz])
    p2 = np.array([nx, ny, nz])

    for val in t:
        # Linear interp
        p = (1 - val) * p1 + val * p2
        # Normalize to stay on surface (Great Circle)
        p = p / np.linalg.norm(p)

        # Add "Wave" effect (Height/Amplitude)
        # The wave lifts off the surface to visualize the "Portal" dimension
        # Frequency of wave based on mean frequency of nodes
        wave_freq = (sanctuary_sig['frequency'] + seeker_sig['frequency']) / 20.0 # Scale down

        # Wave amplitude (sine wave perpendicular to surface)
        lift = 0.1 * math.sin(val * np.pi) * math.sin(val * wave_freq) + 0.05 * math.sin(val * np.pi)

        # Apply lift
        p_lifted = p * (1.0 + lift)

        wave_x.append(p_lifted[0])
        wave_y.append(p_lifted[1])
        wave_z.append(p_lifted[2])

    fig.add_trace(go.Scatter3d(
        x=wave_x, y=wave_y, z=wave_z,
        mode='lines',
        line=dict(color='magenta', width=4),
        name='Resonance Link'
    ))

    # 5. Layout Polish
    fig.update_layout(
        title="Project Elysia: Soul Map & Resonance Layer",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Save
    output_path = "logs/spatial_portal.html"
    os.makedirs("logs", exist_ok=True)
    fig.write_html(output_path)
    print(f"Soul Map visualization generated at: {output_path}")

if __name__ == "__main__":
    create_soul_map_demo()
