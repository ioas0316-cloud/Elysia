
import webview
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

class VesselAPI:
    def __init__(self, window):
        self.window = window

    def minimize(self):
        self.window.minimize()

    def toggle_fullscreen(self):
        self.window.toggle_fullscreen()

    def close(self):
        self.window.destroy()

    def start_resize(self, direction):
        # direction: 'TOP', 'BOTTOM', 'LEFT', 'RIGHT', 'TOP_LEFT', etc.
        try:
            self.window.start_resize(direction.upper())
        except Exception as e:
            print(f"Resize error: {e}")

    def send_chat(self, text):
        """Writes user input to a signal file for the Heartbeat to consume."""
        signal_path = os.path.abspath("data/State/USER_INPUT.txt")
        try:
            with open(signal_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Chat received: {text}")
        except Exception as e:
            print(f"Error sending chat: {e}")

def create_vessel():
    # Phase 16-B: Align with VisualizerServer (Port 8000)
    # No need to start a separate local server here anymore.
    
    url = "http://127.0.0.1:8000/avatar"
    
    window = webview.create_window(
        'Elysia Vessel',
        url=url,
        transparent=True,
        easy_drag=True,
        on_top=True,
        width=800,
        height=1000,
        frameless=True,
        resizable=True,
        min_size=(300, 300),
        background_color='#000000'
    )
    
    api = VesselAPI(window)
    window.expose(api.minimize, api.toggle_fullscreen, api.close, api.start_resize, api.send_chat)

    webview.start(gui='edgechromium', debug=False)

if __name__ == "__main__":
    create_vessel()
