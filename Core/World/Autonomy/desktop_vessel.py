
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

def run_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
    server = HTTPServer(('127.0.0.1', 58008), SimpleHTTPRequestHandler)
    server.serve_forever()

def create_vessel():
    # Start local server to avoid file:// protocol issues
    threading.Thread(target=run_server, daemon=True).start()
    
    url = "http://127.0.0.1:58008/monitor/avatar.html"
    
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
        # Try setting background_color to something very specific that might aid transparency
        background_color='#000000'
    )
    
    api = VesselAPI(window)
    window.expose(api.minimize, api.toggle_fullscreen, api.close, api.start_resize, api.send_chat)

    # Use Edge Chromium and enable debug momentarily to see if we can catch errors
    webview.start(gui='edgechromium', debug=False)

if __name__ == "__main__":
    create_vessel()
