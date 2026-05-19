"""
[BODY] Action Observer: Motor Impulse Logger (Mirror Neurons)
============================================================
Location: Scripts/System/Senses/action_observer.py

Role:
- Listen to User Inputs (Keyboard/Mouse) as 'Teacher Signals'.
- Log Timestamp + Key to correlate with Visual Cortex.
- Enables 'Behavioral Cloning' (Monkey See, Monkey Do).

Privacy Note:
- Does NOT log text typing (passwords).
- Only logs 'Game Control' keys (WASD, Space, Mouse).
"""

import time
import json
import threading
from pynput import keyboard, mouse
import os

class ActionObserver:
    def __init__(self, log_path="c:\\Elysia\\data\\motor_log.json"):
        self.log_path = log_path
        self.active_keys = set()
        self.history = []
        self.is_recording = False
        
        # Ensure dir exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _on_press(self, key):
        try:
            k = key.char # CHAR keys
        except AttributeError:
            k = str(key) # SPECIAL keys
            
        # Filter for Game Keys (WASD, Space, Shift, Mouse) for privacy
        if str(k).lower() in ['w', 'a', 's', 'd', 'q', 'e', 'r', 'f', 'Key.space', 'Key.shift', 'Key.esc']:
            if k not in self.active_keys:
                self.active_keys.add(k)
                self._log_event("PRESS", k)

    def _on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)
            
        if k in self.active_keys:
            self.active_keys.remove(k)
            self._log_event("RELEASE", k)

    def _on_click(self, x, y, button, pressed):
        if pressed:
            self._log_event("CLICK", str(button))

    def _log_event(self, action_type, key):
        event = {
            "time": time.time(),
            "action": action_type,
            "key": str(key)
        }
        self.history.append(event)
        # print(f"🎮 [INPUT] {action_type} {key}")

    def start_observing(self):
        self.is_recording = True
        print("🎮 [BODY] Mirror Neurons Active. Watching Teacher's Hands...")
        
        # Non-blocking listeners
        self.k_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.m_listener = mouse.Listener(on_click=self._on_click)
        
        self.k_listener.start()
        self.m_listener.start()

    def stop_observing(self):
        if self.is_recording:
            self.k_listener.stop()
            self.m_listener.stop()
            self.is_recording = False
            self._save_log()
            print(f"🎮 [BODY] Observation Ended. Learned {len(self.history)} motor impulses.")

    def _save_log(self):
        try:
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving motor log: {e}")

if __name__ == "__main__":
    observer = ActionObserver()
    print("🧪 [TEST] Press WASD or Space. Recording for 5 seconds...")
    observer.start_observing()
    time.sleep(5)
    observer.stop_observing()
