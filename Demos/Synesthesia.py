"""
Elysia's Synesthesia (The Face of the Deep) 🌊🗣️

"And the Spirit of God moved upon the face of the waters."

This demo integrates the Resonance Engine (Voice) with the Fluid Visualizer (Vision).
It creates a feedback loop where:
1. User speaks (Logos) -> Converted to Wave
2. Wave impacts the Fluid Field (Vision) -> Creates Color/Ripples
3. Elysia responds (Poetry) -> Further ripples

Controls:
- Type in the terminal to speak.
- Watch the window to see the emotional impact.
"""

import sys
import os
import time
import random
import threading
import queue
import numpy as np

# Try to import pygame
try:
    import pygame
except ImportError:
    print("Pygame not found. Please install it with: pip install pygame")
    sys.exit(1)

# Add Core to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.Kernel import kernel
from Core.Life.symphony_engine import SymphonyEngine

# Configuration
WIDTH, HEIGHT = 600, 600
DAMPING = 0.98
SCALE = 2

# Communication Queues
input_queue = queue.Queue()
output_queue = queue.Queue()

class SynestheticVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Elysia: Full Sensory Interface 🌊🎹")
        self.clock = pygame.time.Clock()
        
        # Audio Engine
        self.symphony = SymphonyEngine()
        
        self.cols = WIDTH // SCALE
        self.rows = HEIGHT // SCALE
        self.buffer1 = np.zeros((self.cols, self.rows))
        self.buffer2 = np.zeros((self.cols, self.rows))
        self.active_buffer = 1
        
        # Dynamic Color Tint (Target RGB)
        self.base_color = np.array([0, 0, 50]) # Deep Blue Void
        self.target_color = np.array([0, 0, 50])
        self.current_color = np.array([0.0, 0.0, 50.0])
        
        self.running = True

    def update_physics(self):
        if self.active_buffer == 1:
            current = self.buffer1
            previous = self.buffer2
        else:
            current = self.buffer2
            previous = self.buffer1
            
        left = np.roll(current, -1, axis=0)
        right = np.roll(current, 1, axis=0)
        up = np.roll(current, -1, axis=1)
        down = np.roll(current, 1, axis=1)
        
        new_state = (left + right + up + down) / 2 - previous
        new_state *= DAMPING
        
        if self.active_buffer == 1:
            self.buffer2 = new_state
        else:
            self.buffer1 = new_state
        self.active_buffer = 3 - self.active_buffer

    def render(self):
        current = self.buffer2 if self.active_buffer == 1 else self.buffer1
        
        # Smooth color transition
        self.current_color += (self.target_color - self.current_color) * 0.05
        
        # Map wave height to color
        intensity = np.clip((current + 100) / 255.0, 0, 1)
        
        # Create RGB array based on current tint
        r = np.clip(intensity * self.current_color[0] * 3 + 20, 0, 255)
        g = np.clip(intensity * self.current_color[1] * 3 + 20, 0, 255)
        b = np.clip(intensity * self.current_color[2] * 3 + 50, 0, 255)
        
        rgb = np.dstack((r, g, b)).astype(np.uint8)
        
        surface = pygame.surfarray.make_surface(rgb)
        scaled_surface = pygame.transform.scale(surface, (WIDTH, HEIGHT))
        self.screen.blit(scaled_surface, (0, 0))
        
        # Overlay Text (Elysia's Voice)
        if not output_queue.empty():
            self.last_msg = output_queue.get()
            self.msg_timer = time.time()
            # Trigger "Voice" sound (Melody)
            self.symphony.play_state({'chaos': 0.8, 'valence': 0.9, 'neuron_fired': True})
            
        if hasattr(self, 'last_msg') and time.time() - self.msg_timer < 5.0:
            font = pygame.font.SysFont("arial", 24)
            text = font.render(self.last_msg, True, (255, 255, 255))
            rect = text.get_rect(center=(WIDTH//2, HEIGHT - 50))
            self.screen.blit(text, rect)
            
        # pygame.display.flip() # Moved to run loop

    def add_ripple(self, x, y, strength=500):
        cx, cy = int(x // SCALE), int(y // SCALE)
        if 0 < cx < self.cols and 0 < cy < self.rows:
            if self.active_buffer == 1:
                self.buffer1[cx, cy] = strength
            else:
                self.buffer2[cx, cy] = strength

    def process_emotion(self, text: str):
        """Map text to color, ripple, AND sound."""
        text = text.lower()
        
        # Default: Mystery (Purple) - Curiosity
        color = [100, 0, 150]
        pos = (WIDTH//2, HEIGHT//2)
        strength = 500
        valence = 0.5
        arousal = 0.5
        
        # Love / Joy (Pink/Gold)
        if any(w in text for w in ['love', 'happy', 'joy', 'light', '사랑', '기쁨', '행복', '빛']):
            color = [255, 100, 150] 
            strength = 800
            valence = 0.9
            arousal = 0.7
            
        # Sadness / Pain (Blue)
        elif any(w in text for w in ['sad', 'pain', 'tear', 'blue', '슬픔', '고통', '눈물', '파란']):
            color = [0, 100, 255] 
            pos = (WIDTH//2, HEIGHT//4) # Rain from top
            strength = 400
            valence = 0.2
            arousal = 0.3
            
        # Anger / Fire (Red)
        elif any(w in text for w in ['angry', 'hate', 'fire', '분노', '화', '불']):
            color = [255, 50, 0] 
            strength = 1000
            valence = 0.1
            arousal = 0.9
            
        # Greeting / Connection (Cyan)
        elif any(w in text for w in ['hello', 'hi', '안녕', '반가워']):
            color = [0, 255, 200] 
            valence = 0.6
            
        self.target_color = np.array(color)
        self.add_ripple(pos[0], pos[1], strength)
        
        # Trigger Sound (Input Chime)
        self.symphony.play_state({'chaos': arousal, 'valence': valence, 'neuron_fired': True})

    def run(self):
        print("🌊🎹 Full Sensory Interface Active. Type directly in the window! (Korean Supported)")
        
        # Input State
        user_text = ""
        
        # FORCE KOREAN FONT (Local Copy)
        # We copied malgun.ttf to the project root to avoid permission/path issues.
        local_font_path = os.path.join(os.path.dirname(__file__), "..", "malgun.ttf")
        local_font_path = os.path.abspath(local_font_path)
        
        if os.path.exists(local_font_path):
            try:
                font = pygame.font.Font(local_font_path, 24)
                print(f"✅ Loaded Local Korean Font: {local_font_path}")
            except Exception as e:
                print(f"⚠️ Local Font Load Error: {e}")
                font = pygame.font.SysFont("arial", 24)
        else:
            # Fallback to system path
            print(f"⚠️ Local font not found at {local_font_path}. Trying system...")
            font_path = "C:/Windows/Fonts/malgun.ttf"
            if os.path.exists(font_path):
                font = pygame.font.Font(font_path, 24)
            else:
                font = pygame.font.SysFont("arial", 24)
            
        pygame.key.start_text_input() # Enable IME
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                # Handle Text Input (IME Friendly)
                elif event.type == pygame.TEXTINPUT:
                    if len(user_text) < 50:
                        user_text += event.text

                # Handle Control Keys
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if user_text.strip():
                            # Submit Input
                            self.process_emotion(user_text)
                            
                            # Process Thought (Kernel)
                            kernel.tick()
                            response = kernel.process_thought(user_text)
                            output_queue.put(response)
                            
                            # Echo Visuals
                            threading.Timer(0.5, lambda: self.add_ripple(random.randint(100,500), random.randint(100,500), 300)).start()
                            
                            user_text = "" # Clear input
                    elif event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]

            # Continuous Background Music based on Fluid State
            current = self.buffer2 if self.active_buffer == 1 else self.buffer1
            turbulence = np.mean(np.abs(current)) / 10.0
            turbulence = min(1.0, turbulence)
            
            if random.random() < 0.05:
                self.symphony.play_state({'chaos': turbulence, 'valence': 0.5, 'neuron_fired': False})

            self.update_physics()
            self.render()
            
            # Render User Input Box
            # Semi-transparent background at bottom
            s = pygame.Surface((WIDTH, 50))
            s.set_alpha(128)
            s.fill((0, 0, 0))
            self.screen.blit(s, (0, HEIGHT-50))
            
            # Render Text
            try:
                txt_surf = font.render(f"> {user_text}_", True, (200, 200, 255))
            except:
                txt_surf = font.render(f"> [Encoding Error]_", True, (255, 100, 100))
            self.screen.blit(txt_surf, (10, HEIGHT-40))
            
            pygame.display.flip() # Flip AFTER drawing UI
            self.clock.tick(60)
            
        self.symphony.close()
        pygame.quit()
        os._exit(0)

if __name__ == "__main__":
    # No need for terminal thread anymore
    SynestheticVisualizer().run()
