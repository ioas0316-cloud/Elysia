import pygame
import json
import time
import numpy as np
from typing import Dict, Optional, Tuple, List
import os
import sys
import argparse

# Add project root to sys.path to allow imports from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.Foundation.core.world import World
from Core.Foundation.core.world_event_logger import WorldEventLogger
from Core.Foundation.wave_mechanics import WaveMechanics # Mocked, but import needed for world creation
from tools.kg_manager import KGManager # Mocked, but import needed for world creation

# --- Constants and Configuration ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
CELL_RADIUS = 5
GRID_CELL_SIZE = 20
BACKGROUND_COLOR = (10, 10, 20)
TERRAIN_COLOR_LOW = (20, 40, 60)
TERRAIN_COLOR_HIGH = (80, 120, 160)
FONT_COLOR = (200, 200, 220)
CULTURE_COLORS = {
    "neutral": (150, 150, 150),
    "wuxia": (200, 100, 100),
    "knight": (100, 100, 200),
}
INJURED_COLOR = (255, 0, 0) # Red color for injury indicator

# --- Animation Classes ---

class Animation:
    """Base class for all animations."""
    def __init__(self, duration):
        self.duration = duration
        self.elapsed_time = 0.0

    def update(self, dt) -> bool:
        """Update the animation state. Return True if finished."""
        self.elapsed_time += dt
        return self.elapsed_time >= self.duration

    def draw(self, screen, pos):
        """Draw the animation. To be implemented by subclasses."""
        pass

class Lunge(Animation):
    """An animation for a cell lunging towards a target and returning."""
    def __init__(self, start_pos, end_pos, duration=0.5):
        super().__init__(duration)
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)

    def get_current_pos(self) -> np.ndarray:
        """Calculates the current position based on the lunge-and-return path."""
        progress = min(1.0, self.elapsed_time / self.duration)
        # Go to target and back
        if progress < 0.5:
            p = progress * 2
        else:
            p = (1.0 - progress) * 2
        return self.start_pos + (self.end_pos - self.start_pos) * p

class FadeOut(Animation):
    """An animation for a cell fading away upon death."""
    def __init__(self, duration=1.0):
        super().__init__(duration)

    def get_alpha(self) -> int:
        """Calculates the current alpha value for the fade effect."""
        progress = min(1.0, self.elapsed_time / self.duration)
        return int(255 * (1.0 - progress))

class Flash(Animation):
    """A screen-wide flash animation, used for events like lightning."""
    def __init__(self, duration=0.3, color=(255, 255, 220)):
        super().__init__(duration)
        self.color = color

    def draw(self, screen):
        """Draws the flash effect overlay."""
        progress = min(1.0, self.elapsed_time / self.duration)
        # Flash is brightest at the start and fades
        alpha = int(255 * (1.0 - progress))

        flash_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        flash_surface.fill((self.color[0], self.color[1], self.color[2], alpha))
        screen.blit(flash_surface, (0, 0))

# --- Helper Functions ---
def world_to_screen_pos(world_pos: np.ndarray, world_grid_size: int) -> Tuple[int, int]:
    """Converts world coordinates to screen pixel coordinates."""
    x = int((world_pos[0] / world_grid_size) * SCREEN_WIDTH)
    y = int((world_pos[1] / world_grid_size) * SCREEN_HEIGHT)
    return x, y

def draw_terrain(screen, world: World):
    """Draws the world's height map as a shaded background."""
    for x in range(world.grid_size):
        for y in range(world.grid_size):
            height_normalized = world.height_map[x, y] / 10.0 # Assuming max height is 10
            color = tuple(int(TERRAIN_COLOR_LOW[i] + (TERRAIN_COLOR_HIGH[i] - TERRAIN_COLOR_LOW[i]) * height_normalized) for i in range(3))
            rect = pygame.Rect(x * GRID_CELL_SIZE, y * GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

# --- Main Visualization Function ---

def visualize_world_events(world: World):
    """Initializes Pygame and runs the main visualization loop."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Elysia's World - Event Visualizer")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    # --- Event and Animation Management ---
    event_log_path = world.event_logger.log_file_path
    last_log_pos = 0
    animations: Dict[str, Animation] = {}  # cell_id -> Animation object
    dying_cells: Dict[str, float] = {}  # cell_id -> death_timestamp
    lightning_flash: Optional[Flash] = None

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Read New World Events from Log File ---
        try:
            with open(event_log_path, 'r') as f:
                f.seek(last_log_pos)
                new_lines = f.readlines()
                last_log_pos = f.tell()

            for line in new_lines:
                event_data = json.loads(line)
                event_type = event_data.get('type')
                cell_id = event_data.get('cell_id')

                if event_type == 'EAT':
                    if 'target_id' in event_data and cell_id in world.id_to_idx and event_data['target_id'] in world.id_to_idx:
                        attacker_idx = world.id_to_idx[cell_id]
                        target_idx = world.id_to_idx[event_data['target_id']]
                        start_pos = world.positions[attacker_idx]
                        end_pos = world.positions[target_idx]
                        animations[cell_id] = Lunge(start_pos, end_pos)

                elif event_type == 'DEATH':
                    if cell_id:
                        animations[cell_id] = FadeOut()
                        dying_cells[cell_id] = time.time()

                elif event_type == 'LIGHTNING_STRIKE':
                    lightning_flash = Flash()

        except FileNotFoundError:
            pass # Log file might not be created yet

        # --- Update Animations ---
        finished_anims = []
        for cell_id, anim in animations.items():
            if anim.update(dt):
                finished_anims.append(cell_id)
        for cell_id in finished_anims:
            del animations[cell_id]

        if lightning_flash and lightning_flash.update(dt):
            lightning_flash = None

        # --- Drawing ---
        screen.fill(BACKGROUND_COLOR)
        draw_terrain(screen, world)

        # Draw cells
        for cell_id, idx in world.id_to_idx.items():
            # Skip drawing cells that have a death animation running
            if cell_id in dying_cells and time.time() - dying_cells[cell_id] > 1.0:
                 continue

            pos = world.positions[idx]
            screen_pos = world_to_screen_pos(pos, world.grid_size)
            culture = world.culture[idx] if world.culture[idx] in CULTURE_COLORS else "neutral"
            color = CULTURE_COLORS[culture]

            current_pos = screen_pos
            alpha = 255

            if cell_id in animations:
                anim = animations[cell_id]
                if isinstance(anim, Lunge):
                    world_anim_pos = anim.get_current_pos()
                    current_pos = world_to_screen_pos(world_anim_pos, world.grid_size)
                elif isinstance(anim, FadeOut):
                    alpha = anim.get_alpha()
                    color = (min(255, color[0] + alpha), min(255, color[1] + alpha), min(255, color[2] + alpha))


            # Use a surface for alpha transparency
            cell_surface = pygame.Surface((CELL_RADIUS * 2, CELL_RADIUS * 2), pygame.SRCALPHA)
            pygame.draw.circle(cell_surface, (*color, alpha), (CELL_RADIUS, CELL_RADIUS), CELL_RADIUS)

            # Injury indicator
            if world.is_injured[idx]:
                pygame.draw.circle(cell_surface, (*INJURED_COLOR, alpha), (CELL_RADIUS, CELL_RADIUS), CELL_RADIUS + 2, 2)


            screen.blit(cell_surface, (current_pos[0] - CELL_RADIUS, current_pos[1] - CELL_RADIUS))


        # Draw lightning flash on top of everything
        if lightning_flash:
            lightning_flash.draw(screen)

        # Draw info text
        info_text = f"Time: {world.time_step} | Cells: {len(world.id_to_idx)}"
        text_surface = font.render(info_text, True, FONT_COLOR)
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animate Elysia's World events.")
    parser.add_argument("--log_file", type=str, default="logs/world_events.jsonl", help="Path to the world event log file.")
    args = parser.parse_args()

    # --- World Initialization (for state reading) ---
    # We need a world instance to know where to draw cells, but it won't be run.
    mock_kg = MagicMock(spec=KGManager)
    mock_kg.get_node.return_value = {}
    mock_wm = MagicMock(spec=WaveMechanics)
    mock_wm.kg_manager = mock_kg

    logger = WorldEventLogger(log_file_path=args.log_file)

    world_instance = World(
        primordial_dna={'instinct': 'survive'},
        wave_mechanics=mock_wm,
        logger=logger,
        grid_size=50 # Should match the simulation being logged
    )
    # This is just a dummy world for visualization. The actual simulation runs separately.

    print("Starting animated event visualizer...")
    print(f"Reading events from: {os.path.abspath(args.log_file)}")
    visualize_world_events(world_instance)
    print("Visualizer closed.")
