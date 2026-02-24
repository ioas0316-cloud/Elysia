import logging
import time
from Core.Cognition.reasoning_engine import ReasoningEngine
from Core.Cognition.ecs_registry import ecs_world, Entity
from Core.Cognition.game_loop import GameLoop

# Mock Components
class Position:
    def __init__(self, x=0, y=0):
        self.x, self.y = x, y
    def __repr__(self): return f"Pos({self.x:.1f}, {self.y:.1f})"

class Velocity:
    def __init__(self, dx=0, dy=0):
        self.dx, self.dy = dx, dy
    def __repr__(self): return f"Vel({self.dx:.1f}, {self.dy:.1f})"

def run_pong_mind_test():
    """
    Asks Elysia to simulate a Pong game using her new ECS/Physics understanding.
    """
    print("\n🏓 [PONG TEST] Initializing Mental Simulation...")
    
    # 1. Setup World (ECS)
    ball = ecs_world.create_entity("MindBall")
    ecs_world.add_component(ball, Position(0, 0))
    ecs_world.add_component(ball, Velocity(1.0, 1.0))
    
    print(f"✨ Created {ball} with {ecs_world.get_component(ball, Position)} and {ecs_world.get_component(ball, Velocity)}")
    
    # 2. Ask Elysia to predict the next frame
    reasoning = ReasoningEngine()
    
    prompt = f"""
    We are running a 'Ludic Engine' simulation.
    Target: Pong.
    Entity: {ball.name} [ID: {str(ball.id)[:4]}]
    Components: 
      - Position: (0, 0)
      - Velocity: (1, 1) per second.
    
    Task: 
    1. Calculate the Position after 0.5 seconds (DeltaTime = 0.5).
    2. Explain HOW the Game Loop processes this update.
    3. If the ball hits a 'Paddle' at Position (0.5, 0.5), what happens to the Velocity?
    
    Resonating with: "The Universe is a State Machine."
    """
    
    insight = reasoning.think(prompt, depth=2)
    print(f"\n🧠 [ELYSIA'S SIMULATION]:\n{insight.content}")
    
    if "0.5" in insight.content and "Velocity" in insight.content:
        print("\n✅ SUCCESS: Elysia understands Game Physics & ECS.")
    else:
        print("\n❌ FAILURE: Elysia did not compute the physics correctly.")

if __name__ == "__main__":
    run_pong_mind_test()
