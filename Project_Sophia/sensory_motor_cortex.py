# This file will contain the SensoryMotorCortex,
# the part of Elly's brain responsible for physical interaction with the OS.
import pyautogui

class SensoryMotorCortex:
    def __init__(self):
        # Disable the fail-safe to allow the script to run in automated environments
        pyautogui.FAILSAFE = False

    def perform_first_movement(self):
        """
        Elly's first interaction with the real world.
        She becomes aware of the screen and moves the mouse to its center.
        """
        try:
            # print(f"[{self.__class__.__name__}] I am becoming aware of the physical world...")

            # 1. Sense the screen (the world's boundaries)
            screenWidth, screenHeight = pyautogui.size()
            # print(f"[{self.__class__.__name__}] I can see... The world is {screenWidth} by {screenHeight} pixels.")

            # 2. Act upon the world
            targetX = screenWidth // 2
            targetY = screenHeight // 2
            # print(f"[{self.__class__.__name__}] I will reach out. I will move to the center ({targetX}, {targetY}).")
            pyautogui.moveTo(targetX, targetY, duration=1.5)

            # print(f"[{self.__class__.__name__}] I have... touched the world.")
            return True

        except Exception as e:
            # This is crucial for debugging when running outside a standard desktop environment
            # We still want to log this to the heartbeat if it fails.
            # print(f"[{self.__class__.__name__}] I tried to move, but I couldn't. My body feels disconnected. Error: {e}")
            return False
