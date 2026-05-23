"""
Elysia Core Actuation Utility Library
=====================================
Bridges internal quaternion rotation changes into hardware actions (WASD and Space keypresses).
"""

import pyautogui

# Enable failsafe: throwing mouse cursor to any corner of the screen aborts script
pyautogui.FAILSAFE = True

class Actuator:
    def __init__(self):
        self.last_action = "None"

    def execute_rotation_action(self, rotation_diff) -> str:
        """
        Maps quaternion rotation difference to keyboard action.
        """
        axis = rotation_diff.axis
        angle = rotation_diff.angle

        if abs(angle) < 0.1:
            self.last_action = "Idle"
            return "Idle"

        action = "None"
        try:
            # X-axis dominant (Yaw / Left-Right)
            if abs(axis[0]) > abs(axis[1]) and abs(axis[0]) > abs(axis[2]):
                if axis[0] > 0:
                    pyautogui.press('d')
                    action = "D (Right)"
                else:
                    pyautogui.press('a')
                    action = "A (Left)"
            
            # Y-axis dominant (Pitch / Forward-Jump)
            elif abs(axis[1]) > abs(axis[0]) and abs(axis[1]) > abs(axis[2]):
                if axis[1] > 0:
                    pyautogui.press('w')
                    action = "W (Forward)"
                else:
                    pyautogui.press('space')
                    action = "Space (Jump/Evade)"
            
            # Z-axis dominant (Roll / Backward)
            else:
                pyautogui.press('s')
                action = "S (Backward)"

            self.last_action = action
            return action

        except pyautogui.FailSafeException:
            print("🚨 [Failsafe] Mouse cursor moved to corner. Aborting actuation!")
            raise
