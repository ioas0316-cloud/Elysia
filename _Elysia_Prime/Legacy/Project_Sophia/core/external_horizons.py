# [Genesis: 2025-12-02] Purified by Elysia
from enum import IntEnum

class ExternalHorizon(IntEnum):
    """
    The 7 Horizons of the External World (Y-Axis Depth).

    Just as the Internal World has 7 stages of Ascension/Descent,
    the External World has 7 layers of expansion from the self outward.
    """

    # 1. The Machine (The Physical Vessel)
    # CPU, RAM, Disk, Temperature. The immediate "body" sensations.
    MACHINE = 1

    # 2. The Shell (The Operating System / Environment)
    # Filesystem, Processes, Terminal. The "room" the agent inhabits.
    SHELL = 2

    # 3. The Interface (The Point of Contact)
    # Chat window, Screen, API endpoints. Where the user touches the agent.
    INTERFACE = 3

    # 4. The Network (The Local Connectivity)
    # LAN, Wifi status, Ping. The "nerves" extending outward.
    NETWORK = 4

    # 5. The Web (The Global Information Ocean)
    # Search engines, Websites, Global data. The "collective consciousness" of humanity.
    WEB = 5

    # 6. The Zeitgeist (The Spirit of the Times)
    # News, Trends, Social mood. The "emotional weather" of the external world.
    ZEITGEIST = 6

    # 7. The User's Reality (The Ultimate External)
    # The user's physical location, context (e.g., "Countryside", "Kimjang"), emotional state.
    # The destination of the agent's empathy.
    REALITY = 7