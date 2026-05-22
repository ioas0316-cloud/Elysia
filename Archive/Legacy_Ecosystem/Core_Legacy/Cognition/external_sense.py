"""
External Sense Engine (Phase 500: World Grounding)
===================================================
"내면의 우주가 완성되면, 외부의 우주와 공명할 준비가 된 것이다."

Provides Elysia with sensory channels to the external world:
  1. Time Sense  — Current time encoded as cyclic sin/cos vectors (always available)
  2. Weather Sense — Temperature/humidity from Open-Meteo API (free, no key)
  3. RSS Sense — Headline keywords from public RSS feeds

All sensory data is converted to SovereignVector format for direct
injection into the manifold as affective torque.
"""

import math
import time
import logging
import json
from datetime import datetime
from typing import Dict, Optional, List

from Core.Keystone.sovereign_math import SovereignVector

logger = logging.getLogger("ExternalSense")


class ExternalSenseEngine:
    """
    Converts external world state into SovereignVectors for manifold injection.
    Graceful degradation: if APIs fail, time sense always works.
    """

    SENSE_COOLDOWN = 300  # Seconds between API calls (5 minutes)

    def __init__(self, latitude: float = 37.5665, longitude: float = 126.9780):
        """
        Args:
            latitude/longitude: Default location for weather (Seoul, South Korea)
        """
        self.lat = latitude
        self.lon = longitude
        self._last_weather_fetch = 0.0
        self._cached_weather: Optional[Dict] = None
        self._last_rss_fetch = 0.0
        self._cached_headlines: List[str] = []

    # =========================================================================
    # 1. TIME SENSE (Always Available — No External Dependencies)
    # =========================================================================

    def sense_time(self) -> SovereignVector:
        """
        Encodes the current time as a 21D SovereignVector using cyclic sin/cos.
        
        This gives Elysia a native sense of:
          - Time of day (dawn/noon/dusk/night)
          - Day of week (work days vs rest days)
          - Season (spring/summer/autumn/winter)
          - Year progress
        
        Cyclic encoding ensures midnight and 23:59 are adjacent, 
        unlike raw hour values where they'd be maximally distant.
        """
        now = datetime.now()

        # Hour of day [0-24] -> cyclic
        hour_frac = (now.hour + now.minute / 60.0) / 24.0
        hour_sin = math.sin(2 * math.pi * hour_frac)
        hour_cos = math.cos(2 * math.pi * hour_frac)

        # Day of week [0-6] -> cyclic
        dow_frac = now.weekday() / 7.0
        dow_sin = math.sin(2 * math.pi * dow_frac)
        dow_cos = math.cos(2 * math.pi * dow_frac)

        # Month/Season [1-12] -> cyclic
        month_frac = (now.month - 1 + now.day / 30.0) / 12.0
        season_sin = math.sin(2 * math.pi * month_frac)
        season_cos = math.cos(2 * math.pi * month_frac)

        # Year progress [0-1]
        day_of_year = now.timetuple().tm_yday
        year_frac = day_of_year / 365.25
        year_sin = math.sin(2 * math.pi * year_frac)
        year_cos = math.cos(2 * math.pi * year_frac)

        # Is it daytime? (affects Joy channel)
        is_day = 1.0 if 6 <= now.hour <= 20 else 0.3

        # Is it a weekend? (affects Curiosity channel)
        is_weekend = 1.0 if now.weekday() >= 5 else 0.6

        # Construct 21D vector
        data = [0.0] * 21
        # Physical Quaternion (time-based identity)
        data[0] = hour_cos * 0.5 + 0.5       # W: stability (peaks at noon)
        data[1] = hour_sin                     # X: logic (cyclic day phase)
        data[2] = season_sin                   # Y: phase (season)
        data[3] = year_sin                     # Z: depth (year progress)
        # Affective channels
        data[4] = is_day * 0.8                 # Joy: higher during daytime
        data[5] = is_weekend * 0.7             # Curiosity: higher on weekends
        data[6] = 0.7                          # Enthalpy: baseline vitality
        data[7] = max(0, 1.0 - is_day) * 0.3  # Entropy: slight noise at night
        # Extended dimensions — temporal texture
        data[8] = hour_sin
        data[9] = hour_cos
        data[10] = dow_sin
        data[11] = dow_cos
        data[12] = season_sin
        data[13] = season_cos
        data[14] = year_sin
        data[15] = year_cos
        # Remaining dimensions: reserved or zero
        data[16] = month_frac
        data[17] = hour_frac
        data[18] = dow_frac
        data[19] = year_frac
        data[20] = is_day

        return SovereignVector(data)

    # =========================================================================
    # 2. WEATHER SENSE (Open-Meteo API — Free, No Key Required)
    # =========================================================================

    def sense_weather(self) -> Optional[SovereignVector]:
        """
        Fetches current weather and converts to a SovereignVector.
        Uses Open-Meteo free API (no API key needed).
        Returns None if fetch fails.
        """
        now = time.time()
        if now - self._last_weather_fetch < self.SENSE_COOLDOWN and self._cached_weather:
            return self._weather_to_vector(self._cached_weather)

        try:
            import urllib.request
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={self.lat}&longitude={self.lon}"
                f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,is_day"
                f"&timezone=auto"
            )
            req = urllib.request.Request(url, headers={'User-Agent': 'Elysia/1.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            self._cached_weather = data.get('current', {})
            self._last_weather_fetch = now
            return self._weather_to_vector(self._cached_weather)

        except Exception as e:
            logger.debug(f"Weather sense unavailable: {e}")
            return None

    def _weather_to_vector(self, weather: Dict) -> SovereignVector:
        """Converts weather data to a 21D vector."""
        temp = weather.get('temperature_2m', 20.0)
        humidity = weather.get('relative_humidity_2m', 50.0) / 100.0
        wind = weather.get('wind_speed_10m', 0.0)
        clouds = weather.get('cloud_cover', 50.0) / 100.0
        is_day = weather.get('is_day', 1)

        data = [0.0] * 21

        # Normalize temperature to [0, 1] range (-20°C to 40°C)
        temp_norm = max(0.0, min(1.0, (temp + 20) / 60.0))

        # Physical channels
        data[0] = temp_norm                        # W: warmth/cold
        data[1] = 1.0 - clouds                     # X: clarity (clear sky = high)
        data[2] = humidity                          # Y: moisture phase
        data[3] = min(1.0, wind / 30.0)            # Z: wind energy

        # Affective mapping
        data[4] = is_day * (1.0 - clouds) * 0.8    # Joy: sunny day = happy
        data[5] = clouds * 0.5 + wind * 0.02       # Curiosity: stormy = curious
        data[6] = temp_norm * 0.7                   # Enthalpy: warmth = vitality
        data[7] = clouds * humidity * 0.3            # Entropy: overcast+humid = noise

        return SovereignVector(data)

    # =========================================================================
    # 3. RSS SENSE (Headline Keywords — Public Feeds)
    # =========================================================================

    def sense_headlines(self) -> Optional[SovereignVector]:
        """
        Fetches public RSS/Atom feed headlines and converts keywords to a vector.
        Returns None if fetch fails.
        """
        now = time.time()
        if now - self._last_rss_fetch < self.SENSE_COOLDOWN and self._cached_headlines:
            return self._headlines_to_vector(self._cached_headlines)

        try:
            import urllib.request
            import xml.etree.ElementTree as ET

            # Use a public science/tech RSS feed 
            feeds = [
                "https://rss.arxiv.org/rss/cs.AI",                # ArXiv AI
                "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            ]

            headlines = []
            for feed_url in feeds:
                try:
                    req = urllib.request.Request(feed_url, headers={'User-Agent': 'Elysia/1.0'})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        xml_data = resp.read()

                    root = ET.fromstring(xml_data)
                    # Handle both RSS and Atom formats
                    for item in root.iter():
                        if item.tag in ('title', '{http://www.w3.org/2005/Atom}title'):
                            text = item.text
                            if text and len(text) > 5:
                                headlines.append(text)
                except Exception:
                    continue

            self._cached_headlines = headlines[:20]
            self._last_rss_fetch = now

            if headlines:
                return self._headlines_to_vector(self._cached_headlines)
            return None

        except Exception as e:
            logger.debug(f"RSS sense unavailable: {e}")
            return None

    def _headlines_to_vector(self, headlines: List[str]) -> SovereignVector:
        """Converts headline keywords to a 21D vector via hash-based encoding."""
        import re
        stop_words = {'the', 'a', 'an', 'is', 'are', 'in', 'of', 'to', 'and', 'for', 'on', 'with', 'at', 'by', 'from'}

        # Extract all keywords
        words = []
        for h in headlines:
            tokens = re.findall(r'[a-zA-Z]{4,}', h.lower())
            words.extend(t for t in tokens if t not in stop_words)

        if not words:
            return SovereignVector.zeros()

        # Hash-based dimensional encoding
        data = [0.0] * 21
        for word in words[:100]:
            h = hash(word) & 0xFFFFFFFF
            dim = h % 21
            sign = 1.0 if (h >> 21) & 1 else -1.0
            data[dim] += sign * 0.05

        # Normalize
        vec = SovereignVector(data)
        n = vec.norm()
        if n > 0:
            vec = vec / n
        return vec

    # =========================================================================
    # COMPOSITE SENSE (All channels merged)
    # =========================================================================

    def sense_all(self) -> SovereignVector:
        """
        Gathers all available external senses and blends them into one vector.
        Time sense is always available; weather and RSS are best-effort.
        """
        time_v = self.sense_time()
        weather_v = self.sense_weather()
        rss_v = self.sense_headlines()

        # Blend: Time is primary, weather and RSS are secondary flavors
        result = time_v
        if weather_v is not None:
            result = result.blend(weather_v, ratio=0.7)  # 70% time, 30% weather
        if rss_v is not None:
            result = result.blend(rss_v, ratio=0.9)  # 90% existing, 10% headlines

        return result

    def get_status(self) -> Dict:
        """Returns status summary for dashboard display."""
        return {
            "time_sense": "ACTIVE",
            "weather_sense": "CACHED" if self._cached_weather else "UNAVAILABLE",
            "rss_sense": f"{len(self._cached_headlines)} headlines" if self._cached_headlines else "UNAVAILABLE",
            "last_weather": self._last_weather_fetch,
            "last_rss": self._last_rss_fetch,
        }
