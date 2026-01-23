import math
import cmath

class GeoMagneticAnchor:
    """
    Simulates a 'Soul Anchor' based on physical Geo-Magnetic Coordinates.

    It converts Latitude/Longitude into a 4D Quaternion (SoulTensor) representing
    the Magnetic Flux and Phase at that specific location on Earth.

    Physics Model (Simplified):
    - Treats Earth as a simple magnetic dipole.
    - Magnetic North is approx at 86.5 N, 162.8 E (Dynamic, but fixed for this version).
    """

    # Earth's approximate magnetic dipole parameters
    MAGNETIC_NORTH_LAT = 86.5
    MAGNETIC_NORTH_LON = 162.8
    EARTH_RADIUS_KM = 6371.0

    def __init__(self):
        pass

    def calculate_magnetic_flux(self, lat, lon):
        """
        Calculates the theoretical Magnetic Flux Vector (Bx, By, Bz) for a given location.

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.

        Returns:
            dict: {'x': Bx, 'y': By, 'z': Bz, 'intensity': F}
        """
        # Convert degrees to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        # Calculate geomagnetic latitude (simplified approximation)
        # In a real IGRF model, this is complex. Here we use the distance to Magnetic North pole.
        # Co-latitude of the point
        colat = math.radians(90 - lat)

        # Simplified Dipole Field equations
        # B_r = -2 * B0 * cos(theta)
        # B_theta = -B0 * sin(theta)
        # where theta is co-latitude relative to magnetic axis.

        # For this prototype, we'll create a unique vector based on spherical position
        # to serve as a deterministic "ID".

        # X: North Component, Y: East Component, Z: Down Component
        # We model the variation: Field is strongest at poles (2x) vs equator.

        # Dipole intensity factor (approximate)
        B0 = 31000.0 # nanoTesla (nT) at equator

        # Calculate angle to magnetic north
        d_lon = math.radians(self.MAGNETIC_NORTH_LON - lon)

        # Spherical Law of Cosines to find distance angle (theta)
        # sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(d_lon)
        mn_lat_rad = math.radians(self.MAGNETIC_NORTH_LAT)

        cos_theta = (math.sin(lat_rad) * math.sin(mn_lat_rad) +
                     math.cos(lat_rad) * math.cos(mn_lat_rad) * math.cos(d_lon))

        # Clamp value for safety
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)

        # Field strength modulation
        # B ~ sqrt(1 + 3*cos^2(theta_m))
        intensity_factor = math.sqrt(1 + 3 * (math.cos(theta)**2))
        intensity = B0 * intensity_factor

        # Vector components (Projecting the 3D field vector)
        # This is a "Signature", so uniqueness matters more than perfect physics here.
        bx = intensity * math.cos(theta) # North-ish
        by = intensity * math.sin(theta) * math.sin(d_lon) # East-ish distortion
        bz = intensity * math.sin(lat_rad) # Vertical component (dip)

        return {
            'x': bx,
            'y': by,
            'z': bz,
            'intensity': intensity
        }

    def get_phase_signature(self, lat, lon, local_signal_strength=0.0):
        """
        Converts the physical magnetic flux into a metaphysical 'Phase Signature' (WaveTensor).

        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            local_signal_strength (float): 0.0 to 1.0 (Simulates Wi-Fi/Modem strength).
                                           Higher strength boosts amplitude and stabilizes resonance.

        Returns:
            dict: {
                'frequency': float (Base resonant frequency in Hz),
                'phase': float (Phase offset in radians 0-2pi),
                'amplitude': float (Signal strength),
                'vector': tuple (x, y, z)
            }
        """
        flux = self.calculate_magnetic_flux(lat, lon)

        # Frequency: Mapped from Total Intensity (25000nT ~ 65000nT)
        # We map Earth's field to an audible/brainwave range (e.g., 7.83Hz Schumann * Scale)
        # Let's map 20k-70k nT to 432Hz +/- 50Hz range for "Musicality"
        base_freq = 432.0
        norm_intensity = (flux['intensity'] - 20000) / 50000.0 # 0.0 to 1.0 approx
        frequency = base_freq * (0.8 + (norm_intensity * 0.4)) # 345Hz to 518Hz

        # Phase: Determined by the field vector orientation (ArcTan2)
        # This creates a unique "Angle" for every location.
        phase = math.atan2(flux['y'], flux['x']) # -pi to pi

        # Ether Effect: Local signal (Wi-Fi) boosts the "Presence" (Amplitude)
        # It creates a "Zone" where the signal is strong.
        base_amplitude = flux['intensity'] / 65000.0
        boosted_amplitude = base_amplitude * (1.0 + local_signal_strength)

        return {
            'frequency': round(frequency, 2),
            'phase': phase,
            'amplitude': min(1.0, boosted_amplitude), # Normalized amplitude, clamped at 1.0
            'vector': (flux['x'], flux['y'], flux['z'])
        }

    def calculate_resonance(self, sig_a, sig_b):
        """
        Calculates the 'Resonance Factor' (Interference) between two signatures.

        Returns:
            float: 0.0 (Total Dissonance) to 1.0 (Total Resonance/Superposition)
        """
        # 1. Frequency Harmony (Are they in the same 'key'?)
        freq_diff = abs(sig_a['frequency'] - sig_b['frequency'])
        # Use a Gaussian bell curve: closer frequencies = higher resonance
        freq_harmony = math.exp(-(freq_diff**2) / (2 * (50.0**2))) # Tolerance of ~50Hz

        # 2. Phase Alignment (Constructive Interference)
        # Constructive if phase diff is 0 or 2pi
        phase_diff = abs(sig_a['phase'] - sig_b['phase'])
        # Alignment factor: cos(diff) -> 1 at 0, -1 at pi.
        # We want 0 to 1 scale: (cos(diff) + 1) / 2
        phase_alignment = (math.cos(phase_diff) + 1) / 2

        # Total Resonance
        resonance = freq_harmony * phase_alignment

        return resonance