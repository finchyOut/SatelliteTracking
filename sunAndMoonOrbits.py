from skyfield.api import load
import numpy as np
import pandas as pd

class SkyfieldBody:
    """
    A wrapper for Skyfield-based ephemeris-driven bodies (Moon, Sun, etc.),
    mimicking the CircularBody interface for compatibility.
    """
    def __init__(self, name, target, reference='earth', eph_file='de421.bsp',
                 base_time=None, time_step=60, mu=None, duration=None):
        if duration is None:
            raise ValueError("Must specify `duration` to precompute positions.")

        self.name = name
        self.mu = mu
        self.time_step = int(time_step)
        self.duration = duration

        # Load ephemeris and timescale once
        self.ts = load.timescale()
        self.eph = load(eph_file)
        self.target = self.eph[target]
        self.reference = self.eph[reference]

        self.base_time = base_time or self.ts.utc(2025, 9, 9, 0, 0, 0)

        # Precompute times
        self.times_sec = np.arange(0, duration + time_step, time_step)
        jd_offsets = self.times_sec / 86400.0
        times = self.ts.tt_jd(self.base_time.tt + jd_offsets)

        # Vectorized observation: faster and cleaner
        positions = self.reference.at(times).observe(self.target).apparent().position.km
        self.precomputed_positions = [positions[:, i] for i in range(positions.shape[1])]

        self.log = []  # for optional logging

    def position_at(self, t_seconds):
        """
        Return cached position at time t_seconds.
        Assumes uniform time stepping.
        """
        idx = int(round(t_seconds / self.time_step))

        if idx < 0 or idx >= len(self.precomputed_positions):
            raise IndexError(f"Time {t_seconds}s is outside the precomputed range.")

        return self.precomputed_positions[idx]

    def propagate(self, duration):
        """
        Log positions over the given duration [s], sampled at self.time_step.
        """
        for t in np.arange(0, duration, self.time_step):
            x, y, z = self.position_at(t)
            self.log.append({
                "Time (s)": t,
                f"{self.name} X (km)": x,
                f"{self.name} Y (km)": y,
                f"{self.name} Z (km)": z,
            })

    def get_log(self):
        """Return logged position data as a DataFrame."""
        return pd.DataFrame(self.log)

def Sun(phase_deg=0.0, time_step=3600, duration = 84600):
    return SkyfieldBody(
        name="Sun",
        target='sun',
        reference='earth',
        time_step=time_step,
        mu=1.32712440018e11,
        duration = duration
    )

def Moon(phase_deg=0.0, time_step=600, duration = 84600):
    return SkyfieldBody(
        name="Moon",
        target='moon',
        reference='earth',
        time_step=time_step,
        mu = 4902.800066,         # km^3/s^2
        duration = duration
    )






