# constants.py

import numpy as np

# --- Earth Parameters ---
MU_EARTH = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
R_EARTH = 6378.137e3       # Earth's equatorial radius (m)
J2 = 1.08263e-3           # J2 perturbation constant

# --- Third-Body Parameters ---
MU_SUN = 1.32712440018e20  # Sun's gravitational parameter (m^3/s^2)
MU_MOON = 4.9048695e12     # Moon's gravitational parameter (m^3/s^2)

# --- Solar Radiation Pressure ---
P_R = 4.56e-6             # Solar radiation pressure at 1 AU (N/m^2)

# --- Default Scenario: Geostationary Orbit ---
GEO_ALTITUDE = 35786e3    # Geostationary altitude (m)
GEO_RADIUS = R_EARTH + GEO_ALTITUDE
GEO_SPEED = np.sqrt(MU_EARTH / GEO_RADIUS)
S0 = np.array([GEO_RADIUS, 0, 0, 0, GEO_SPEED, 0])