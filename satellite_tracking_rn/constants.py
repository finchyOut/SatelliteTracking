# constants.py

import numpy as np

# --- Earth Parameters in Kilometers ---
# Original MU_EARTH: 3.986004418e14 m^3/s^2
# Conversion factor: (1 km / 1000 m)^3 = 1e-9
MU_EARTH_KM = 3.986004418e5  # Earth's gravitational parameter (km^3/s^2)
R_EARTH_KM = 6378.137       # Earth's equatorial radius (km)
J2 = 1.08263e-3             # J2 perturbation constant (unitless)

# --- Third-Body Parameters in Kilometers ---
# Original MU_SUN: 1.32712440018e20 m^3/s^2
# Conversion factor: 1e-9
MU_SUN_KM = 1.32712440018e11 # Sun's gravitational parameter (km^3/s^2)
# Original MU_MOON: 4.9048695e12 m^3/s^2
# Conversion factor: 1e-9
MU_MOON_KM = 4.9048695e3     # Moon's gravitational parameter (km^3/s^2)

# --- Solar Radiation Pressure Constant ---
# P_R has units of N/m^2. Acceleration (a) from SRP is F/m = (P_R * A) / m.
# The units of 'a' are m/s^2. To get 'a' in km/s^2, we must divide by 1000.
# So, the constant P_R must also be adjusted.
# P_R_KM = (4.56e-6) / 1000 = 4.56e-9
P_R_KM_COEFF = 4.56e-9 # A constant that gives acceleration in km/s^2

# --- Default Scenario: Geostationary Orbit in Kilometers ---
GEO_ALTITUDE_KM = 35786.0
GEO_RADIUS_KM = R_EARTH_KM + GEO_ALTITUDE_KM
GEO_SPEED_KM = np.sqrt(MU_EARTH_KM / GEO_RADIUS_KM)
S0_KM = np.array([GEO_RADIUS_KM, 0, 0, 0, GEO_SPEED_KM, 0])