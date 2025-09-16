
from scipy.linalg import solve_continuous_are
import numpy as np
import pandas as pd
# This creates the satellite Bang Bang Controller Class

class SatelliteBB:
    # Satellite class simulates orbital propagation with optional stationkeeping
    EARTH_RADIUS = 6371  # km
    # Earth's radius in kilometers (used to set orbital altitude)
    MU = 398600.4418     # km^3/s^2
    # Earth's gravitational parameter in km^3/s^2

    def __init__(self, sat_id, longitude_deg,
        # Initialize satellite with orbital parameters and simulation settings
                 eccentricity=0, noise_std=0.01, max_drift_km=100,
                 time_step=60, bodies=None, a_max = 1E-3):
        self.bodies = bodies or []
        self.sat_id = sat_id

        self.a = self.EARTH_RADIUS + 35786
    # Earth's radius in kilometers (used to set orbital altitude)
        self.longitude = np.radians(longitude_deg)
        # Longitude converted to radians (for initial position)
        self.ecc = eccentricity
        # Orbital eccentricity (0 for circular orbit)
        self.noise_std = noise_std
        # Standard deviation of Gaussian noise for velocity perturbation
        self.max_drift = max_drift_km
        # Maximum drift distance from ideal position before stationkeeping triggers
        self.time_step = time_step
        # Time increment for simulation (in seconds)
        self.omega = np.sqrt(self.MU / self.a**3)  # rad/sec
        # Earth's gravitational parameter in km^3/s^2

        # Initialize position and velocity (circular, equatorial)
        self.x = self.a * np.cos(self.longitude)
        # Longitude converted to radians (for initial position)
        self.y = self.a * np.sin(self.longitude)
        # Longitude converted to radians (for initial position)
        self.z = 0
        # Initial z position (equatorial orbit = 0)
        self.vx = -self.orbital_velocity() * np.sin(self.longitude)
        # Longitude converted to radians (for initial position)
        self.vy = self.orbital_velocity() * np.cos(self.longitude)
        # Longitude converted to radians (for initial position)
        self.vz = 0
        #Include a_max
        self.a_max = a_max
        # Initial z velocity (equatorial orbit = 0)
        self.log = []

        # --- LQR Setup ---
        A = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [np.zeros((3, 3)), np.zeros((3, 3))]
        ])
        B = np.block([
            [np.zeros((3, 3))],
            [np.eye(3)]
        ])
        Q = np.diag([1e-3, 1e-3, 1e-3, 1.0, 1.0, 1.0])  # Penalize velocity more than position
        R = np.eye(3) * 100000  # Penalize large acceleration

        # Solve the continuous-time Riccati equation
        P = solve_continuous_are(A, B, Q, R)

        # Compute the optimal gain matrix K
        self.K_lqr = np.linalg.inv(R) @ B.T @ P

    def lqr_control(self, current_state, desired_state):
        """
        LQR feedback controller.
        current_state: np.array([x, y, z, vx, vy, vz])
        desired_state: np.array([x*, y*, z*, vx*, vy*, vz*])
        Returns: control acceleration (ax, ay, az)
        """
        error_state = current_state - desired_state
        control = -self.K_lqr @ error_state
        return control  # [ax, ay, az]


    def third_body_acceleration(self, t_seconds):
        """
    Sum third-body accelerations from any bodies (e.g., Sun, Moon) in a geocentric frame.
    a_3b = Σ μ_b * [ (r_b - r)/|r_b - r|^3  -  r_b/|r_b|^3 ]
    where r is satellite position, r_b is body position, μ_b is body's GM.
        """
        if not self.bodies:
            return np.zeros(3)

        r = np.array([self.x, self.y, self.z], dtype=float)
        a = np.zeros(3, dtype=float)
        for body in self.bodies:
            r_b = body.position_at(t_seconds)
            delta = r_b - r
            a += body.mu * (delta / np.linalg.norm(delta)**3 - r_b / np.linalg.norm(r_b)**3)
        return a

    def orbital_velocity(self):
        # Compute ideal orbital velocity for circular orbit (used for control)
        return np.sqrt(self.MU / self.a)
    # Earth's gravitational parameter in km^3/s^2

    def add_third_body_columns(df, bodies):
        # Expect a "Time (s)" column in df
        times = df["Time (s)"].to_numpy()

        for body in bodies:
            # Vectorize position_at over all times
            xyz = np.array([body.position_at(t) for t in times])  # shape (N, 3)
            df[f"{body.name} X (km)"] = xyz[:, 0]
            df[f"{body.name} Y (km)"] = xyz[:, 1]
            df[f"{body.name} Z (km)"] = xyz[:, 2]
        return df


    def propagate(self, duration):
        dt = float(self.time_step)
        # Main simulation loop: propagates position and velocity over time
        #for t in range(0, duration, self.time_step):
        for t in np.arange(0, duration, self.time_step):
        # Time increment for simulation (in seconds)
            angle = self.longitude + self.omega * t
        # Longitude converted to radians (for initial position)
            ideal_x = self.a * np.cos(angle)
            ideal_y = self.a * np.sin(angle)
            ideal_z = 0

            dx = self.x - ideal_x
            dy = self.y - ideal_y
            dz = self.z - ideal_z
            drift_vec = np.array([dx, dy, dz])
            drift_distance = np.linalg.norm(drift_vec)

            if drift_distance > self.max_drift:
                status = "stationkeeping"

                # Fixed max thrust (in km/s^2) — adjust to your system
                a_max = self.a_max

                # Normalize direction
                acc_direction = -drift_vec / drift_distance
                ax, ay, az = a_max * acc_direction
            else:
                status = "non-stationkeeping"
                ax = ay = az = 0

            #added code to account for the effects of j2, solar radiation, and central gravity
            a_grav = self.central_gravity()
            a_j2 = self.j2_perturbation()
            a_solar = self.solar_radiation_pressure(t)
            a_3b    = self.third_body_acceleration(t)

            a_total = np.array([ax, ay, az], float) + a_grav + a_j2 + a_solar + a_3b

            # Update velocity based on new acceleration - v_{k+1} = v_k + a_total * dt
            self.vx += a_total[0] * dt
            self.vy += a_total[1] * dt
            self.vz += a_total[2] * dt

            # Update position based on previous velocity
            self.x += self.vx * self.time_step
            self.y += self.vy * self.time_step
            self.z += self.vz * self.time_step

            # Add noise to velocity instead of position
            noise = np.random.normal(0, self.noise_std, 3)

            # Standard deviation of Gaussian noise for velocity perturbation
            self.vx += noise[0]
            self.vy += noise[1]
            self.vz += noise[2]

        # Time increment for simulation (in seconds)
            angle = self.longitude + self.omega * (t + self.time_step)
        # Longitude converted to radians (for initial position)
            ideal_x = self.a * np.cos(angle)
            ideal_y = self.a * np.sin(angle)
            ideal_z = 0

            self.log.append({
                "Time (s)": t,
                "Satellite ID": self.sat_id,
                "Longitude (deg)": np.degrees(self.longitude),
        # Longitude converted to radians (for initial position)
                "X (km)": self.x,
                "Y (km)": self.y,
                "Z (km)": self.z,
                "VX": self.vx,
                "VY": self.vy,
                "VZ": self.vz,
                "ideal x": ideal_x,
                "ideal_y": ideal_y,
                "ideal_z": ideal_z,
                "Status": status,
                "Acceleration(x)": ax,
                "Acceleration(y)": ay,
                "Acceleration(z)": az,
                "a_j2x": a_j2[0],
                "a_j2y": a_j2[1],
                "a_j2z": a_j2[2],
                "a_solarx": a_solar[0],
                 "a_solary": a_solar[1],
                 "a_solarz": a_solar[2],
            })



    def get_log(self):
        return pd.DataFrame(self.log)

    #calculates the effect of central gravity in a system
    def central_gravity(self):
        # Compute central gravitational acceleration vector
        r_vec = np.array([self.x, self.y, self.z])
        r_mag = np.linalg.norm(r_vec)
        acc = -self.MU * r_vec / r_mag**3
    # Earth's gravitational parameter in km^3/s^2
        return acc


    #adds the effect of j2 pertubation
    def j2_perturbation(self):
        # Compute acceleration from Earth's J2 oblateness effect
        J2 = 1.08263e-3
        R = self.EARTH_RADIUS
    # Earth's radius in kilometers (used to set orbital altitude)
        mu = self.MU
    # Earth's gravitational parameter in km^3/s^2
        x, y, z = self.x, self.y, self.z
        r = np.sqrt(x**2 + y**2 + z**2)

        factor = 1.5 * J2 * mu * R**2 / r**5
        zx_ratio_sq = 5 * z**2 / r**2

        ax = factor * x * (zx_ratio_sq - 1)
        ay = factor * y * (zx_ratio_sq - 1)
        az = factor * z * (zx_ratio_sq - 3)


        return np.array([ax, ay, az])

    #calculates solar radiation effects
    def solar_radiation_pressure(self, t):
        # constants (you can move these to __init__)
        P0 = 4.56e-6     # N/m^2 at 1 AU
        AU = 149_597_870.7  # km
        C_R = 1.5
        A    = 20.0      # m^2
        m    = 1000.0    # kg

        # get Sun from self.bodies
        sun = next((b for b in getattr(self, "bodies", []) if b.name == "Sun"), None)
        if sun is None:
            return np.zeros(3)

        r_sat  = np.array([self.x, self.y, self.z], dtype=float)      # km
        r_sun  = sun.position_at(t)                                   # km
        r_vec  = r_sat - r_sun
                                           # Sun -> sat (correct push direction)
        d = np.linalg.norm(r_vec)
        if d == 0:
            return np.zeros(3)
        u = r_vec / d

        a_mag  = (P0 * C_R * A / m) * (AU**2 / d**2) / 1000.0         # km/s^2
        return a_mag * u
