import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#correctlty implements the drifting of satellite however only accounts for noise in the position
#variable instead of the velocity variable

class Satellite:
    EARTH_RADIUS = 6371  # km
    MU = 398600.4418     # km^3/s^2

    def __init__(self, sat_id, longitude_deg,
                 eccentricity=0, noise_std=0.01, max_drift_km=100,
                 time_step=60):
        self.sat_id = sat_id
        self.a = self.EARTH_RADIUS + 35786
        self.longitude = np.radians(longitude_deg)
        self.ecc = eccentricity
        self.noise_std = noise_std
        self.max_drift = max_drift_km
        self.time_step = time_step
        self.omega = np.sqrt(self.MU / self.a**3)  # rad/sec

        # Initialize position and velocity (circular, equatorial)
        self.x = self.a * np.cos(self.longitude)
        self.y = self.a * np.sin(self.longitude)
        self.z = 0
        self.vx = -self.orbital_velocity() * np.sin(self.longitude)
        self.vy = self.orbital_velocity() * np.cos(self.longitude)
        self.vz = 0

        self.log = []

    def orbital_velocity(self):
        return np.sqrt(self.MU / self.a)

    def propagate(self, duration):
        for t in range(0, duration, self.time_step):
            angle = self.longitude + self.omega * t
            ideal_x = self.a * np.cos(angle)
            ideal_y = self.a * np.sin(angle)
            ideal_z = 0

            # Drift + motion
            noise = np.random.normal(0, self.noise_std, 3)
            self.x += self.vx * self.time_step + noise[0]
            self.y += self.vy * self.time_step + noise[1]
            self.z += self.vz * self.time_step + noise[2]

            dx = self.x - ideal_x
            dy = self.y - ideal_y
            dz = self.z - ideal_z
            drift_distance = np.sqrt(dx**2 + dy**2 + dz**2)

            if drift_distance > self.max_drift:
                status = "stationkeeping"

                # Proportional-Derivative Control
                kp = 1e-5
                kd = 2e-4
                desired_vx = -self.orbital_velocity() * np.sin(angle)
                desired_vy = self.orbital_velocity() * np.cos(angle)
                desired_vz = 0

                dvx = self.vx - desired_vx
                dvy = self.vy - desired_vy
                dvz = self.vz - desired_vz

                ax = -kp * dx - kd * dvx
                ay = -kp * dy - kd * dvy
                az = -kp * dz - kd * dvz
            else:
                status = "non-stationkeeping"
                ax = ay = az = 0

            self.vx += ax * self.time_step
            self.vy += ay * self.time_step
            self.vz += az * self.time_step

            self.log.append({
                "Time (s)": t,
                "Satellite ID": self.sat_id,
                "Longitude (deg)": np.degrees(self.longitude),
                "X (km)": self.x,
                "Y (km)": self.y,
                "Z (km)": self.z,
                "Status": status
            })

    def get_log(self):
        return pd.DataFrame(self.log)

def plot3d(log_df):
    x = log_df["X (km)"].values
    y = log_df["Y (km)"].values
    z = log_df["Z (km)"].values
    status = log_df["Status"].values

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot satellite trajectory
    ax.scatter(x[status == "stationkeeping"], y[status == "stationkeeping"], z[status == "stationkeeping"], label="stationkeeping", s=3)
    ax.scatter(x[status == "non-stationkeeping"], y[status == "non-stationkeeping"], z[status == "non-stationkeeping"], label="non-stationkeeping", s=3, marker='x')
    
    # Plot Earth
    R = 6371  # Earth's radius in km
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    ex = R * np.cos(u) * np.sin(v)
    ey = R * np.sin(u) * np.sin(v)
    ez = R * np.cos(v)
    ax.plot_surface(ex, ey, ez, color='blue', alpha=0.3, linewidth=0)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Satellite Orbit with Earth Centered")
    ax.legend()
    # Set equal aspect ratio for 3D plot
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

def plot2d(log_df):
    import matplotlib.pyplot as plt

    x = log_df["X (km)"].values
    y = log_df["Y (km)"].values
    status = log_df["Status"].values

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot satellite trajectory
    ax.scatter(x[status == "stationkeeping"], y[status == "stationkeeping"],
               label="stationkeeping", s=3, marker='^')
    ax.scatter(x[status == "non-stationkeeping"], y[status == "non-stationkeeping"],
               label="non-stationkeeping", s=3, marker='x')

    # Plot Earth
    earth = plt.Circle((0, 0), 6371, color='blue', alpha=0.2, label="Earth (not to scale)")
    ax.add_artist(earth)

    ax.set_xlabel("X Position (km)")
    ax.set_ylabel("Y Position (km)")
    ax.set_title("2D XY View of Satellite Orbit")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


sat = Satellite(
        sat_id= '01',
        longitude_deg= 45,
        noise_std=0.1,
        max_drift_km=1000,
        time_step=10
    )

sat.propagate(duration=86400)
df = sat.get_log()
plot2d(df)
plot3d(df)
