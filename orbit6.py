import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Satellite:
    def __init__(self, sat_id, semi_major_axis, eccentricity, noise_freq, max_deviation, time_step=10, mu=398600.4418):
        self.sat_id = sat_id
        self.a = semi_major_axis
        self.e = eccentricity
        self.noise_freq = noise_freq  # 0 to 1 probability per step
        self.max_deviation = max_deviation
        self.mu = mu  # Earth's gravitational parameter [km^3/s^2]
        self.time_step = time_step  # seconds
        self.theta = 0  # true anomaly in radians
        self.noisy = False
        self.state = "normal"
        self.orbit_log = []

    def _calculate_position_velocity(self, theta):
        # 3D position using inclination to simulate z-dimension
        r = (self.a * (1 - self.e ** 2)) / (1 + self.e * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0.1 * r * np.sin(2 * theta)  # fake Z-motion

        h = np.sqrt(self.mu * self.a * (1 - self.e ** 2))  # specific angular momentum
        vx = -self.mu / h * np.sin(theta)
        vy = self.mu / h * (self.e + np.cos(theta))
        vz = 0.1 * vx  # fake Z-velocity

        return np.array([x, y, z]), np.array([vx, vy, vz])
    """
        r = (self.a * (1 - self.e ** 2)) / (1 + self.e * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        h = np.sqrt(self.mu * self.a * (1 - self.e ** 2))  # specific angular momentum
        vx = -self.mu / h * np.sin(theta)
        vy = self.mu / h * (self.e + np.cos(theta))
        return np.array([x, y]), np.array([vx, vy])
        """

    def propagate(self, duration):
        t = 0
        while t < duration:
            pos_nominal, vel = self._calculate_position_velocity(self.theta)

            if self.state == "normal":
                if np.random.rand() < self.noise_freq:
                    self.state = "drifting"
                    self.deviation_vector = np.random.uniform(10, 20, size=3)  # NOW 3D
                pos = pos_nominal

            elif self.state == "drifting":
                self.deviation_vector += np.random.uniform(0.5, 1, size=3)     # NOW 3D
                pos = pos_nominal + self.deviation_vector
                if np.linalg.norm(self.deviation_vector) >= self.max_deviation:
                    self.state = "correcting"

            elif self.state == "correcting":
                correction_step = -0.2 * self.deviation_vector
                self.deviation_vector += correction_step
                pos = pos_nominal + self.deviation_vector
                if np.linalg.norm(self.deviation_vector) <= 1.0:
                    self.state = "normal"
                    self.deviation_vector = np.zeros(3)
                    pos = pos_nominal

            self.orbit_log.append({
                "sat_id": self.sat_id,
                "time": t,
                "posX": pos[0], "posY": pos[1], "posZ": pos[2],
                "vecX": vel[0], "vecY": vel[1], "vecZ": vel[2],
                "status": self.state
            })

            self.theta += np.sqrt(self.mu / self.a ** 3) * self.time_step
            t += self.time_step

    def get_log(self):
        return pd.DataFrame(self.orbit_log)

def plot3d(log_df):
    x = log_df["posX"].values
    y = log_df["posY"].values
    z = log_df["posZ"].values
    status = log_df["status"].values

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot satellite trajectory
    ax.scatter(x[status == "normal"], y[status == "normal"], z[status == "normal"], label="Normal", s=3)
    ax.scatter(x[status == "drifting"], y[status == "drifting"], z[status == "drifting"], label="Drifting", s=3, marker='x')
    ax.scatter(x[status == "correcting"], y[status == "correcting"], z[status == "correcting"], label="Correcting", s=3, marker='^')

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
    plt.show()

def plot2d(log_df):
    import matplotlib.pyplot as plt

    x = log_df["posX"].values
    y = log_df["posY"].values
    status = log_df["status"].values

    plt.figure(figsize=(8, 8))

    plt.scatter(x[status == "normal"], y[status == "normal"], label="Normal", s=3, alpha=0.6)
    plt.scatter(x[status == "drifting"], y[status == "drifting"], label="Drifting", s=3, alpha=0.6, marker='x')
    plt.scatter(x[status == "correcting"], y[status == "correcting"], label="Correcting", s=3, alpha=0.6, marker='^')

    # Optional: plot Earth as a blue circle (not to scale)
    earth = plt.Circle((0, 0), 6371, color='blue', alpha=0.2, label="Earth (not to scale)")
    plt.gca().add_artist(earth)

    plt.xlabel("X Position (km)")
    plt.ylabel("Y Position (km)")
    plt.title("2D XY View of Satellite Orbit")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


"""
def plot_satellite_trajectory(log_df):
    positions = np.array(log_df["position"].tolist())
    x = positions[:, 0]
    y = positions[:, 1]
    status = np.array(log_df["status"])

    plt.figure(figsize=(8, 8))

    # Plot normal (on-orbit)
    mask_normal = status == "normal"
    plt.scatter(x[mask_normal], y[mask_normal], s=5, label="Normal", alpha=0.6)

    # Plot drifting
    mask_drifting = status == "drifting"
    plt.scatter(x[mask_drifting], y[mask_drifting], s=5, label="Drifting", alpha=0.6, marker='x')

    # Plot correcting
    mask_correcting = status == "correcting"
    plt.scatter(x[mask_correcting], y[mask_correcting], s=5, label="Correcting", alpha=0.6, marker='^')

    plt.xlabel("X Position (km)")
    plt.ylabel("Y Position (km)")
    plt.title("Satellite Trajectory with Drift and Correction")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()
    """
# Example usage
sat = Satellite(
    sat_id="SAT-001",
    semi_major_axis=7000,  # in km
    eccentricity=0.00001,
    noise_freq=0.0001,  # 5% chance of going off-orbit per time step
    max_deviation=200  # km
)
sat.propagate(duration=1000000)  # simulate for 2 hour
log_df = sat.get_log()
print(log_df.head(40))


# Use this after generating the log
plot3d(log_df)
plot2d(log_df)
#plot_satellite_trajectory(log_df)
