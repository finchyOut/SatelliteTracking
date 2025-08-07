import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#correctlty implements the drifting of satellite 
# accounts for drift in the velocity variable

#This is the initial orbit now with the effects of j2, solar radiation, and central gravity effects

#/opt/anaconda3/envs/cleanorbit/bin/python "/Users/emmafinch/Desktop/Thesis/Python Code/orbit14.py"
class Satellite:
    # Satellite class simulates orbital propagation with optional stationkeeping
    EARTH_RADIUS = 6371  # km
    # Earth's radius in kilometers (used to set orbital altitude)
    MU = 398600.4418     # km^3/s^2
    # Earth's gravitational parameter in km^3/s^2

    def __init__(self, sat_id, longitude_deg,
        # Initialize satellite with orbital parameters and simulation settings
                 eccentricity=0, noise_std=0.01, max_drift_km=100,
                 time_step=60):
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
        # Initial z velocity (equatorial orbit = 0)

        self.log = []

    def orbital_velocity(self):
        # Compute ideal orbital velocity for circular orbit (used for control)
        return np.sqrt(self.MU / self.a)
    # Earth's gravitational parameter in km^3/s^2

    def propagate(self, duration):
        # Main simulation loop: propagates position and velocity over time
        for t in range(0, duration, self.time_step):
        # Time increment for simulation (in seconds)
            angle = self.longitude + self.omega * t
        # Longitude converted to radians (for initial position)
            ideal_x = self.a * np.cos(angle)
            ideal_y = self.a * np.sin(angle)
            ideal_z = 0

            # Add noise to velocity instead of position
            noise = np.random.normal(0, self.noise_std, 3)
        # Standard deviation of Gaussian noise for velocity perturbation
            self.vx += noise[0]
            self.vy += noise[1]
            self.vz += noise[2]

            # Then update position as usual
            self.x += self.vx * self.time_step
        # Time increment for simulation (in seconds)
            self.y += self.vy * self.time_step
        # Time increment for simulation (in seconds)
            self.z += self.vz * self.time_step
        # Time increment for simulation (in seconds)


            dx = self.x - ideal_x
            dy = self.y - ideal_y
            dz = self.z - ideal_z
            drift_distance = np.sqrt(dx**2 + dy**2 + dz**2)

            if drift_distance > self.max_drift:
        # Maximum drift distance from ideal position before stationkeeping triggers
                status = "stationkeeping"

                # Proportional-Derivative Control
                kp = 1e-5
                kd = 2e-4
                #ideal velocity to get into perfect position
                desired_vx = -self.orbital_velocity() * np.sin(angle)
                desired_vy = self.orbital_velocity() * np.cos(angle)
                desired_vz = 0

                #calculate the difference between ideal velocity and current velocity
                dvx = self.vx - desired_vx
                dvy = self.vy - desired_vy
                dvz = self.vz - desired_vz

                #the updated acceleration based on our possicy taking our pd
                #controller into account
                ax = -kp * dx - kd * dvx
                ay = -kp * dy - kd * dvy
                az = -kp * dz - kd * dvz
            else:
                status = "non-stationkeeping"
                #if there is no station keeping, then keep velocity constant 
                ax = ay = az = 0
                desired_vx = self.vx
                desired_vy = self.vy
                desired_vz = self.vz

            #added code to account for the effects of j2, solar radiation, and central gravity
            a_grav = self.central_gravity()
            a_j2 = self.j2_perturbation()
            a_solar = self.solar_radiation_pressure()

            # update acceleration policies to account for these issues
            #note the ax is the satellite's concious decision to adjust
            #the forces naturally adjust the satellite's location therefore velocity and acceleration 
            # without the satellite's active awareness. 
            ax_total = a_grav[0] + a_j2[0] + a_solar[0] + ax
            ay_total = a_grav[1] + a_j2[1] + a_solar[1] + ay
            az_total = a_grav[2] + a_j2[2] + a_solar[2] + az

            # Apply total acceleration
            self.vx += ax_total * self.time_step
        # Time increment for simulation (in seconds)
            self.vy += ay_total * self.time_step
        # Time increment for simulation (in seconds)
            self.vz += az_total * self.time_step
        # Time increment for simulation (in seconds)

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
                "ideal VX": desired_vx,
                "ideal VY": desired_vy,
                "ideal VZ": desired_vz,
                "ideal x": ideal_x,
                "ideal_y": ideal_y,
                "ideal_z": ideal_z,
                "Status": status
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
    def solar_radiation_pressure(self):
        # Compute acceleration due to solar radiation pressure
        P_sr = 4.5e-6  # N/m^2
        C_R = 1.5
        A = 20          # m^2
        m = 1000        # kg
        sun_pos = np.array([1.496e8, 0, 0])  # km
        sat_pos = np.array([self.x, self.y, self.z])
        r_vec = sun_pos - sat_pos
        r_mag = np.linalg.norm(r_vec)
        direction = r_vec / r_mag
        acc_magnitude = (P_sr * C_R * A / m) / 1000  # convert to km/s²
        return acc_magnitude * direction

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
        noise_std=0.08,
        max_drift_km=1000,
        time_step=10
    )

sat.propagate(duration=86400)
df = sat.get_log()
df.to_csv("j2TestLog.csv", index=False)

plot2d(df)
plot3d(df)
