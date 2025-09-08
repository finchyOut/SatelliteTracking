import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D

#correctlty implements the drifting of satellite 
# accounts for drift in the velocity variable

#This is the initial orbit now with the effects of j2, solar radiation, and central gravity effects

#/opt/anaconda3/envs/cleanorbit/bin/python "/Users/emmafinch/Desktop/Thesis/Python Code/orbit16.py"
# --- Perfect circular third-body models (geocentric frame) ---
# QP solver import shim: prefers `qp_solver`, falls back to `qpsolvers`
try:
    import qp_solver as _qps  # if your environment provides qp_solver
    _solve_qp = _qps.solve_qp
except Exception:
    try:
        import qpsolvers as _qps  # pip install qpsolvers
        _solve_qp = _qps.solve_qp
    except Exception:
        _qps = None
        _solve_qp = None


class CircularBody:
    """
    Idealized *circular* orbit model in a geocentric, inertial frame (Earth at origin).
    Physics/Math background:
    ------------------------
    • We assume uniform circular motion in the XY-plane:
          r(t) = [ R cos(θ(t)), R sin(θ(t)), 0 ]
       where θ(t) = θ0 + ω t.

    • Angular rate ω relates to period T by:
          ω = 2π / T.

    • If this body were actually orbiting Earth under Newtonian gravity,
      the circular orbit period would satisfy Kepler's 3rd law (two-body form):
          T = 2π √( a^3 / μ_E ),
      with a = R (circular orbit radius) and μ_E the Earth's gravitational parameter.
      In our simple model we *input* a period directly (e.g., 1 sidereal year for the Sun's apparent geocentric motion,
      or 1 sidereal month for the Moon), so we don’t need μ_E to compute ω.

    • We *also* store the body's own gravitational parameter μ (μ = G M_body).
      We do NOT use this μ to move the body (because we’re forcing a perfect circle),
      but we DO use it to compute the body’s gravitational perturbation on the satellite via the standard third‑body term:
          a_3b = μ_b [ (r_b − r)/|r_b − r|^3 − r_b/|r_b|^3 ].
      This captures the *differential* acceleration between the satellite at r and Earth’s center at 0.

    • The z–component is fixed to 0 ⇒ orbit plane is the equatorial/ecliptic plane in this simplified model.
      Inclination, RAAN, and argument of latitude are ignored here for clarity.
    """
    def __init__(self, name, R, mu, period, phase=0.0, time_step=60):
        # Human-readable label for logs/plots (e.g., "Sun" or "Moon")
        self.name = name

        # Orbital radius R [km]: distance from Earth's center to this body.
        # In this simplified model R is constant (perfect circle).
        self.R = float(R)

        # Body's gravitational parameter μ = G * M_body [km^3/s^2].
        # Used to compute its gravity on the satellite (third‑body effect), NOT to advance this body's position.
        self.mu = float(mu)

        # Orbital period T [s]: time for one full revolution around Earth (geocentric view).
        # Examples: Sun ~ 1 sidereal year; Moon ~ 1 sidereal month.
        self.period = float(period)

        # Initial phase θ0 [rad] at simulation time t = 0.
        # θ0 = 0 ⇒ body starts on +X axis; θ0 = π/2 ⇒ starts on +Y axis, etc.
        self.phase = float(phase)

        # A convenience for *optional* internal logging. It does NOT affect physics.
        # If you call body.propagate(), this is the step between saved snapshots.
        self.time_step = int(time_step)

        # Angular speed ω = 2π / T [rad/s]. For uniform circular motion, θ(t) = θ0 + ω t.
        self.omega = 2.0 * np.pi / self.period

        # --- Internal state for optional standalone propagation/logging ---
        # Internal clock for this body (seconds since its own start). The satellite never reads this;
        # the satellite always queries the *analytic* position_at(t) with its own time 't'.
        self.t = 0

        # Initialize position consistent with the chosen phase at t = 0:
        # x(0) = R cos(θ0), y(0) = R sin(θ0), z(0) = 0 (orbit plane is XY).
        self.x = self.R * np.cos(self.phase)
        self.y = self.R * np.sin(self.phase)
        self.z = 0.0

        # A simple list of dicts to store optional logs (time, x, y, z) if you call propagate().
        self.log = []

    def position_at(self, t_seconds):
        """
        Analytic position at *absolute* simulation time t (seconds),
        using uniform circular motion:
            θ(t) = θ0 + ω t
            r(t) = [ R cos(θ), R sin(θ), 0 ].
        Note: This ignores any perturbations and enforces a perfect circle by construction.
        """
        theta = self.phase + self.omega * t_seconds   # current true anomaly in our simple circular model
        x = self.R * np.cos(theta)
        y = self.R * np.sin(theta)
        z = 0.0
        return np.array([x, y, z], dtype=float)

    def propagate(self, duration):
        """
        OPTIONAL helper to create a time series of this body's own positions for plotting.
        Physics note:
        -------------
        • This does not 'integrate' any forces; it simply samples the analytic solution above
          at discrete times separated by self.time_step.

        • The satellite NEVER depends on this; during satellite propagation we always call
          position_at(t) with the satellite's current absolute time, which gives exact (analytic) positions.
        """
        for _ in range(0, duration, self.time_step):
            # Advance this body's internal clock
            self.t += self.time_step

            # Compute position from the analytic circular solution at the new time
            x, y, z = self.position_at(self.t)

            # Cache state for potential plotting/CSV export
            self.x, self.y, self.z = x, y, z
            self.log.append({
                "Time (s)": self.t,
                f"{self.name} X (km)": x,
                f"{self.name} Y (km)": y,
                f"{self.name} Z (km)": z,
            })

    def get_log(self):
        """
        Returns a DataFrame of the optional logged positions.
        Not used by the satellite’s gravity calculation; purely for convenience.
        """
        return pd.DataFrame(self.log)



class Sun(CircularBody):
    """
    Perfect circular geocentric Sun (for simple third-body acceleration).
    Note: This is a geocentric approximation for perturbation modeling.
    """
    # 1 AU in km; mu_sun in km^3/s^2; sidereal year ~365.256 days
    AU_KM = 149_597_870.7
    MU_SUN = 1.32712440018e11
    YEAR_S = 365.256363004 * 86400.0

    def __init__(self, phase_deg=0.0, time_step=3600):
        super().__init__(
            name="Sun",
            R=self.AU_KM,
            mu=self.MU_SUN,
            period=self.YEAR_S,
            phase=np.radians(phase_deg),
            time_step=time_step
        )


class Moon(CircularBody):
    """
    Perfect circular geocentric Moon.
    """
    # Mean distance and mu_moon; sidereal month ~27.321661 days
    R_MOON = 384_400.0            # km
    MU_MOON = 4902.800066         # km^3/s^2
    MONTH_S = 27.321661 * 86400.0

    def __init__(self, phase_deg=0.0, time_step=600):
        super().__init__(
            name="Moon",
            R=self.R_MOON,
            mu=self.MU_MOON,
            period=self.MONTH_S,
            phase=np.radians(phase_deg),
            time_step=time_step
        )


class Satellite:
    # Satellite class simulates orbital propagation with optional stationkeeping
    EARTH_RADIUS = 6371  # km
    # Earth's radius in kilometers (used to set orbital altitude)
    MU = 398600.4418     # km^3/s^2
    # Earth's gravitational parameter in km^3/s^2

    def __init__(self, sat_id, longitude_deg,
        # Initialize satellite with orbital parameters and simulation settings
                 eccentricity=0, noise_std=0.01, max_drift_km=100,
                 time_step=60, bodies=None, 
                 gamma_km=10.0,      # radius of the allowed "ball" around target (component-wise L∞ bound)
                H_steps=10,         # decide control once every H discrete steps
                a_max=None):
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
        # Initial z velocity (equatorial orbit = 0)
        self.gamma_km = float(gamma_km)
        self.H_steps  = int(H_steps)
        self.a_max    = None if a_max is None else np.asarray(a_max, dtype=float)
        self.log = []

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
        # Main simulation loop
        dt = float(self.time_step)
        kStep = 0
        prevStatus = "non-stationkeeping"

        for t in range(0, duration, self.time_step):
            kStep += 1

            # Ideal GEO ring position for logging/targeting
            angle   = self.longitude + self.omega * t
            ideal_x = self.a * np.cos(angle)
            ideal_y = self.a * np.sin(angle)
            ideal_z = 0.0

            # Natural accelerations at current state
            a_grav  = self.central_gravity()
            a_j2    = self.j2_perturbation()
            a_solar = self.solar_radiation_pressure(t)
            a_3b    = self.third_body_acceleration(t)

            # Control (every H_steps)
            ax = ay = az = 0.0
            status = "non-stationkeeping"
            if kStep >= self.H_steps:
                #print("I have entered the loop")
                r_now = np.array([self.x,  self.y,  self.z ], float)
                v_now = np.array([self.vx, self.vy, self.vz], float)
                a_ctrl = self._qp_control_once(r_now, v_now, float(t))
                ax, ay, az = a_ctrl
                prevStatus = status
                if np.any(a_ctrl != 0.0):
                    
                    status = "stationkeeping"
                if np.any(a_ctrl > 1e5):  
                    print("the acceleration", a_ctrl)
                    exit()
                kStep = 0
            if prevStatus == "non-stationkeeping" and status == "stationkeeping":
                self.vx = -self.orbital_velocity()*np.sin(angle)
                self.vy = self.orbital_velocity()*np.cos(angle)
                self.vz =0 

            # Total acceleration
            a_total = np.array([ax, ay, az], float) + a_grav + a_j2 + a_solar + a_3b

            # --- Symplectic (semi-implicit) Euler ---
            # 1) v_{k+1} = v_k + a_total * dt
            self.vx += a_total[0] * dt
            self.vy += a_total[1] * dt
            self.vz += a_total[2] * dt

            # Small process noise on velocity (km/s) AFTER physics update
            if getattr(self, "noise_std", 0.0) > 0.0:
                self.vx += np.random.normal(0.0, self.noise_std)
                self.vy += np.random.normal(0.0, self.noise_std)
                self.vz += np.random.normal(0.0, self.noise_std)

            # 2) r_{k+1} = r_k + v_{k+1} * dt
            self.x  += self.vx * dt
            self.y  += self.vy * dt
            self.z  += self.vz * dt

            # Log once per step
            entry = {
                "Time (s)": t,
                "Satellite ID": self.sat_id,
                "Longitude (deg)": float(np.degrees(self.longitude)),
                "X (km)": float(self.x), "Y (km)": float(self.y), "Z (km)": float(self.z),
                "VX": float(self.vx), "VY": float(self.vy), "VZ": float(self.vz),
                "ideal VX": -self.orbital_velocity()*np.sin(angle),
                "ideal VY":  self.orbital_velocity()*np.cos(angle),
                "ideal VZ":  0.0,
                "ideal x": float(ideal_x), "ideal_y": float(ideal_y), "ideal_z": float(ideal_z),
                "Status": status,
            }
            for body in getattr(self, "bodies", []) or []:
                bx, by, bz = body.position_at(t)
                entry[f"{body.name} X (km)"] = float(bx)
                entry[f"{body.name} Y (km)"] = float(by)
                entry[f"{body.name} Z (km)"] = float(bz)
            self.log.append(entry)

                
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
    
    def _qp_control_once1(self, r_now, v_now, t_now):
        """
        Solve  min 1/2 * ||alpha||^2
            s.t.  | (r_now + v_now*dt + 0.5*alpha*dt^2) - r_target_next | <= gamma   (component-wise)
        Returns alpha (km/s^2).
        """
        dt = float(self.time_step)
        angle_next = self.longitude + self.omega * (t_now + dt)
        r_target_next = np.array([
            self.a * np.cos(angle_next),
            self.a * np.sin(angle_next),
            0.0
        ], dtype=float)

        # Predicted error at t+dt without control (NOTE: sign!)
        c = (np.asarray(r_now, float) + np.asarray(v_now, float) * dt) - r_target_next

        dt2   = dt * dt
        gamma = float(self.gamma_km)

        # Correct box for alpha
        lb = 2.0 * (-gamma - c) / dt2
        ub = 2.0 * ( gamma - c) / dt2

        # Optional actuator limits
        if self.a_max is not None:
            amax = np.atleast_1d(np.asarray(self.a_max, float))
            if amax.size == 1:
                amax = np.repeat(amax.item(), 3)
            lb = np.maximum(lb, -np.abs(amax))
            ub = np.minimum(ub,  np.abs(amax))

        # If a QP solver is available
        if _solve_qp is not None:
            P = np.eye(3); q = np.zeros(3)
            try:
                sol = _solve_qp(P, q, G=None, h=None, A=None, b=None, lb=lb, ub=ub)
                if sol is not None:
                    print( sol)
                    return np.asarray(sol, float)
            except Exception:
                pass
        alpha = np.clip(np.zeros(3), lb, ub)
        print(alpha)
        # Closed form if no solver available: project 0 into [lb, ub]
        return alpha
    
    def _qp_control_once2(self, r_now, v_now, t_now):
        """
        Solve  min 1/2 * ||alpha||^2
            s.t.  | (r_now + v_now*dt + 0.5*alpha*dt^2) - r_target_next | <= gamma   (component-wise)
        Returns alpha (km/s^2).
        """
        dt = float(self.time_step)
        angle_next = self.longitude + self.omega * (t_now + dt)
        r_target_next = np.array([
            self.a * np.cos(angle_next),
            self.a * np.sin(angle_next),
            0.0
        ], dtype=float)

        I = np.eye(3)
        h = self.gamma_km*I - r_now + self.omega * r_now *self.time_step + r_target_next
        G = I *self.time_step

        # Optional actuator limits
        if self.a_max is not None:
            amax = np.atleast_1d(np.asarray(self.a_max, float))
            if amax.size == 1:
                amax = np.repeat(amax.item(), 3)

        sol = None
        # If a QP solver is available
        if _solve_qp is not None:
            P = np.eye(3); q = np.zeros(3)
            try:
                print("I have entered the try")
                sol = _solve_qp(P, q, G=G, h= h, A=None, b=None, lb = None, ub=None)
                print("The current solution", sol)
                if sol is not None:
                    print("Here is the solution", sol)
                    return np.asarray(sol, float)
            except Exception:
                print("You have hit the exception")
                pass
    
        if sol is not None:
            #print(sol)
            alpha = sol
        else:
            alpha = [0,0,0]
        # Closed form if no solver available: project 0 into [lb, ub]
        #print(alpha)
        return alpha
    
    def _qp_control_once(self, r_now, v_now, t_now):
        """
        Solve:  minimize  1/2 * ||alpha||^2
                subject to  | (r_now + v_now*dt + 0.5*alpha*dt^2) - r_target_next | <= gamma
        Returns alpha (km/s^2).
        """
        import numpy as np

        dt = float(self.time_step)
        a = float(self.a)  # semi-major axis in km
        gamma = float(self.gamma_km)  # GEO "box" half-width per axis in km (scalar)

        # Ideal target position one step ahead on the GEO ring (equatorial, circular)
        angle_next = self.longitude + self.omega * (t_now + dt)
        r_target_next = np.array([a * np.cos(angle_next),
                                a * np.sin(angle_next),
                                0.0], dtype=float)

        # If already inside the box, optimal alpha is 0
        #I = np.eye(3)
        #delta = self.gamma_km*I - r_now + self.omega * r_now *self.time_step + r_target_next
        delta = (np.asarray(r_now, float) + np.asarray(v_now, float) * dt) - r_target_next
        if np.all(np.abs(delta) <= gamma):
            return np.zeros(3, dtype=float)

        # Build QP: min 1/2 alpha^T I alpha  s.t.  G alpha <= h
        P = np.eye(3, dtype=float)
        q = np.zeros(3, dtype=float)

        Aabs = 0.5 * (dt ** 2) * np.eye(3, dtype=float)  # coefficient of alpha in constraint
        # Stack + and - inequalities: shape (6,3)
        G = np.vstack([ Aabs, -Aabs ])

        # Right-hand sides: shape (6,)
        #   +:  Aabs*alpha <= gamma - delta
        #   -: -Aabs*alpha <= gamma + delta
        h = np.hstack([ gamma - delta,  gamma + delta ]).astype(float)
        print("h:", h)
        print("G: ", G)
        print("gamma: ", gamma)


        # Optional actuator limits: -amax <= alpha <= amax
        lb = ub = None
        if getattr(self, "a_max", None) is not None:
            amax = np.atleast_1d(np.asarray(self.a_max, float))
            if amax.size == 1:
                amax = np.repeat(amax.item(), 3)
            lb = -amax
            ub =  amax

        # Solve
        sol = None
        if _solve_qp is not None:
            
            try:
                print("Entered the try")
                #print(_solve_qp(P, q, G=G, h=h, A=None, b=None, lb=lb, ub=ub, solver="osqp"))
                sol = _solve_qp(P, q, G=G, h=h, A=None, b=None, lb=lb, ub=ub, solver= 'scs')
                print("The solution is:", sol)
            except Exception as e:
                # Surface the error to actually see what's wrong during debugging
                print(f"[QP ERROR] {e}")

        # Fallback
        if sol is None:
            # If infeasible (e.g., gamma too small for given dt), return best-effort zeros
            # or consider projecting to bounds if lb/ub exist.
            return np.zeros(3, dtype=float)

        return np.asarray(sol, float)

    
