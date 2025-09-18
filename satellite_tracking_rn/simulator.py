# simulator.py

import numpy as np
import constants as const
import pandas as pd
from skyfield.api import load

class OrbitSimulator:
    """
    SDE-based orbit simulator incorporating major orbital perturbations
    and a control policy.
    """

    def __init__(self, satellite_params: dict, policy_params: dict, sigma: float = 0.0):
        self.satellite_params = satellite_params
        self.policy_params = policy_params
        self.sigma = sigma
        self.eph = load('de421.bsp')
        self.ts = load.timescale()
        self.earth = self.eph['earth']
        self.sun = self.eph['sun']
        self.moon = self.eph['moon']
        self.G = np.zeros((6, 3))
        self.G[3:, :] = np.identity(3)
        self.log = [] # Initialize the log list

    def precompute_ephemeris(self, times: np.ndarray):
        """Pre-computes Sun and Moon positions for the simulation duration."""
        sky_times = self.ts.utc(2025, 1, 1, 0, 0, times)
        
        self.sun_pos_m = self.sun.at(sky_times).position.m - self.earth.at(sky_times).position.m
        self.moon_pos_m = self.moon.at(sky_times).position.m - self.earth.at(sky_times).position.m


    # --- Core Force Models ---

    def _central_gravity(self, r: np.ndarray) -> np.ndarray:
        return -const.MU_EARTH / np.linalg.norm(r)**3 * r

    def _j2_perturbation(self, r: np.ndarray) -> np.ndarray:
        x, y, z = r
        r_norm = np.linalg.norm(r)
        pre_factor = -1.5 * const.J2 * const.MU_EARTH * const.R_EARTH**2 / r_norm**5
        z_factor = 5 * z**2 / r_norm**2
        ax = pre_factor * x * (1 - z_factor)
        ay = pre_factor * y * (1 - z_factor)
        az = pre_factor * z * (3 - z_factor)
        return np.array([ax, ay, az])

    def _third_body_perturbation(self, r: np.ndarray, k: int) -> np.ndarray:
        r_sun = self.sun_pos_m[:, k]
        r_moon = self.moon_pos_m[:, k]
        
        r_sat_sun = r_sun - r
        a_sun = const.MU_SUN * (r_sat_sun / np.linalg.norm(r_sat_sun)**3 - r_sun / np.linalg.norm(r_sun)**3)
        r_sat_moon = r_moon - r
        a_moon = const.MU_MOON * (r_sat_moon / np.linalg.norm(r_sat_moon)**3 - r_moon / np.linalg.norm(r_moon)**3)
        return a_sun + a_moon

    def _srp_perturbation(self, r: np.ndarray, k: int) -> np.ndarray:
        C_R, A, m = self.satellite_params['C_R'], self.satellite_params['A'], self.satellite_params['m']
        r_sun = self.sun_pos_m[:, k]
        r_sat_sun = r_sun - r
        return -const.P_R * C_R * A / m * (r_sat_sun / np.linalg.norm(r_sat_sun))

    # PD Control Policy with Deadband
    def _control_policy(self, s: np.ndarray, s_ref: np.ndarray) -> np.ndarray:
        """
        Calculates the control acceleration based on a PD controller with a deadband.
        The controller is only active when the position error exceeds a radius gamma.
        """
        gamma = self.policy_params.get('gamma', 0)
        Kp = self.policy_params.get('Kp', 0)
        Kd = self.policy_params.get('Kd', 0)
        
        delta_r = s_ref[:3] - s[:3]  
        delta_v = s_ref[3:] - s[3:]

        # Check if the satellite is outside the deadband radius
        position_error_norm = np.linalg.norm(delta_r)
        
        if position_error_norm > gamma:
            # Controller is active: apply correction
            return -Kp * delta_r - Kd * delta_v
        else:
            # Controller is inactive: no thrust
            return np.zeros(3)

    # --- Simulation ---

    def _get_total_acceleration(self, s: np.ndarray, t: float, s_ref: np.ndarray, k: int) -> np.ndarray:
        r = s[:3]
        a_gravity = self._central_gravity(r)
        a_j2 = self._j2_perturbation(r)
        a_3b = self._third_body_perturbation(r, k)
        a_srp = self._srp_perturbation(r, k)
        a_policy = self._control_policy(s, s_ref)
        
        return a_gravity + a_j2 + a_3b + a_srp + a_policy


    
    def simulate_SLV(self, s0: np.ndarray, t_span: tuple, dt: float, ref_trajectory: np.ndarray) -> tuple:
        """
        Runs the full simulation using the Leapfrog Verlet stochastic algorithm.
        """
        times = np.arange(t_span[0], t_span[1], dt)
        states = np.zeros((len(times), 6))
        control_actions = np.zeros((len(times), 3))
        states[0] = s0

        # Precompute ephemeris data 
        print("Precomputing ephemeris data...")
        self.precompute_ephemeris(times)

        # Log the initial state
        self._log_state(0, times[0], states[0], ref_trajectory[0], np.zeros(3))

        
        sqrt_dt = np.sqrt(dt)

        for k in range(len(times) - 1):
            s_k = states[k]
            t_k = times[k]
            s_ref_k = ref_trajectory[k]
            
            # Extract position and velocity at step k
            r_k = s_k[:3]
            v_k = s_k[3:]
            
            # Get the control action for the log
            a_policy = self._control_policy(s_k, s_ref_k)
            control_actions[k] = a_policy

            # --- Leapfrog Verlet Algorithm ---
            
            # Step 1: Draw Wiener process increment
            dW_k = np.random.normal(0.0, 1.0, 3) * sqrt_dt
            noise_term = self.sigma * self.G @ dW_k
            
            # Step 2: Compute half-step supporting value
            r_half_step = r_k + 0.5 * v_k * dt
            
            # Step 3: Compute the total acceleration at the half-step position
            s_half_step = np.hstack([r_half_step, v_k])
            a_half_step = self._get_total_acceleration(s_half_step, t_k, s_ref_k, k)

            # Step 4: Update velocity and position
            v_k_plus_1 = v_k + a_half_step * dt + noise_term[3:]
            r_k_plus_1 = r_half_step + 0.5 * v_k_plus_1 * dt
            
            states[k+1] = np.hstack([r_k_plus_1, v_k_plus_1])
            
            # Log the state after the step
            self._log_state(k+1, times[k+1], states[k+1], ref_trajectory[k+1], a_policy)
            

    def simulate(self, s0: np.ndarray, t_span: tuple, dt: float, ref_trajectory: np.ndarray) -> tuple:
        """
        Runs the full simulation and returns states and control actions.
        Returns:
            tuple: (time array, state trajectory array, control action history array)
        """
        times = np.arange(t_span[0], t_span[1], dt)
        states = np.zeros((len(times), 6))
        control_actions = np.zeros((len(times), 3)) # Added storage for control history
        states[0] = s0

        # Precompute ephemeris data 
        print("Precomputing ephemeris data...")
        self.precompute_ephemeris(times)

        # Log the initial state
        self._log_state(0, times[0], states[0], ref_trajectory[0], np.zeros(3))


        sqrt_dt = np.sqrt(dt)

        for k in range(len(times) - 1):
            s_k = states[k]
            t_k = times[k]
            s_ref_k = ref_trajectory[k]
            
            # Get the total acceleration for the dynamics step
            a_k = self._get_total_acceleration(s_k, t_k, s_ref_k, k)

            a_policy = self._control_policy(s_k, s_ref_k)
            
            # Semi-Implicit Euler-Maruyama Step
            v_k_plus_1 = s_k[3:] + a_k * dt
            r_k_plus_1 = s_k[:3] + v_k_plus_1 * dt
            s_k_plus_1_drift = np.hstack([r_k_plus_1, v_k_plus_1])
            
            dW_k = np.random.normal(0.0, 1.0, 3) * sqrt_dt
            noise_term = self.sigma * self.G @ dW_k
            
            states[k+1] = s_k_plus_1_drift + noise_term

            # Log the state after the step
            self._log_state(k+1, times[k+1], states[k+1], ref_trajectory[k+1], a_policy)
            
    
    def _log_state(self, k: int, t: float, sim_state: np.ndarray, ref_state: np.ndarray, control_accel: np.ndarray):
        """Helper function to log the state at each time step."""
        r_sim, v_sim = sim_state[:3], sim_state[3:]
        r_ref, v_ref = ref_state[:3], ref_state[3:]
        
        # Determine control status
        status = "non-stationkeeping"
        if np.linalg.norm(control_accel) > 1e-15:
            status = "stationkeeping"

        # Calculate longitude in degrees from the simulated position
        longitude_deg = np.degrees(np.arctan2(r_sim[1], r_sim[0]))
        
        # Get the full acceleration including all perturbations
        a_total = self._get_total_acceleration(sim_state, t, ref_state, k)

        log_entry = {
            "Time (s)": t,
            "Satellite ID": "Geosat-1",  # Placeholder, could be a class attribute
            "Longitude (deg)": longitude_deg,
            "X (m)": r_sim[0],
            "Y (m)": r_sim[1],
            "Z (m)": r_sim[2],
            "VX (m/s)": v_sim[0],
            "VY (m/s)": v_sim[1],
            "VZ (m/s)": v_sim[2],
            "ideal X (m)": r_ref[0],
            "ideal Y (m)": r_ref[1],
            "ideal Z (m)": r_ref[2],
            "ideal VX (m/s)": v_ref[0],
            "ideal VY (m/s)": v_ref[1],
            "ideal VZ (m/s)": v_ref[2],
            "Status": status,
            "Control Accel (X)": control_accel[0],
            "Control Accel (Y)": control_accel[1],
            "Control Accel (Z)": control_accel[2],
            "Total Accel (X)": a_total[0],
            "Total Accel (Y)": a_total[1],
            "Total Accel (Z)": a_total[2],
        }
        self.log.append(log_entry)

    def get_log(self) -> pd.DataFrame:
        """Returns the simulation log as a Pandas DataFrame."""
        return pd.DataFrame(self.log)