import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import vmap

from simulator import OrbitSimulator
import constants as const



def jax_diff_control_policy(s, s_ref, Kp, Kd, gamma, beta): # NOTE: Added beta parameter
    delta_r = s[:3] - s_ref[:3]
    delta_v = s[3:] - s_ref[3:]
    
    position_error_norm = jnp.linalg.norm(delta_r)
    
    # Sigmoid function for soft-switching
    soft_weight = 1.0 / (1.0 + jnp.exp(-beta * (position_error_norm - gamma)))

    # Apply the soft weight to the PD control term
    thrust = soft_weight * (-Kp * delta_r - Kd * delta_v)
    
    return thrust


# --- 1. JAX-COMPATIBLE ACCELERATION FUNCTIONS ---
def jax_control_policy(s, s_ref, Kp, Kd, gamma):
    delta_r = s[:3] - s_ref[:3]
    delta_v = s[3:] - s_ref[3:]
    
    position_error_norm = jnp.linalg.norm(delta_r)
    
    # Use jnp.where for vectorized and JIT-compatible logic
    thrust = jnp.where(
        position_error_norm > gamma,
        -Kp * delta_r - Kd * delta_v,
        jnp.zeros(3)
    )
    return thrust

def jax_central_gravity(r):
    return -const.MU_EARTH_KM / jnp.linalg.norm(r)**3 * r

def jax_j2_perturbation(r):
    x, y, z = r
    r_norm = jnp.linalg.norm(r)
    pre_factor = -1.5 * const.J2 * const.MU_EARTH_KM * const.R_EARTH_KM**2 / r_norm**5
    z_factor = 5 * z**2 / r_norm**2
    ax = pre_factor * x * (1 - z_factor)
    ay = pre_factor * y * (1 - z_factor)
    az = pre_factor * z * (3 - z_factor)
    return jnp.array([ax, ay, az])

def jax_third_body_perturbation(r, k, sun_pos, moon_pos):
    r_sun = sun_pos[k]
    r_moon = moon_pos[k]
    
    r_sat_sun = r_sun - r
    a_sun = const.MU_SUN_KM * (r_sat_sun / jnp.linalg.norm(r_sat_sun)**3 - r_sun / jnp.linalg.norm(r_sun)**3)
    r_sat_moon = r_moon - r
    a_moon = const.MU_MOON_KM * (r_sat_moon / jnp.linalg.norm(r_sat_moon)**3 - r_moon / jnp.linalg.norm(r_moon)**3)
    return a_sun + a_moon

def jax_srp_perturbation(r, k, sat_params, sun_pos):
    C_R, A, m = sat_params['C_R'], sat_params['A'], sat_params['m']
    r_sun = sun_pos[k]
    r_sat_sun = r_sun - r
    return -const.P_R_KM_COEFF * C_R * A / m * (r_sat_sun / jnp.linalg.norm(r_sat_sun))

def jax_get_total_acceleration(s, s_ref, k, sun_pos, moon_pos, sat_params, policy_params):
    r = s[:3]
    
    a_gravity = jax_central_gravity(r)
    a_j2 = jax_j2_perturbation(r)
    a_3b = jax_third_body_perturbation(r, k, sun_pos, moon_pos)
    a_srp = jax_srp_perturbation(r, k, sat_params, sun_pos)
    
    
    a_policy = jax_diff_control_policy(s, s_ref, **policy_params)
    # a_policy = jax_control_policy(s, s_ref, **policy_params)
    
    return a_gravity + a_j2 + a_3b + a_srp + a_policy
    

# --- 2. DATA GENERATION ---
def generate_reference_trajectory(s0: np.ndarray, t_span: tuple, dt: float) -> tuple:
    def keplerian_rhs(t, s):
        r, v = s[:3], s[3:]
        a = -const.MU_EARTH_KM / np.linalg.norm(r)**3 * r
        return np.hstack([v, a])
    
    times = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(fun=keplerian_rhs, t_span=t_span, y0=s0, t_eval=times, rtol=1e-9, atol=1e-12)
    return sol.t, sol.y.T


# --- 3. NUMPYRO BAYESIAN INFERENCE MODEL ---
def sde_model(times, observed_states, ref_states, sun_pos, moon_pos, sat_params):

    logKp  = numpyro.sample("logKp",  dist.Normal(jnp.log(5e-2), 10.0)) 
    logKd  = numpyro.sample("logKd",  dist.Normal(jnp.log(5e-1), 10.0))
    logGam = numpyro.sample("logGam", dist.Normal(jnp.log(10), 10.0))
    logSigma = numpyro.sample("logSigma", dist.Normal(jnp.log(1e-3), 10.0))
    Kp     = numpyro.deterministic("Kp", jnp.exp(logKp))
    Kd     = numpyro.deterministic("Kd", jnp.exp(logKd))
    gamma  = numpyro.deterministic("gamma", jnp.exp(logGam))
    sigma  = numpyro.deterministic("sigma", jnp.exp(logSigma))
    logBeta = numpyro.sample("logBeta", dist.Normal(jnp.log(1.0), 1.0)) # Example prior
    beta = numpyro.deterministic("beta", jnp.exp(logBeta))

    # sigma = 1e-5
    # Kp = 5e-4
    # Kd = 5e-3
    # gamma = 5
    # beta = 0.5


    policy_params = {"Kp": Kp, "Kd": Kd, "gamma": gamma, "beta": beta}

    vmap_accel = vmap(jax_get_total_acceleration, in_axes=(0, 0, 0, None, None, None, None))

    v_k_plus_1 = observed_states[1:, 3:]
    v_k = observed_states[:-1, 3:]
    r_k = observed_states[:-1, :3]
    dt = times[1:] - times[:-1]

    r_half = r_k + 0.5 * v_k * dt[:, None]
    s_half = jnp.concatenate([r_half, v_k], axis=1)

    t_idx = jnp.arange(times.shape[0] - 1, dtype=jnp.int32)
    a_half = vmap_accel(s_half, ref_states[:-1], t_idx, sun_pos, moon_pos, sat_params, policy_params)

    predicted_v = v_k + a_half * dt[:, None]
    velocity_innovations = v_k_plus_1 - predicted_v


    numpyro.deterministic("predicted_v", predicted_v)
    numpyro.deterministic("a_half_step", a_half)

    N = times.shape[0] - 1
    dt = times[1:] - times[:-1]               # (N,)
    innov = velocity_innovations              # (N, 3)
    scale_t = sigma * jnp.sqrt(dt)      # (N,)

    with numpyro.plate("time", N, dim=-2):    # aligns with axis -2 of (N,3)
        with numpyro.plate("coord", 3, dim=-1):
            numpyro.sample(
                "obs",
                dist.Normal(0.0, scale_t[:, None]),   # (N,1) broadcasts to (N,3)
                obs=innov,
            )


# --- 4. EXECUTION ---
def run_bayesian_inference():
    print("--- 1. Generating Ground Truth Data ---")
    duration_days = 1.0
    dt = 60.0 
    t_span = (0, duration_days * 24 * 3600)

    # These are our "true" parameters that we'll try to recover
    true_satellite_params = {'C_R': 1.5, 'A': 20.0, 'm': 1000.0}
    true_policy_params = {
        'gamma': 100.0,
        'Kp': 5e-4,
        'Kd': 5e-3,
        'beta': 0.5 
    }
    true_sigma = 1e-5

    # Generate the data
    simulator = OrbitSimulator(
        satellite_params=true_satellite_params,
        policy_params=true_policy_params,
        sigma=true_sigma
    )
    ref_times, ref_states = generate_reference_trajectory(const.S0_KM, t_span, dt)
    
    sim_times, sim_states, _ = simulator.simulate_SLV(const.S0_KM, t_span, dt, ref_states)

    print("Data generation complete.")
    
    # Prepare data for NumPyro (convert to JAX arrays)
    observed_states = jnp.array(sim_states)
    times = jnp.array(sim_times)
    ref_states = jnp.array(ref_states)
    sun_pos = jnp.array(simulator.sun_pos_km.T)
    moon_pos = jnp.array(simulator.moon_pos_km.T)

    print("\n--- 2. Running Bayesian Inference with NumPyro ---")
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_ = jax.random.split(rng_key)
    
    # Set up the MCMC sampler
    kernel = NUTS(sde_model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
    
    # Run the MCMC sampler
    mcmc.run(
        rng_key_,
        times=times,
        observed_states=observed_states,
        ref_states=ref_states,
        sun_pos=sun_pos,
        moon_pos=moon_pos,
        sat_params=true_satellite_params
    )
    
    print("\n--- 3. Inference Results ---")

    return mcmc
  


if __name__ == "__main__":
    run_bayesian_inference()