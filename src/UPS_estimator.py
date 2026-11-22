# Here we implement the UPS estimator for a given path or ditribution

import jax
import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm
from src.RWMH_coupling import *
from src.umcmc_estimator import *
from src.tuning_umcmc import *



# Given the tuned parametes k, m, lags and the log_target_path we estimate the the second moment with estimate_moment 

# Now we implement E(lambda)


def E_lambda(key, lam, lambda_grid, k_grid, m_grid, lag, mean_init, Sigma_init, Sigma, cholesky_sigma, log_target_path, grad_log_target_path):

    # --- find the nearest lambda to the grid ---

    index_lambda = jnp.argmin(jnp.abs(lambda_grid - lam))
    k_lam = int(k_grid[index_lambda])
    m_lam = int(m_grid[index_lambda])
    lag_array = jnp.asarray(lag)
    lag_lam = int(lag_array) if lag_array.ndim == 0 else int(lag_array[index_lambda])

    # --- log_target spécifique ---
    log_target = lambda x: log_target_path(x, path=lam)
    grad_log_target = lambda x: grad_log_target_path(x, path=lam)

    # --- initialisation ---
    def ri(key): 
        return initial_distribution(key, log_target, mean_init, Sigma_init)
    
    def sk(key, x, f):
        return MH_kernel(
            key, x, f, Sigma, log_target, cholesky_sigma
        )
    
    def ck(key, x1, f1, x2, f2):
        return MH_coupled_kernel(key, x1, f1, x2, f2, cholesky_sigma, log_target)
    

    # --- appel estimateur ---
    umcmc_estimator = unbiased_estimator(
        key, sk, ck, ri,
        h=[lambda x: grad_log_target(x)],
        k=k_lam,
        m=m_lam,
        lag=lag_lam
    )

    return umcmc_estimator["uestimator"][0], umcmc_estimator["cost"]



# Having the estimates sqrt_m2 one can construct the approximation of the optimal proposal distribution



@jax.jit
def optimal_q(key, lambda_grid, m2):

    sqrt_m2 = jnp.sqrt(jnp.maximum(m2, 0.0))

    # Trapezoidal approximation
    widths = jnp.diff(lambda_grid)
    heights = 0.5 * (sqrt_m2[:-1] + sqrt_m2[1:])

    weights = widths * heights
    weights = weights / jnp.sum(weights)

    # CDF
    cdf = jnp.cumsum(weights)

    # Sample U and V
    key1, key2 = jax.random.split(key)

    u = jax.random.uniform(key1)
    v = jax.random.uniform(key2)

    # Binary search
    i = jnp.searchsorted(cdf, u, side="right")
    i = jnp.minimum(i, lambda_grid.shape[0] - 2)

    # Sample lambda inside interval
    lam = lambda_grid[i] + v * widths[i]

    # Piecewise constant density
    q_lam = weights[i] / widths[i]

    return lam, q_lam

# We implement the UPS estimator as follows:

def UPS_estimator(key, lambda_grid, m2, k_grid, m_grid, lag, mean_init, Sigma_init, Sigma, cholesky_Sigma, log_target_path, grad_log_target_path, M):

    key1, key2 = jax.random.split(key)

    subkeys_q = jax.random.split(key1, M)
    subkeys_E = jax.random.split(key2, M)

    # Sample lambda ~ q
    lambdas, q_lambdas = jax.vmap(optimal_q, in_axes=(0, None, None))(subkeys_q, lambda_grid, m2)

    E_lambda_results = [
        E_lambda(
            subkey, lam, lambda_grid, k_grid, m_grid, lag,
            mean_init, Sigma_init, Sigma, cholesky_Sigma, log_target_path,
            grad_log_target_path
        )
        for subkey, lam in zip(subkeys_E, lambdas)
    ]
    E_lambda_estimates = jnp.stack([result[0] for result in E_lambda_results])
    total_cost = jnp.sum(jnp.stack([result[1] for result in E_lambda_results]))

    # Importance sampling
    estimates = E_lambda_estimates / q_lambdas

    return jnp.mean(estimates), total_cost





