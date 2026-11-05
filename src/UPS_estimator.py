# Here we implement the UPS estimator for a given path or ditribution
import jax
import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm
from src.RWMH_coupling import *
from src.umcmc_estimator import *
from src.tuning_umcmc import *



# Given the tuned parametes k, m, lags and the log_target_path we estimate the the second moment 


def sqrt_m2(
    lambda_grid,
    k_grid,
    m_grid,
    Sigma,
    mean_init,
    Sigma_init,
    log_target_path,
    lag=1,
    nrep=10,
    key=None
):

    if key is None:
        key = jax.random.PRNGKey(0)

    n_lambda = len(lambda_grid)
    sqrt_m2 = jnp.zeros(n_lambda)

    for ilambda, lam in enumerate(tqdm(lambda_grid, desc="Estimation sqrt(m2)")):

        log_target = lambda x: log_target_path(x, path=lam)
        h_list = [lambda x: log_target_path(x, path=lam)]

        def ri(key):
            return initial_distribution(key, log_target, mean_init, Sigma_init)

        def sk(key, x, f):
            return MH_kernel(key, x, f, Sigma, log_target)

        def ck(key, x1, f1, x2, f2):
            return MH_coupled_kernel(key, x1, f1, x2, f2, Sigma, log_target)

        def one_replication(key):
            res = unbiased_estimator(
                key, sk, ck, ri,
                h_list,
                k=int(k_grid[ilambda]),
                m=int(m_grid[ilambda]),
                lag=lag
            )

            return res["uestimator"][0]

        # Generate independent random keys
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, nrep)

        estimators = jnp.stack([one_replication(rep_key) for rep_key in keys])

        # Estimate sqrt(E[I_lambda^2])
        sqrt_m2_lambda = jnp.sqrt(jnp.mean(estimators**2))
        sqrt_m2 = sqrt_m2.at[ilambda].set(sqrt_m2_lambda)

    return sqrt_m2






