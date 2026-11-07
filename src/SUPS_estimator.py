# Here we implement the SUPS estimator for a given path or ditribution
import jax
import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm
from src.RWMH_coupling import *
from src.umcmc_estimator import *
from src.tuning_umcmc import *
from src.UPS_estimator import *

# Budget is the number od samples within each bin


## We begin by estimating the variance of E_lambda within each bin as follows: 

def estimate_sigma(key, lambda_grid, k_grid, m_grid, lag, mean_init, Sigma_init, Sigma, log_target_path, grad_log_target_path, N_pilot):

    L = lambda_grid.shape[0] - 1

    keys = random.split(key, L)

    widths = jnp.diff(lambda_grid)

    variances = []

    for ell in range(L):

        # Uniform pilot sampling in bin ell
        key_lambda, key_E = random.split(keys[ell])

        lambdas = random.uniform(key_lambda, shape=(N_pilot,), minval=lambda_grid[ell], maxval=lambda_grid[ell + 1])

        # Independent keys for E_lambda
        subkeys_E = random.split(key_E, N_pilot)

        # Monte Carlo evaluation of E_lambda
        E_values = jnp.stack([
            E_lambda(
                subkeys_E[j],
                lambdas[j],
                lambda_grid, k_grid, m_grid, lag,
                mean_init, Sigma_init, Sigma,
                log_target_path, grad_log_target_path
            )
            for j in range(N_pilot)
        ])

        # Empirical variance
        var_ell = jnp.var(E_values, ddof=1)

        variances.append(var_ell)

    variances = jnp.stack(variances)

    # Standard deviations
    sigma = jnp.sqrt(variances)

    return sigma


# now we implement the neyman allocation with multinomial distribution of the risiduals


## Before we implement the multinomail law 

def multinomial_sample(key, n, p):
    keys = jax.random.split(key, n)

    samples = jax.vmap(
        lambda k: jax.random.categorical(k, jnp.log(p))
    )(keys)

    return jnp.bincount(samples, length=len(p))


def multinomial_sample(key, n, p):

    L = p.shape[0]

    # Generate L categorical samples (fixed shape)
    samples = jax.random.categorical(
        key, jnp.log(p), shape=(L,)
    )

    # Keep only the first n samples
    active = (jnp.arange(L) < n).astype(jnp.int32)

    # Count occurrences
    counts = jnp.bincount(
        samples,
        weights=active,
        length=L
    )

    return counts.astype(jnp.int32)



def neyman_allocation(key, lambda_grid, sigma, M):

    # Number of bins
    L = lambda_grid.shape[0] - 1

    # Bin widths
    widths = jnp.diff(lambda_grid)

    # Neyman allocation weights
    scores = widths * sigma

    # Normalize allocation weights
    probs = scores / jnp.sum(scores)

    # Remaining budget after assigning 1 to every bin
    remaining_budget = M - L

    # Ideal continuous allocation
    ideal = remaining_budget * probs

    # Integer part
    base = jnp.floor(ideal).astype(jnp.int32)

    # Fractional remainders
    residual = ideal - base

    # Number of samples still to allocate
    R = remaining_budget - jnp.sum(base)

    # Multinomial residual allocation
    residual_probs = residual / jnp.sum(residual)

    residual_counts = multinomial_sample(key, R, residual_probs).astype(jnp.int32)

    # Final allocation: at least one sample per bin
    budget = 1 + base + residual_counts

    return budget
 





def SUPS_estimator(key, lambda_grid, budget, k_grid, m_grid, lag, mean_init, Sigma_init, Sigma, log_target_path, grad_log_target_path):

    L = len(budget)
    # Compute all bin widths
    widths = jnp.diff(lambda_grid)

    # Generate independent keys for each bin
    keys = random.split(key, L)

    estimates_per_bin = []

    for ell in range(L):

        M_ell = budget[ell]

        # Uniform sampling inside bin
        lambdas = random.uniform(keys[ell], shape=(M_ell,), minval=lambda_grid[ell], maxval=lambda_grid[ell + 1])

        subkeys_E = random.split(random.fold_in(keys[ell], 1), M_ell)

        E_values = jnp.stack([
            E_lambda(
                subkeys_E[j],
                lambdas[j],
                lambda_grid, k_grid, m_grid, lag,
                mean_init, Sigma_init, Sigma,
                log_target_path, grad_log_target_path
            )
            for j in range(M_ell)
        ])

        # Monte Carlo contribution
        bin_estimate = widths[ell] * jnp.mean(E_values)

        estimates_per_bin.append(bin_estimate)

    return jnp.sum(jnp.stack(estimates_per_bin))

