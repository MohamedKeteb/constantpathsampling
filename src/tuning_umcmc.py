import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm
from src.RWMH_coupling import *
from src.umcmc_estimator import *

# Tuning of the parametrs k and m of Unbiased MCMC estimatiors for a given grid


import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random
from tqdm import tqdm


def tune_km_grid(
    key,
    lambda_grid,
    Sigma,
    mean_init,
    Sigma_init,
    log_target_path,
    lag=1,
    nrep=100
):

    k_list = []
    meantau_list = []
    meetings_list = []

    for ilambda, lam in enumerate(tqdm(lambda_grid, desc="mk tuning")):

        log_target = lambda x: log_target_path(x, path=lam)

        def ri(key):
            return initial_distribution(
                key, log_target, mean_init, Sigma_init
            )

        def sk(key, x, f):
            return MH_kernel(
                key, x, f, Sigma, log_target
            )

        def ck(key, x1, f1, x2, f2):
            return MH_coupled_kernel(
                key, x1, f1, x2, f2, Sigma, log_target
            )

        # One coupled-chain simulation
        def one_replication(key):
            tau = coupled_chain(key, sk, ck, ri, 0, lag)["meetingtime"]

            return tau

        # Generate independent keys
        key, subkey = random.split(key)
        keys = random.split(subkey, nrep)

        meetings = jnp.stack([one_replication(rep_key) for rep_key in keys])

        # Compute statistics using JAX
        k_lambda = jnp.floor(jnp.quantile(meetings, 0.99)).astype(jnp.int32)
        mean_tau = jnp.mean(meetings)

        k_list.append(k_lambda)
        meantau_list.append(mean_tau)

        meetings_list.append(
            pd.DataFrame({
                "ilambda": ilambda,
                "lambda": float(lam),
                "meetings": np.asarray(
                    jax.device_get(meetings)
                )
            })
        )

    # Collect results
    k_grid = jnp.stack(k_list)
    meantau = jnp.stack(meantau_list)

    # Compute m_grid
    m_grid = (
        10 * jnp.max(k_grid)
        + jnp.floor(jnp.max(meantau))
        - jnp.floor(meantau)
    ).astype(jnp.int32)

    meetings_df = pd.concat(
        meetings_list,
        ignore_index=True
    )

    return {
        "k_grid": k_grid,
        "m_grid": m_grid,
        "meetings_df": meetings_df,
        "meantau": meantau,
        "key": key
    }




# One can visualize the tuning witht the following function

def plot_km_grid(key, lambda_grid, Sigma, mean_init, Sigma_init, log_target_path):

    results = tune_km_grid(key, lambda_grid, Sigma, mean_init, Sigma_init, log_target_path, lag=1, nrep=100)

    k_grid = np.asarray(jax.device_get(results["k_grid"]))
    m_grid = np.asarray(jax.device_get(results["m_grid"]))
    meetings_df = results["meetings_df"]
    key = results["key"]

    lambda_np = np.asarray(lambda_grid)

    line_df = pd.DataFrame({
        "lambda": lambda_np,
        r"$k_\lambda$ (99% quantile)": k_grid,
        r"$m_\lambda$ (constant cost)": m_grid
    }).melt(id_vars="lambda", var_name="Parameter", value_name="Value")

    sns.set_theme(style="whitegrid", context="paper", rc={
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#4D4D4D",
        "axes.linewidth": 0.8,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.8,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), dpi=150)

    sns.violinplot(data=meetings_df, x="lambda", y="meetings", inner=None, cut=0,
                   color="#DCEAF4", linewidth=0.8, ax=axes[0])

    sns.lineplot(x=np.arange(len(lambda_np)), y=k_grid, color="#0072B2",
                 marker="o", linewidth=2.2, markersize=6.5, ax=axes[0],
                 label=r"$k_\lambda$")

    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("Meeting time")
    axes[0].set_title(r"Meeting-time distribution and tuned $k$ for each $\lambda$")
    axes[0].grid(True, axis="y")
    axes[0].grid(False, axis="x")
    sns.despine(ax=axes[0], top=True, right=True)
    axes[0].legend(frameon=False, loc="upper left")

    sns.lineplot(data=line_df, x="lambda", y="Value", hue="Parameter",
                 style="Parameter", markers=True, dashes=False, linewidth=2.2,
                 markersize=6.5, palette=["#0072B2", "#D55E00"], ax=axes[1])

    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel("Value")
    axes[1].set_title(r"Tuned $k$ and $m$ for each $\lambda$")
    axes[1].set_xlim(np.min(lambda_np), np.max(lambda_np))
    axes[1].grid(True, axis="both")
    sns.despine(ax=axes[1], top=True, right=True)
    axes[1].legend(title=None, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.92))

    plt.tight_layout()
    plt.show()

    return key, results






#  we can tune the initialisation idstribution by computing the unbiased estimator of the mean and the covaraince


def estimate_mom(
    lambda_grid,
    k_grid,
    m_grid,
    Sigma,
    mean_init,
    Sigma_init,
    log_target_path,
    lag=1,
    key=None
):

    if key is None:
        key = jax.random.PRNGKey(0)

    h_list = [
        lambda x: x,
        lambda x: jnp.outer(x, x)
    ]

    d = len(mean_init)
    n_lambda = len(lambda_grid)

    m1 = jnp.zeros((n_lambda, d))
    m2 = jnp.zeros((n_lambda, d, d))
    covariance = jnp.zeros((n_lambda, d, d))

    for ilambda, lam in enumerate(
        tqdm(lambda_grid, desc="tuning of initialisation distribution")
    ):

        log_target = lambda x: log_target_path(x, path=lam)

        def ri(rep_key):
            return initial_distribution(rep_key, log_target, mean_init, Sigma_init)

        def sk(rep_key, x, f):
            return MH_kernel(rep_key, x, f, Sigma, log_target)

        def ck(rep_key, x1, f1, x2, f2):
            return MH_coupled_kernel(rep_key, x1, f1, x2, f2, Sigma, log_target)

        key, subkey = random.split(key)

        res = unbiased_estimator(
            subkey, sk, ck, ri,
            h_list,
            k=int(k_grid[ilambda]),
            m=int(m_grid[ilambda]),
            lag=lag
        )

        mom1_estimators = res["uestimator"][0]
        mom2_estimators = res["uestimator"][1]

        cov = mom2_estimators - jnp.outer(
            mom1_estimators,
            mom1_estimators
        )

        m1 = m1.at[ilambda].set(mom1_estimators)
        m2 = m2.at[ilambda].set(mom2_estimators)
        covariance = covariance.at[ilambda].set(cov)

    return {
        "m1": m1,
        "m2": m2,
        "covariance": covariance,
        "key": key
    }





