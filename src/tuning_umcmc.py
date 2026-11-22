import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotnine as p9
import jax
import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm
from src.RWMH_coupling import *
from src.umcmc_estimator import *

# Tuning of the parametrs k and m of Unbiased MCMC estimatiors for a given grid




def tune_km_grid(
    key,
    lambda_grid,
    mean_init,
    Sigma_init,
    Sigma, 
    cholesky_sigma,
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
                key, x, f, Sigma, log_target, cholesky_sigma
            )

        def ck(key, x1, f1, x2, f2):
            return MH_coupled_kernel(
                key, x1, f1, x2, f2, cholesky_sigma, log_target
            )

        # One coupled-chain simulation
        def one_replication(key):
            tau = coupled_chain(key, sk, ck, ri, 0, lag)["meetingtime"]

            return tau

        # Generate independent keys
        keys = random.split(key, nrep)

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
        "meantau": meantau
    }




# One can visualize the tuning witht the following function

# results = tune_km_grid(key, lambda_grid, Sigma, mean_init, Sigma_init, log_target_path, lag=1, nrep=100)

def plot_km_grid(results, lambda_grid):

    k_grid = np.asarray(jax.device_get(results["k_grid"]))
    lambda_np = np.asarray(lambda_grid)
    meetings_df = results["meetings_df"].copy()
    k_df = pd.DataFrame({"lambda": lambda_np, "k": k_grid})
    k_df["series"] = "k"
    violin_width = 0.8 * np.min(np.diff(lambda_np))

    plot = (
        p9.ggplot()
        + p9.geom_violin(
            data=meetings_df,
            mapping=p9.aes(x="lambda", y="meetings", group="lambda"),
            width=violin_width,
            fill="white",
            color="black",
            size=0.35,
        )
        + p9.geom_line(
            data=k_df,
            mapping=p9.aes(x="lambda", y="k", color="series"),
            inherit_aes=False,
            size=0.6,
        )
        + p9.geom_point(
            data=k_df,
            mapping=p9.aes(x="lambda", y="k", color="series"),
            inherit_aes=False,
            fill="white",
            size=1.5,
        )
        + p9.scale_color_manual(
            values={"k": "black"},
            labels={"k": r"$\boldsymbol{k_\lambda}$"},
        )
        + p9.scale_x_continuous(
            breaks=np.linspace(lambda_np.min(), lambda_np.max(), 5)
        )
        + p9.labs(x=r"$\lambda$", y="Meeting time", color="", title=None)
        + p9.theme_classic(base_size=9)
        + p9.theme(
            figure_size=(5, 3),
            legend_position=(0.02, 0.98),
            legend_justification=(0, 1),
            legend_direction="horizontal",
            legend_text=p9.element_text(size=14),
            plot_background=p9.element_rect(fill="white", color="white"),
            panel_background=p9.element_rect(fill="white", color="white"),
        )
    )
    return plot




def estimate_moments(key,
    lambda_grid,
    k_grid,
    m_grid,
    mean_init,
    Sigma_init,
    Sigma, 
    cholesky_sigma, 
    log_target_path,
    grad_log_target_path,
    lag=1,
    nrep=20):

    from src.UPS_estimator import E_lambda


    n_lambda = len(lambda_grid)
    m1 = jnp.zeros(n_lambda)
    m2 = jnp.zeros(n_lambda)
    var = jnp.zeros(n_lambda)
    lambda_keys = random.split(key, n_lambda)

    for ilambda, lam in enumerate(tqdm(lambda_grid, desc="estimating moments")):
        replication_keys = random.split(lambda_keys[ilambda], nrep)
        estimates = []

        for rep_key in replication_keys:
            estimate, _ = E_lambda(
                rep_key,
                lam,
                lambda_grid,
                k_grid,
                m_grid,
                lag,
                mean_init,
                Sigma_init,
                Sigma,
                cholesky_sigma,
                log_target_path,
                grad_log_target_path,
            )
            estimates.append(estimate)

        estimates = jnp.stack(estimates)
        first_moment = jnp.mean(estimates)
        second_moment = jnp.mean(estimates**2)

        m1 = m1.at[ilambda].set(first_moment)
        m2 = m2.at[ilambda].set(second_moment)
        var = var.at[ilambda].set(second_moment - first_moment**2)

    return {"m1": m1, "m2": m2, "var": var}

def plot_m2_var(results, lambda_grid):
    lambda_values = np.asarray(lambda_grid)
    labels = {
        r"$\sqrt{\widehat{m}_2(\lambda)}$": np.asarray(
            jax.device_get(jnp.sqrt(results["m2"]))
        ),
        r"$\widehat{m}_1(\lambda)$": np.asarray(jax.device_get(results["m1"])),
    }
    data = pd.DataFrame({"lambda": lambda_values, **labels})
    data = data.melt(
        id_vars="lambda",
        var_name="Quantity",
        value_name="Value",
    )
    data["Quantity"] = pd.Categorical(
        data["Quantity"],
        categories=list(labels),
        ordered=True,
    )

    plot = (
        p9.ggplot(data, p9.aes(x="lambda", y="Value", group="Quantity"))
        + p9.geom_line(color="black", size=0.6)
        + p9.geom_point(color="black", fill="white", size=1.5)
        + p9.facet_wrap("~Quantity", ncol=1, scales="free_y")
        + p9.labs(x=r"$\lambda$", y="", title=None)
        + p9.theme_classic(base_size=11)
        + p9.theme(
            figure_size=(5, 3),
            strip_background=p9.element_blank(),
            strip_text=p9.element_text(size=11),
            axis_title_x=p9.element_text(size=11),
            axis_title_y=p9.element_blank(),
            panel_background=p9.element_rect(fill="white", color="white"),
            plot_background=p9.element_rect(fill="white", color="white"),
        )
    )
    return plot


















#  we can tune the initialisation idstribution by computing the unbiased estimator of the mean and the covaraince

def tune_initial_distribution(
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
    cholesky_sigma = cholesky(Sigma)

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
            return MH_kernel(
                rep_key, x, f, Sigma, log_target, cholesky_sigma
            )

        def ck(rep_key, x1, f1, x2, f2):
            return MH_coupled_kernel(
                rep_key, x1, f1, x2, f2, cholesky_sigma, log_target
            )

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

